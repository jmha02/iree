//===- FlexiNPUTensorize.cpp - Lower Linalg to FlexiNPU ----------------*- C++ -*-===//
//
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/LLVMCPU/RoCC/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/RoCC/Utils/Utils.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FLEXINPUTENSORIZEPASS
#include "iree/compiler/Codegen/LLVMCPU/RoCC/Passes.h.inc"

namespace {

  static Value idxFromMixed(OpFoldResult ofr, PatternRewriter &rewriter, Location loc) {
    if (auto v = ofr.dyn_cast<Value>()) return v;
    if (auto a = ofr.dyn_cast<Attribute>()) {
      if (auto ia = dyn_cast<IntegerAttr>(a)) {
        return rewriter.create<arith::ConstantIndexOp>(loc, ia.getInt());
      }
    }
    return rewriter.create<arith::ConstantIndexOp>(loc, 0);
  }

  static bool isFlexiNPUInt4ElementType(Type type) {
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType)
      return false;
    return intType.getWidth() == 4 || intType.getWidth() == 8;
  }

  static int64_t flexiTileDim(Type elemTy) {
    return isFlexiNPUInt4ElementType(elemTy) ? 128 : 32;
  }

  static int64_t flexiGemmElemsPerRow(Type elemTy, int64_t tileDim) {
    if (isFlexiNPUInt4ElementType(elemTy))
      return (tileDim + 1) / 2;
    return tileDim;
  }

  static flexinpu::FlexiNPUTypes flexiNpuDtypeForMatmul(
      Operation *anchor, Type elemTy) {
    auto dtype = convertFlexiNPUType(elemTy);
    if (dtype != flexinpu::FlexiNPUTypes::i8)
      return dtype;
    if (auto target = IREE::HAL::ExecutableTargetAttr::lookup(anchor)) {
      if (hasFlexiNPUInt4Feature(target.getConfiguration()))
        return flexinpu::FlexiNPUTypes::i4;
    }
    return dtype;
  }

  static Value subview2D(PatternRewriter &rewriter, Location loc, Value source,
                         int64_t offset0, int64_t offset1, int64_t size0,
                         int64_t size1) {
    SmallVector<OpFoldResult> offsets = {
        rewriter.getIndexAttr(offset0), rewriter.getIndexAttr(offset1)};
    SmallVector<OpFoldResult> sizes = {
        rewriter.getIndexAttr(size0), rewriter.getIndexAttr(size1)};
    SmallVector<OpFoldResult> strides = {
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    return rewriter.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                              strides).getResult();
  }

  static int64_t getConstStride(MemRefType ty, int dim) {
    auto so = ty.getStridesAndOffset();
    const auto &strides = so.first;
    if (strides.empty())
      return -1;
    int64_t s = strides[dim];
    return (s >= 1) ? s : -1;
  }

  static int64_t getStaticOrTiledDim(MemRefType ty, int dim) {
    if (!ty.isDynamicDim(dim))
      return ty.getDimSize(dim);
    // IREE's inner tiled matmul can materialize as memref<32x?xT> with a
    // fixed row stride equal to the inner N tile, e.g. layer3 N=196 -> 28.
    // FlexiNPU commands need immediate tile sizes, so recover that bounded
    // inner tile size from the row stride.
    if (dim == 1) {
      int64_t rowStride = getConstStride(ty, 0);
      if (rowStride > 0 && rowStride <= 32)
        return rowStride;
    }
    return -1;
  }

  static Value castFloatValue(PatternRewriter &rewriter, Location loc, Value value,
                              Type dstType) {
    Type srcType = value.getType();
    if (srcType == dstType)
      return value;
    unsigned srcWidth = srcType.getIntOrFloatBitWidth();
    unsigned dstWidth = dstType.getIntOrFloatBitWidth();
    if (srcWidth < dstWidth)
      return rewriter.create<arith::ExtFOp>(loc, dstType, value);
    return rewriter.create<arith::TruncFOp>(loc, dstType, value);
  }

  static void copyAndCast2D(PatternRewriter &rewriter, Location loc, Value src,
                            Value dst, Type dstElemTy, int64_t rows,
                            int64_t cols, bool accumulate) {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rowEnd = rewriter.create<arith::ConstantIndexOp>(loc, rows);
    Value colEnd = rewriter.create<arith::ConstantIndexOp>(loc, cols);

    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, rowEnd, c1);
    {
      OpBuilder::InsertionGuard rowGuard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value row = rowLoop.getInductionVar();
      auto colLoop = rewriter.create<scf::ForOp>(loc, c0, colEnd, c1);
      {
        OpBuilder::InsertionGuard colGuard(rewriter);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value col = colLoop.getInductionVar();
        Value loaded = rewriter.create<memref::LoadOp>(loc, src, ValueRange{row, col});
        Value casted = castFloatValue(rewriter, loc, loaded, dstElemTy);
        if (accumulate) {
          Value old =
              rewriter.create<memref::LoadOp>(loc, dst, ValueRange{row, col});
          casted = rewriter.create<arith::AddFOp>(loc, old, casted);
        }
        rewriter.create<memref::StoreOp>(loc, casted, dst, ValueRange{row, col});
      }
    }
  }

  static void fillZero(PatternRewriter &rewriter, Location loc, Value memref,
                       Type elemTy) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemTy, rewriter.getZeroAttr(elemTy));
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{memref});
  }

  static bool isDimPair(AffineMap map, int64_t first, int64_t second) {
    if (map.getNumResults() != 2)
      return false;
    auto lhs = dyn_cast<AffineDimExpr>(map.getResult(0));
    auto rhs = dyn_cast<AffineDimExpr>(map.getResult(1));
    return lhs && rhs && lhs.getPosition() == first && rhs.getPosition() == second;
  }

  static LogicalResult emitTiledFlexiMatmul(
      Operation *anchor, PatternRewriter &rewriter, Location loc, Value matA,
      Value matB, Value matC, bool transposeA, bool transposeB) {
    auto A = dyn_cast<MemRefType>(matA.getType());
    auto B = dyn_cast<MemRefType>(matB.getType());
    auto C = dyn_cast<MemRefType>(matC.getType());

    if (!A || !B || !C) {
      return rewriter.notifyMatchFailure(anchor, "expected memref operands");
    }
    if (A.getRank() != 2 || B.getRank() != 2 || C.getRank() != 2) {
      return rewriter.notifyMatchFailure(anchor, "expected rank-2 operands");
    }

    const int64_t DIM = flexiTileDim(A.getElementType());
    const int64_t GEMM_ROW = flexiGemmElemsPerRow(A.getElementType(), DIM);

    int64_t cM = getStaticOrTiledDim(C, 0);
    int64_t cN = getStaticOrTiledDim(C, 1);
    int64_t dim_M =
        transposeA ? getStaticOrTiledDim(A, 1) : getStaticOrTiledDim(A, 0);
    int64_t dim_K_from_A =
        transposeA ? getStaticOrTiledDim(A, 0) : getStaticOrTiledDim(A, 1);
    int64_t dim_N =
        transposeB ? getStaticOrTiledDim(B, 0) : getStaticOrTiledDim(B, 1);
    int64_t dim_K_from_B =
        transposeB ? getStaticOrTiledDim(B, 1) : getStaticOrTiledDim(B, 0);
    if (dim_M < 1)
      dim_M = cM;
    if (dim_N < 1)
      dim_N = cN;
    if (dim_K_from_A < 1)
      dim_K_from_A = dim_K_from_B;
    if (dim_K_from_B < 1)
      dim_K_from_B = dim_K_from_A;
    if (dim_M < 1 || dim_N < 1 || dim_K_from_A < 1 || dim_K_from_B < 1 ||
        cM < 1 || cN < 1) {
      return rewriter.notifyMatchFailure(anchor,
                                         "could not recover static tile sizes");
    }
    if (dim_K_from_A != dim_K_from_B || cM != dim_M || cN != dim_N) {
      return rewriter.notifyMatchFailure(anchor,
                                         "illegal logical matmul shapes");
    }

    int64_t dim_K = dim_K_from_A;
    auto aRowStride = getConstStride(A, 0);
    auto bRowStride = getConstStride(B, 0);
    auto cRowStride = getConstStride(C, 0);
    if (aRowStride < 1 || bRowStride < 1 || cRowStride < 1) {
      return rewriter.notifyMatchFailure(anchor, "dynamic/invalid row stride");
    }

    int64_t ONC_INTERVAL = 1024 * 1024; // 1048576

    Value oncAddrA = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    Value oncAddrB = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(ONC_INTERVAL));
    Value oncAddrC = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(ONC_INTERVAL * 2));
    Value oncAddrD = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(ONC_INTERVAL * 3));

    auto dtype = flexiNpuDtypeForMatmul(anchor, A.getElementType());
    bool useBf16ResultStaging =
        A.getElementType().isBF16() && C.getElementType().isF32();
    Type npuResultElemTy =
        useBf16ResultStaging ? A.getElementType() : C.getElementType();
    auto res_dtype = flexiNpuDtypeForMatmul(anchor, npuResultElemTy);

    for (int64_t m0 = 0; m0 < dim_M; m0 += DIM) {
      int64_t mAccess = std::min<int64_t>(DIM, dim_M - m0);
      for (int64_t n0 = 0; n0 < dim_N; n0 += DIM) {
        int64_t nAccess = std::min<int64_t>(DIM, dim_N - n0);
        int64_t pad_N = DIM - nAccess;
        Value cTile = subview2D(rewriter, loc, matC, m0, n0, mAccess, nAccess);
        Value npuCTile = cTile;
        int64_t npuCRowStride = cRowStride;
        if (useBf16ResultStaging) {
          auto tempType =
              MemRefType::get({mAccess, nAccess}, npuResultElemTy);
          npuCTile = rewriter.create<memref::AllocaOp>(loc, tempType);
          npuCRowStride = nAccess;
        } else {
          rewriter.create<flexinpu::DmaLoadOp>(
              loc, npuCTile, oncAddrC, nAccess, mAccess, npuCRowStride, false,
              res_dtype, pad_N);

          rewriter.create<flexinpu::PreloadOp>(
              loc, oncAddrC, GEMM_ROW, mAccess, flexinpu::FlexiNPUMatTypes::C,
              res_dtype);
        }

        for (int64_t k0 = 0; k0 < dim_K; k0 += DIM) {
          int64_t kAccess = std::min<int64_t>(DIM, dim_K - k0);
          int64_t pad_K = DIM - kAccess;

          if (useBf16ResultStaging) {
            fillZero(rewriter, loc, npuCTile, npuResultElemTy);
            rewriter.create<flexinpu::DmaLoadOp>(
                loc, npuCTile, oncAddrC, nAccess, mAccess, npuCRowStride,
                false, res_dtype, pad_N);
            rewriter.create<flexinpu::PreloadOp>(
                loc, oncAddrC, GEMM_ROW, mAccess,
                flexinpu::FlexiNPUMatTypes::C, res_dtype);
          }

          Value aTile = transposeA
                            ? subview2D(rewriter, loc, matA, k0, m0, kAccess,
                                        mAccess)
                            : subview2D(rewriter, loc, matA, m0, k0, mAccess,
                                        kAccess);
          Value bTile = transposeB
                            ? subview2D(rewriter, loc, matB, n0, k0, nAccess,
                                        kAccess)
                            : subview2D(rewriter, loc, matB, k0, n0, kAccess,
                                        nAccess);

          rewriter.create<flexinpu::DmaLoadOp>(
              loc, bTile, oncAddrB, transposeB ? kAccess : nAccess,
              transposeB ? nAccess : kAccess, bRowStride, !transposeB, dtype,
              pad_K);

          rewriter.create<flexinpu::DmaLoadOp>(
              loc, aTile, oncAddrA, transposeA ? mAccess : kAccess,
              transposeA ? kAccess : mAccess, aRowStride, transposeA, dtype,
              pad_K);

          rewriter.create<flexinpu::PreloadOp>(
              loc, oncAddrB, GEMM_ROW, nAccess, flexinpu::FlexiNPUMatTypes::B,
              dtype);

          rewriter.create<flexinpu::ExecuteS1Op>(
              loc, rewriter.getI32Type(), oncAddrA, GEMM_ROW, mAccess,
              flexinpu::FlexiNPUMatTypes::A, dtype, res_dtype);

          if (useBf16ResultStaging) {
            rewriter.create<flexinpu::FlushOp>(
                loc, oncAddrD, GEMM_ROW, mAccess,
                flexinpu::FlexiNPUMatTypes::D, res_dtype);
            rewriter.create<flexinpu::DmaStoreOp>(
                loc, npuCTile, oncAddrD, GEMM_ROW, mAccess, npuCRowStride,
                false, res_dtype, pad_N);
            copyAndCast2D(rewriter, loc, npuCTile, cTile, C.getElementType(),
                          mAccess, nAccess, /*accumulate=*/true);
          }
        }

        if (!useBf16ResultStaging) {
          rewriter.create<flexinpu::FlushOp>(
              loc, oncAddrD, GEMM_ROW, mAccess, flexinpu::FlexiNPUMatTypes::D,
              res_dtype);
          rewriter.create<flexinpu::DmaStoreOp>(
              loc, npuCTile, oncAddrD, GEMM_ROW, mAccess, npuCRowStride, false,
              res_dtype, pad_N);
        }
      }
    }

    return success();
  }

struct LinalgVecmatToFlexiNPUPattern: public OpRewritePattern<linalg::VecmatOp> {
  using OpRewritePattern<linalg::VecmatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::VecmatOp vecmatOp,
                              PatternRewriter &rewriter) const override {
    // tensor<32xbf16>, tensor<32x10xbf16>) outs(%arg1 : tensor<10xbf16>) -> tensor<10xbf16>
    Location loc = vecmatOp.getLoc();

    Value vecA = vecmatOp.getInputs()[0];
    Value matB = vecmatOp.getInputs()[1];
    Value vecC = vecmatOp.getOutputs()[0];

    auto A = dyn_cast<MemRefType>(vecA.getType());
    auto B = dyn_cast<MemRefType>(matB.getType());
    auto C = dyn_cast<MemRefType>(vecC.getType());

    if (!A || !B || !C) {
      return rewriter.notifyMatchFailure(vecmatOp,
                                         "expected memref operands");
    }

    if (!A.hasStaticShape() || !B.hasStaticShape() || !C.hasStaticShape()) {
      return rewriter.notifyMatchFailure(vecmatOp,
                                         "expected static operand shapes");
    }

    if (A.getShape()[0] != B.getShape()[0] || B.getShape()[1] != C.getShape()[0]) {
      return rewriter.notifyMatchFailure(vecmatOp,
                                         "illegal vecmat operand shapes");
    }

    const int64_t DIM = flexiTileDim(A.getElementType());
    const int64_t GEMM_ROW = flexiGemmElemsPerRow(A.getElementType(), DIM);

    int64_t dim_M = 1;
    int64_t dim_N = B.getShape()[1];
    int64_t dim_K = B.getShape()[0];

    if (dim_K > DIM || dim_N > DIM) {
      return rewriter.notifyMatchFailure(
          vecmatOp, "vecmat FlexiNPU path does not yet tile large K/N");
    }

    int64_t pad_N = std::max<int64_t>(0, DIM - dim_N);
    int64_t pad_K = std::max<int64_t>(0, DIM - dim_K);

    auto aRowStride = dim_K;
    auto bRowStride = dim_N; // N
    auto cRowStride = dim_N; // N

    int64_t ONC_INTERVAL = 1024 * 1024; // 1048576

    Value oncAddrA = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(0));
    Value oncAddrB = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(ONC_INTERVAL));
    Value oncAddrC = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(ONC_INTERVAL * 2));
    Value oncAddrD = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(ONC_INTERVAL * 3));

    auto dtype = flexiNpuDtypeForMatmul(vecmatOp, A.getElementType());
    auto res_dtype = flexiNpuDtypeForMatmul(vecmatOp, C.getElementType());

    rewriter.create<flexinpu::DmaLoadOp>(
        loc, vecC,
        oncAddrC,
        dim_N, // elemsPerRow = MIN(DIM, N - DIM * n)
        dim_M, // rows = MIN(DIM, K - DIM * k)
        cRowStride, // stride = N
        false, // trans = true
        res_dtype, // dtype = mat_c_dtype
        pad_N
    );

    rewriter.create<flexinpu::DmaLoadOp>(
        loc, matB,
        oncAddrB, // oncPtr = onc_addr_mat_b
        dim_N, // n_access
        dim_K, // k_access
        bRowStride, // stride = N
        true, // trans = true
        dtype, // dtype = mat_b_dtype
        pad_K
    );

    rewriter.create<flexinpu::DmaLoadOp>(
        loc, vecA,
        oncAddrA, // oncPtr = onc_addr_mat_a
        dim_K, // elemsPerRow = k_access
        dim_M, // rows = m_access
        aRowStride, // stride = K
        false, // trans = false
        dtype, // dtype = mat_a_dtype
        pad_K
    );

    rewriter.create<flexinpu::PreloadOp>(
        loc,
        oncAddrC, // oncPtr = onc_addr_mat_c
        GEMM_ROW, // elemsPerRow
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::C, // matType = MAT_C
        res_dtype // type = mat_c_dtype
      );


    rewriter.create<flexinpu::PreloadOp>(
        loc,
        oncAddrB, // oncPtr = onc_addr_mat_b
        GEMM_ROW, // elemsPerRow
        dim_N, // rows = k_access
        flexinpu::FlexiNPUMatTypes::B, // matType = MAT_B
        dtype // type = mat_b_dtype
    );

    rewriter.create<flexinpu::ExecuteS1Op>(
        loc,
        rewriter.getI32Type(),
        oncAddrA, // oncPtr = onc_addr_mat_a
        GEMM_ROW, // elemsPerRow
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::A, // matType = MAT_A
        dtype, // type = mat_a_dtype
        res_dtype // res_type = res_dtype
    );

    rewriter.create<flexinpu::FlushOp>(
        loc, oncAddrD,
        GEMM_ROW, // elemsPerRow
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::D, // matType = MAT_D
        res_dtype // type = res_dtype
    );

    rewriter.create<flexinpu::DmaStoreOp>(
        loc, vecC,
        oncAddrD,
        GEMM_ROW, // elemsPerRow
        dim_M, // rows = m_access
        cRowStride, // stride = N
        false, // trans = false
        res_dtype, // type = mat_d_dtype
        pad_N
    );

    rewriter.eraseOp(vecmatOp);
    return success();
  }
};

struct LinalgMatmulToFlexiNPUPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                              PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    Value matA = matmulOp.getInputs()[0];
    Value matB = matmulOp.getInputs()[1];
    Value matC = matmulOp.getOutputs()[0];

    bool transposeA = false;
    bool transposeB = false;
    auto maps = matmulOp.getIndexingMapsArray();
    if (maps.size() == 3) {
      if (isDimPair(maps[0], 0, 2)) {
        transposeA = false;
      } else if (isDimPair(maps[0], 2, 0)) {
        transposeA = true;
      } else {
        return rewriter.notifyMatchFailure(matmulOp,
                                           "unsupported A indexing map");
      }

      if (isDimPair(maps[1], 2, 1)) {
        transposeB = false;
      } else if (isDimPair(maps[1], 1, 2)) {
        transposeB = true;
      } else {
        return rewriter.notifyMatchFailure(matmulOp,
                                           "unsupported B indexing map");
      }

      if (!isDimPair(maps[2], 0, 1)) {
        return rewriter.notifyMatchFailure(matmulOp,
                                           "unsupported C indexing map");
      }
    }

    if (failed(emitTiledFlexiMatmul(matmulOp, rewriter, loc, matA, matB, matC,
                                    transposeA, transposeB))) {
      return failure();
    }

    rewriter.eraseOp(matmulOp);
    return success();
  }
};

class FlexiNPUTensorizePass : public impl::FlexiNPUTensorizePassBase<FlexiNPUTensorizePass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<flexinpu::FlexiNPUDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgMatmulToFlexiNPUPattern>(&getContext());
    patterns.add<LinalgVecmatToFlexiNPUPattern>(&getContext());

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
