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
      llvm::errs() << "A or B or C is not a MemRefType\n";
      return failure();
    }

    if (!A.hasStaticShape() || !B.hasStaticShape() || !C.hasStaticShape()) {
      llvm::errs() << "A or B or C is not a static shape\n";
      return failure();
    }

    if (A.getShape()[0] != B.getShape()[0] || B.getShape()[1] != C.getShape()[0]) {
      llvm::errs() << "Illgeal shape for vecmatOp\n";
      return failure();
    }

    const int64_t DIM = 32; // TODO

    int64_t dim_M = 1;
    int64_t dim_N = B.getShape()[1];
    int64_t dim_K = B.getShape()[0];

    // int64_t pad_M = 1; // not used
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

    auto dtype = convertFlexiNPUType(A.getElementType());
    auto res_dtype = convertFlexiNPUType(C.getElementType());

    rewriter.create<flexinpu::DmaLoadOp>(
        loc, vecC,
        oncAddrC,
        dim_N, // elemsPerRow = MIN(DIM, N - DIM * n)
        dim_M, // rows = MIN(DIM, K - DIM * k)
        cRowStride, // stride = N
        false, // trans = true
        dtype, // dtype = mat_c_dtype
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
        DIM, // elemsPerRow = n_access + n_pad = dim
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::C, // matType = MAT_C
        res_dtype // type = mat_c_dtype
      );


    rewriter.create<flexinpu::PreloadOp>(
        loc,
        oncAddrB, // oncPtr = onc_addr_mat_b
        DIM, // elemsPerRow = n_access + n_pad = dim
        dim_N, // rows = k_access
        flexinpu::FlexiNPUMatTypes::B, // matType = MAT_B
        dtype // type = mat_b_dtype
    );

    rewriter.create<flexinpu::ExecuteS1Op>(
        loc,
        rewriter.getI32Type(),
        oncAddrA, // oncPtr = onc_addr_mat_a
        DIM, // elemsPerRow = k_access + k_pad
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::A, // matType = MAT_A
        dtype, // type = mat_a_dtype
        res_dtype // res_type = res_dtype
    );

    rewriter.create<flexinpu::FlushOp>(
        loc, oncAddrD,
        DIM, // elemsPerRow = DIM
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::D, // matType = MAT_D
        res_dtype // type = res_dtype
    );

    rewriter.create<flexinpu::DmaStoreOp>(
        loc, vecC,
        oncAddrD,
        DIM, // elemsPerRow = DIM
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

    auto A = dyn_cast<MemRefType>(matA.getType());
    auto B = dyn_cast<MemRefType>(matB.getType());
    auto C = dyn_cast<MemRefType>(matC.getType());

    if (!A || !B || !C) {
      llvm::errs() << "A or B or C is not a MemRefType\n";
      return failure();
    }

    if (!A.hasStaticShape() || !B.hasStaticShape() || !C.hasStaticShape()) {
      llvm::errs() << "A or B or C is not a static shape\n";
      return failure();
    }

    const int64_t DIM = 32; // TODO

    int64_t dim_M = A.getShape()[0];
    int64_t dim_K = A.getShape()[1];
    int64_t dim_N = B.getShape()[1];

    int64_t pad_N = std::max<int64_t>(0, DIM - dim_N);
    int64_t pad_K = std::max<int64_t>(0, DIM - dim_K);

    auto getConstStride = [&](MemRefType ty, int dim) -> int64_t {
      auto so = ty.getStridesAndOffset();
      const auto &strides = so.first;
      if (strides.empty()) return -1;
      int64_t s = strides[dim];
      return (s >= 1) ? s : -1;
    };

    auto aRowStride = getConstStride(A, 0); // K
    auto bRowStride = getConstStride(B, 0); // N
    auto cRowStride = getConstStride(C, 0); // N
    if (aRowStride < 1 || bRowStride < 1 || cRowStride < 1) {
      llvm::errs() << "dynamic/invalid row stride\n";
      return rewriter.notifyMatchFailure(matmulOp, "dynamic/invalid row stride");
    }

    int64_t ONC_INTERVAL = 1024 * 1024; // 1048576

    Value oncAddrA = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(0));
    Value oncAddrB = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(ONC_INTERVAL));
    Value oncAddrC = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(ONC_INTERVAL * 2));
    Value oncAddrD = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(ONC_INTERVAL * 3));

    auto dtype = convertFlexiNPUType(A.getElementType());
    auto res_dtype = convertFlexiNPUType(C.getElementType());

    rewriter.create<flexinpu::DmaLoadOp>(
        loc, matC,
        oncAddrC,
        dim_N, // elemsPerRow = MIN(DIM, N - DIM * n)
        dim_M, // rows = MIN(DIM, K - DIM * k)
        cRowStride, // stride = N
        false, // trans = true
        dtype, // dtype = mat_c_dtype
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
        loc, matA,
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
        DIM, // elemsPerRow = n_access + n_pad = dim
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::C, // matType = MAT_C
        res_dtype // type = mat_c_dtype
      );


    rewriter.create<flexinpu::PreloadOp>(
        loc,
        oncAddrB, // oncPtr = onc_addr_mat_b
        DIM, // elemsPerRow = n_access + n_pad = dim
        dim_N, // rows = k_access
        flexinpu::FlexiNPUMatTypes::B, // matType = MAT_B
        dtype // type = mat_b_dtype
    );

    rewriter.create<flexinpu::ExecuteS1Op>(
        loc,
        rewriter.getI32Type(),
        oncAddrA, // oncPtr = onc_addr_mat_a
        DIM, // elemsPerRow = k_access + k_pad
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::A, // matType = MAT_A
        dtype, // type = mat_a_dtype
        res_dtype // res_type = res_dtype
    );

    rewriter.create<flexinpu::FlushOp>(
        loc, oncAddrD,
        DIM, // elemsPerRow = DIM
        dim_M, // rows = m_access
        flexinpu::FlexiNPUMatTypes::D, // matType = MAT_D
        res_dtype // type = res_dtype
    );

    rewriter.create<flexinpu::DmaStoreOp>(
        loc, matC,
        oncAddrD,
        DIM, // elemsPerRow = DIM
        dim_M, // rows = m_access
        cRowStride, // stride = N
        false, // trans = false
        res_dtype, // type = mat_d_dtype
        pad_N
    );

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