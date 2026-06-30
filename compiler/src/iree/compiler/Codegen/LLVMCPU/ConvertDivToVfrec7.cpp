// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::iree_compiler {
namespace {

static bool isConstantLike(Value value) {
  if (!value) {
    return false;
  }
  if (matchPattern(value, m_Constant())) {
    return true;
  }
  if (dyn_cast_or_null<LLVM::ConstantOp>(value.getDefiningOp())) {
    return true;
  }
  if (auto bitcast = dyn_cast_or_null<LLVM::BitcastOp>(value.getDefiningOp())) {
    return isConstantLike(bitcast.getOperand());
  }
  return false;
}

static std::optional<double> getScalarFloatConstant(Value value) {
  auto constant = dyn_cast_or_null<LLVM::ConstantOp>(value.getDefiningOp());
  if (!constant) {
    return std::nullopt;
  }
  auto floatAttr = dyn_cast<FloatAttr>(constant.getValue());
  if (!floatAttr) {
    return std::nullopt;
  }
  return floatAttr.getValueAsDouble();
}

static std::optional<double> getSplatFloatConstant(Value value) {
  auto constant = dyn_cast_or_null<LLVM::ConstantOp>(value.getDefiningOp());
  if (!constant) {
    return std::nullopt;
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(constant.getValue())) {
    return floatAttr.getValueAsDouble();
  }
  auto denseAttr = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  if (!denseAttr || !denseAttr.isSplat()) {
    return std::nullopt;
  }
  return denseAttr.getSplatValue<APFloat>().convertToDouble();
}

static Value buildVfrec7Intrinsic(PatternRewriter &rewriter, Location loc,
                                  VectorType scalableType, Value operand) {
  if (!scalableType.isScalable()) {
    return {};
  }

  std::string name = "llvm.riscv.vfrec7.nxv" +
                     std::to_string(scalableType.getNumElements());
  auto elementType = scalableType.getElementType();
  if (elementType.isF16()) {
    name += "f16";
  } else if (elementType.isBF16()) {
    name += "bf16";
  } else if (elementType.isF32()) {
    name += "f32";
  } else if (elementType.isF64()) {
    name += "f64";
  } else {
    return {};
  }
  name += ".i64";

  auto nameAttr = StringAttr::get(rewriter.getContext(), name);
  auto frmConst = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(7));
  auto vlConst = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(scalableType.getNumElements()));

  auto call = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, scalableType, nameAttr,
      ValueRange{operand, operand, frmConst, vlConst});
  return call.getResult(0);
}

static Value buildVfrsqrt7Intrinsic(PatternRewriter &rewriter, Location loc,
                                    VectorType scalableType, Value operand) {
  if (!scalableType.isScalable()) {
    return {};
  }

  std::string name = "llvm.riscv.vfrsqrt7.nxv" +
                     std::to_string(scalableType.getNumElements());
  auto elementType = scalableType.getElementType();
  if (elementType.isF16()) {
    name += "f16";
  } else if (elementType.isBF16()) {
    name += "bf16";
  } else if (elementType.isF32()) {
    name += "f32";
  } else if (elementType.isF64()) {
    name += "f64";
  } else {
    return {};
  }
  name += ".i64";

  auto nameAttr = StringAttr::get(rewriter.getContext(), name);
  auto vlConst = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(scalableType.getNumElements()));

  auto call = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, scalableType, nameAttr, ValueRange{operand, operand, vlConst});
  return call.getResult(0);
}

static Value buildScalarVfrsqrt7(PatternRewriter &rewriter, Location loc,
                                 FloatType floatType, Value operand) {
  auto scalableType = VectorType::get({1}, floatType, {true});
  Value poison = rewriter.create<LLVM::PoisonOp>(loc, scalableType);
  Value zero = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
  Value vectorOperand = rewriter.create<LLVM::InsertElementOp>(
      loc, scalableType, poison, operand, zero);
  Value vectorResult =
      buildVfrsqrt7Intrinsic(rewriter, loc, scalableType, vectorOperand);
  if (!vectorResult) {
    return {};
  }
  return rewriter.create<LLVM::ExtractElementOp>(loc, floatType, vectorResult,
                                                 zero);
}

struct FDivToVfrec7Pattern : OpRewritePattern<LLVM::FDivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FDivOp op,
                                PatternRewriter &rewriter) const override {
    if (auto floatType = dyn_cast<FloatType>(op.getType())) {
      std::optional<double> lhs = getScalarFloatConstant(op.getLhs());
      if (lhs && *lhs == 1.0) {
        if (auto sqrtCall =
                op.getRhs().getDefiningOp<LLVM::CallIntrinsicOp>()) {
          if (sqrtCall.getIntrin() == "llvm.sqrt.f32" &&
              sqrtCall.getNumOperands() == 1 && floatType.isF32()) {
            Value approxRsqrt = buildScalarVfrsqrt7(
                rewriter, op.getLoc(), floatType, sqrtCall.getOperand(0));
            if (approxRsqrt) {
              rewriter.replaceOp(op, approxRsqrt);
              return success();
            }
          }
        }
      }
      std::optional<double> rhs = getScalarFloatConstant(op.getRhs());
      if (!rhs || *rhs == 0.0) {
        return failure();
      }
      auto reciprocalAttr =
          rewriter.getFloatAttr(floatType, 1.0 / static_cast<double>(*rhs));
      Value reciprocal = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), op.getType(), reciprocalAttr);
      Value mul = rewriter.create<LLVM::FMulOp>(op.getLoc(), op.getType(),
                                                op.getLhs(), reciprocal);
      rewriter.replaceOp(op, mul);
      return success();
    }

    auto vectorType = dyn_cast<VectorType>(op.getType());
    if (!vectorType || !vectorType.hasStaticShape()) {
      return failure();
    }
    auto elementType = dyn_cast<FloatType>(vectorType.getElementType());
    if (!elementType) {
      return failure();
    }

    if (std::optional<double> rhs = getSplatFloatConstant(op.getRhs())) {
      if (*rhs == 0.0) {
        return failure();
      }
      auto reciprocalAttr = SplatElementsAttr::get(
          vectorType, rewriter.getFloatAttr(elementType,
                                            1.0 / static_cast<double>(*rhs)));
      Value reciprocal = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), op.getType(), reciprocalAttr);
      Value mul = rewriter.create<LLVM::FMulOp>(op.getLoc(), op.getType(),
                                                op.getLhs(), reciprocal);
      rewriter.replaceOp(op, mul);
      return success();
    }

    if (!vectorType.isScalable() || isConstantLike(op.getRhs())) {
      return failure();
    }
    Value reciprocal =
        buildVfrec7Intrinsic(rewriter, op.getLoc(), vectorType, op.getRhs());
    if (!reciprocal) {
      return failure();
    }

    Value mul = rewriter.create<LLVM::FMulOp>(op.getLoc(), op.getType(),
                                              op.getLhs(), reciprocal);
    rewriter.replaceOp(op, mul);
    return success();
  }
};

struct SDivToVfrec7Pattern : OpRewritePattern<LLVM::SDivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SDivOp op,
                                PatternRewriter &rewriter) const override {
    auto vectorType = dyn_cast<VectorType>(op.getType());
    if (!vectorType || !vectorType.hasStaticShape()) {
      return failure();
    }
    auto integerType = dyn_cast<IntegerType>(vectorType.getElementType());
    if (!integerType || integerType.getWidth() != 32) {
      return failure();
    }

    auto floatVectorType =
        VectorType::get(vectorType.getShape(), rewriter.getF32Type(),
                        vectorType.getScalableDims());
    Value lhsF = rewriter.create<LLVM::SIToFPOp>(op.getLoc(), floatVectorType,
                                                 op.getLhs());
    Value rhsF = rewriter.create<LLVM::SIToFPOp>(op.getLoc(), floatVectorType,
                                                 op.getRhs());

    if (!floatVectorType.isScalable() || isConstantLike(rhsF)) {
      return failure();
    }
    Value reciprocal =
        buildVfrec7Intrinsic(rewriter, op.getLoc(), floatVectorType, rhsF);
    if (!reciprocal) {
      return failure();
    }

    Value mul = rewriter.create<LLVM::FMulOp>(op.getLoc(), floatVectorType,
                                              lhsF, reciprocal);
    Value result = rewriter.create<LLVM::FPToSIOp>(op.getLoc(), op.getType(),
                                                   mul);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LLVMCPUDivToVfrec7Pass
    : public PassWrapper<LLVMCPUDivToVfrec7Pass,
                         OperationPass<LLVM::LLVMFuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCPUDivToVfrec7Pass)

  StringRef getArgument() const final { return "iree-llvmcpu-div-to-vfrec7"; }

  StringRef getDescription() const final {
    return "Rewrite LLVM vector div to vfrec7 reciprocal approximation.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FDivToVfrec7Pattern, SDivToVfrec7Pattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<LLVM::LLVMFuncOp>>
createLLVMCPUDivToVfrec7Pass() {
  return std::make_unique<LLVMCPUDivToVfrec7Pass>();
}

} // namespace mlir::iree_compiler
