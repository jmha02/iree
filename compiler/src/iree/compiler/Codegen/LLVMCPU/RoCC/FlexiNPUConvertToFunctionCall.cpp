//===- FlexiNPUConvertToFunctionCall.cpp - Convert FlexiNPU Ops ----*- C++ -*-===//
//
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/LLVMCPU/RoCC/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/RoCC/FlexiNPUFuncImpl.h"
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FLEXINPUCONVERTTOFUNCTIONCALLPASS
#include "iree/compiler/Codegen/LLVMCPU/RoCC/Passes.h.inc"

namespace {

static std::string mangleWithExecAndVariant(Operation *anchor, llvm::StringRef base) {
  using namespace mlir::iree_compiler::IREE::HAL;
  std::string out = base.str();
  if (auto variant = anchor->getParentOfType<ExecutableVariantOp>()) {
    if (auto exec = variant->getParentOfType<ExecutableOp>()) {
      out += "_";
      out += exec.getSymName().str();       // e.g. main_dispatch_2
    }
    out += "_";
    out += variant.getSymName().str();      // e.g. embedded_elf_riscv_64
  }
  return out;
}

static func::FuncOp getOrCreateFunction(ModuleOp moduleOp,
  OpBuilder &builder,
  Location loc,
  StringRef baseName,
  FunctionType funcType) {
  SymbolTable symtab(moduleOp);

  if (auto existing = moduleOp.lookupSymbol<func::FuncOp>(baseName)) {
  // Force private if someone made it public elsewhere.
  if (!existing.isPrivate())
  existing.setVisibility(SymbolTable::Visibility::Private);

  // If the type differs, mangle a new name.
  if (existing.getFunctionType() != funcType) {
  auto hc = hash_value(funcType);
  llvm::SmallString<32> buf;
  llvm::raw_svector_ostream os(buf);
  os << hc;
  std::string mangled = (baseName + "." + os.str().str()).str();

  if (auto dup = moduleOp.lookupSymbol<func::FuncOp>(mangled)) {
  if (!dup.isPrivate())
  dup.setVisibility(SymbolTable::Visibility::Private);
  return dup;
  }

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto f = builder.create<func::FuncOp>(loc, mangled, funcType);
  f.setVisibility(SymbolTable::Visibility::Private);
  symtab.insert(f);
  return f;
  }
  return existing;
  }

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto f = builder.create<func::FuncOp>(loc, baseName, funcType);
  f.setVisibility(SymbolTable::Visibility::Private);
  symtab.insert(f); // ensure uniqueness if a race occurs
  return f;
}

// Helper function to convert FlexiNPU types to numeric values
static int32_t convertFlexiNPUType(flexinpu::FlexiNPUTypes type) {
  switch (type) {
    case flexinpu::FlexiNPUTypes::i4: return 0;    // INT4 runtime value
    case flexinpu::FlexiNPUTypes::i8: return 1;    // INT8 runtime value
    case flexinpu::FlexiNPUTypes::i32: return 2;   // INT32 runtime value
    case flexinpu::FlexiNPUTypes::i64: return 3;   // INT64 runtime value
    case flexinpu::FlexiNPUTypes::f16: return 4;   // FP16 runtime value
    case flexinpu::FlexiNPUTypes::f32: return 5;   // FP32 runtime value
    case flexinpu::FlexiNPUTypes::f64: return 6;   // FP64 runtime value
    case flexinpu::FlexiNPUTypes::bf16: return 7;  // BF16 runtime value
  }
  return 5; // default FP32
}

static int32_t convertFlexiNPUMatType(flexinpu::FlexiNPUMatTypes matType) {
  switch (matType) {
    case flexinpu::FlexiNPUMatTypes::A: return 0;
    case flexinpu::FlexiNPUMatTypes::B: return 1;
    case flexinpu::FlexiNPUMatTypes::C: return 2;
    case flexinpu::FlexiNPUMatTypes::D: return 3;
  }
  return 0; // default
}

static Value computeMemRefAddress(OpBuilder &builder, Location loc,
                                  Value memref) {
  auto i64Ty   = builder.getI64Type();
  auto indexTy = builder.getIndexType();

  auto meta = builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
  Value baseBuf  = meta.getBaseBuffer();     // memref<T>
  Value offset   = meta.getOffset();         // index
  // auto strides   = meta.getStrides();        // SmallVector<Value> (index)

  Value baseIdx  = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
                      loc, indexTy, baseBuf);
  Value baseI64  = builder.create<arith::IndexCastOp>(loc, i64Ty, baseIdx);

  auto memrefTy = mlir::cast<mlir::MemRefType>(memref.getType());
  unsigned elemSizeBytes =
      memrefTy.getElementType().getIntOrFloatBitWidth() / 8;
  Value elemBytes =
      builder.create<arith::ConstantOp>(loc, i64Ty,
        builder.getI64IntegerAttr(static_cast<int64_t>(elemSizeBytes)));

  Value totalElemOffI64 =
  builder.create<arith::IndexCastOp>(loc, i64Ty, offset); // start from offset

  Value byteOff = builder.create<arith::MulIOp>(loc, totalElemOffI64, elemBytes);
  Value absAddr = builder.create<arith::AddIOp>(loc, baseI64, byteOff);
  return absAddr;
}

struct FlexiNPUDmaLoadOpPattern : public OpRewritePattern<flexinpu::DmaLoadOp> {
  using OpRewritePattern<flexinpu::DmaLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(flexinpu::DmaLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Extract parameters from the operation attributes
    int32_t elemsPerRow = op.getElemsPerRow();
    int32_t rows = op.getRows();
    int32_t stride = op.getStride();
    bool trans = op.getTrans();
    int32_t dtype = convertFlexiNPUType(op.getType());
    int32_t zeroPad = op.getZeroPad();

    // Create function signature matching flexi.h DMALoad macro:
    // DMALoad(ofc_addr, onc_addr, row_numel, total_row, stride, trans, dtype, zero_pad)
    SmallVector<Type> argTypes;
    argTypes.push_back(rewriter.getI64Type());    // ofc_addr
    argTypes.push_back(rewriter.getI64Type());    // onc_addr
    argTypes.push_back(rewriter.getI32Type());    // row_numel
    argTypes.push_back(rewriter.getI32Type());    // total_row
    argTypes.push_back(rewriter.getI32Type());    // stride
    argTypes.push_back(rewriter.getI1Type());     // trans
    argTypes.push_back(rewriter.getI32Type());    // dtype
    argTypes.push_back(rewriter.getI32Type());    // zero_pad

    auto funcType = rewriter.getFunctionType(argTypes, {});
    // auto funcName = "flexinpu_dma_load";

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto base = StringRef("flexinpu_dma_load");
    std::string uniqueName = mangleWithExecAndVariant(op, base);
    auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(),
                                    /*baseName=*/uniqueName, funcType);
    // auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(), funcName, funcType);

    // If function body is empty, add implementation using FlexiNPUFuncImpl.h
    if (funcOp.getBody().empty()) {
      auto &funcBody = funcOp.getBody();
      Block *block = &funcBody.emplaceBlock();
      block->addArgument(rewriter.getI64Type(), op.getLoc());  // ofc_addr
      block->addArgument(rewriter.getI64Type(), op.getLoc());  // onc_addr
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // row_numel
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // total_row
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // stride
      block->addArgument(rewriter.getI1Type(), op.getLoc());   // trans
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // dtype
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // zero_pad
      auto bodyBuilder = OpBuilder::atBlockEnd(block);

      FlexiNPUDmaLoad dmaLoadImpl(bodyBuilder, op.getLoc(), elemsPerRow, rows, stride,
                                  trans ? 1 : 0, dtype, zeroPad);
      dmaLoadImpl.impl();

      bodyBuilder.create<func::ReturnOp>(op.getLoc());
    }

    // Compute actual memref address for ofc_addr
    Value ofcAddr = computeMemRefAddress(rewriter, op.getLoc(), op.getSrcOfcMemref());

    // Create arguments for function call
    SmallVector<Value> args;
    args.push_back(ofcAddr);                                           // ofc_addr
    args.push_back(op.getOncPtr());                                   // onc_addr
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(elemsPerRow))); // row_numel
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(rows)));        // total_row
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(stride)));      // stride
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI1Type(),
                                                       rewriter.getBoolAttr(trans)));             // trans
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(dtype)));       // dtype
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(zeroPad)));     // zero_pad

    rewriter.create<func::CallOp>(op.getLoc(), funcOp, ValueRange(args));
    rewriter.eraseOp(op);
    return success();
  }
};

// Pattern to convert FlexiNPU Preload to function call
struct FlexiNPUPreloadOpPattern : public OpRewritePattern<flexinpu::PreloadOp> {
  using OpRewritePattern<flexinpu::PreloadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(flexinpu::PreloadOp op,
                                PatternRewriter &rewriter) const override {
    // Extract parameters
    int32_t elemsPerRow = op.getElemsPerRow();
    int32_t rows = op.getRows();
    int32_t matType = convertFlexiNPUMatType(op.getMatType());
    int32_t dtype = convertFlexiNPUType(op.getType());

    // Create function signature matching flexi.h GEMMPreload macro:
    // GEMMPreload(addr, row_numel, total_row, mat_type, dtype)
    SmallVector<Type> argTypes;
    argTypes.push_back(rewriter.getI64Type());    // addr
    argTypes.push_back(rewriter.getI32Type());    // row_numel
    argTypes.push_back(rewriter.getI32Type());    // total_row
    argTypes.push_back(rewriter.getI32Type());    // mat_type
    argTypes.push_back(rewriter.getI32Type());    // dtype

    auto funcType = rewriter.getFunctionType(argTypes, {});
    // auto funcName = "flexinpu_gemm_preload";

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto base = StringRef("flexinpu_gemm_preload");
    std::string uniqueName = mangleWithExecAndVariant(op, base);
    auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(),
                                    /*baseName=*/uniqueName, funcType);
    // auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(), funcName, funcType);

    // If function body is empty, add implementation using FlexiNPUFuncImpl.h
    if (funcOp.getBody().empty()) {
      auto &funcBody = funcOp.getBody();
      Block *block = &funcBody.emplaceBlock();
      block->addArgument(rewriter.getI64Type(), op.getLoc());  // addr
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // row_numel
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // total_row
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // mat_type
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // dtype
      auto bodyBuilder = OpBuilder::atBlockEnd(block);

      FlexiNPUGemmPreload preloadImpl(bodyBuilder, op.getLoc(), elemsPerRow, rows, matType, dtype);
      preloadImpl.impl();

      bodyBuilder.create<func::ReturnOp>(op.getLoc());
    }

    // Create arguments for function call
    SmallVector<Value> args;
    args.push_back(op.getOncPtr());                                   // addr
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(elemsPerRow))); // row_numel
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(rows)));        // total_row
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(matType)));     // mat_type
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(dtype)));       // dtype

    rewriter.create<func::CallOp>(op.getLoc(), funcOp, ValueRange(args));
    rewriter.eraseOp(op);
    return success();
  }
};

// Pattern to convert FlexiNPU ExecuteS1 to function call
struct FlexiNPUExecuteS1OpPattern : public OpRewritePattern<flexinpu::ExecuteS1Op> {
  using OpRewritePattern<flexinpu::ExecuteS1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(flexinpu::ExecuteS1Op op,
                                PatternRewriter &rewriter) const override {
    // Extract parameters
    int32_t elemsPerRow = op.getElemsPerRow();
    int32_t rows = op.getRows();
    int32_t matType = convertFlexiNPUMatType(op.getMatType());
    int32_t dtype = convertFlexiNPUType(op.getType());
    int32_t resDtype = convertFlexiNPUType(op.getResType());

    // Create function signature matching flexi.h GEMMExecuteS1 macro:
    // GEMMExecuteS1(addr, row_numel, total_row, mat_type, dtype, res_dtype)
    SmallVector<Type> argTypes;
    argTypes.push_back(rewriter.getI64Type());    // addr
    argTypes.push_back(rewriter.getI32Type());    // row_numel
    argTypes.push_back(rewriter.getI32Type());    // total_row
    argTypes.push_back(rewriter.getI32Type());    // mat_type
    argTypes.push_back(rewriter.getI32Type());    // dtype
    argTypes.push_back(rewriter.getI32Type());    // res_dtype

    // Use the original result type from the operation
    auto resultType = op.getOut().getType();
    auto funcType = rewriter.getFunctionType(argTypes, {resultType});
    // auto funcName = "flexinpu_gemm_execute_s1";

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto base = StringRef("flexinpu_gemm_execute_s1");
    std::string uniqueName = mangleWithExecAndVariant(op, base);
    auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(),
                                    /*baseName=*/uniqueName, funcType);
    // auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(), funcName, funcType);

    if (funcOp.getBody().empty()) {
      auto &funcBody = funcOp.getBody();
      Block *block = &funcBody.emplaceBlock();
      block->addArgument(rewriter.getI64Type(), op.getLoc());  // addr
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // row_numel
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // total_row
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // mat_type
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // dtype
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // res_dtype
      auto bodyBuilder = OpBuilder::atBlockEnd(block);

      FlexiNPUGemmExecuteS1 executeImpl(bodyBuilder, op.getLoc(), elemsPerRow, rows, matType, dtype, resDtype);
      executeImpl.impl();

      // Create dummy return value - this is a placeholder
      auto dummyResult = bodyBuilder.create<arith::ConstantOp>(op.getLoc(), resultType,
                                                               bodyBuilder.getZeroAttr(resultType));
      bodyBuilder.create<func::ReturnOp>(op.getLoc(), ValueRange{dummyResult});
    }

    // Create arguments for function call
    SmallVector<Value> args;
    args.push_back(op.getOncPtr());                                   // addr
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(elemsPerRow))); // row_numel
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(rows)));        // total_row
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(matType)));     // mat_type
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(dtype)));       // dtype
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(resDtype)));    // res_dtype

    auto result = rewriter.create<func::CallOp>(op.getLoc(), funcOp, ValueRange(args));
    rewriter.replaceOp(op, result.getResults());
    return success();
  }
};

// Pattern to convert FlexiNPU Flush to function call
struct FlexiNPUFlushOpPattern : public OpRewritePattern<flexinpu::FlushOp> {
  using OpRewritePattern<flexinpu::FlushOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(flexinpu::FlushOp op,
                                PatternRewriter &rewriter) const override {
    // Extract parameters
    int32_t elemsPerRow = op.getElemsPerRow();
    int32_t rows = op.getRows();
    int32_t matType = convertFlexiNPUMatType(op.getMatType());
    int32_t dtype = convertFlexiNPUType(op.getType());

    // Create function signature matching flexi.h GEMMFlush macro:
    // GEMMFlush(addr, row_numel, total_row, mat_type, dtype)
    SmallVector<Type> argTypes;
    argTypes.push_back(rewriter.getI64Type());    // addr
    argTypes.push_back(rewriter.getI32Type());    // row_numel
    argTypes.push_back(rewriter.getI32Type());    // total_row
    argTypes.push_back(rewriter.getI32Type());    // mat_type
    argTypes.push_back(rewriter.getI32Type());    // dtype

    auto funcType = rewriter.getFunctionType(argTypes, {});
    // auto funcName = "flexinpu_gemm_flush";

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto base = StringRef("flexinpu_gemm_flush");
    std::string uniqueName = mangleWithExecAndVariant(op, base);
    auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(),
                                    /*baseName=*/uniqueName, funcType);
    // auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(), funcName, funcType);

    // If function body is empty, add implementation using FlexiNPUFuncImpl.h
    if (funcOp.getBody().empty()) {
      auto &funcBody = funcOp.getBody();
      Block *block = &funcBody.emplaceBlock();
      block->addArgument(rewriter.getI64Type(), op.getLoc());  // addr
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // row_numel
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // total_row
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // mat_type
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // dtype
      auto bodyBuilder = OpBuilder::atBlockEnd(block);

      FlexiNPUGemmFlush flushImpl(bodyBuilder, op.getLoc());
      flushImpl.impl();

      bodyBuilder.create<func::ReturnOp>(op.getLoc());
    }

    // Create arguments for function call
    SmallVector<Value> args;
    args.push_back(op.getOncPtr());                                   // addr
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(elemsPerRow))); // row_numel
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(rows)));        // total_row
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(matType)));     // mat_type
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(dtype)));       // dtype

    rewriter.create<func::CallOp>(op.getLoc(), funcOp, ValueRange(args));
    rewriter.eraseOp(op);
    return success();
  }
};

// Pattern to convert FlexiNPU DmaStore to function call
struct FlexiNPUDmaStoreOpPattern : public OpRewritePattern<flexinpu::DmaStoreOp> {
  using OpRewritePattern<flexinpu::DmaStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(flexinpu::DmaStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Extract parameters from the operation attributes
    int32_t elemsPerRow = op.getElemsPerRow();
    int32_t rows = op.getRows();
    int32_t stride = op.getStride();
    bool trans = op.getTrans();
    int32_t dtype = convertFlexiNPUType(op.getType());
    int32_t zeroPad = op.getZeroPad();

    // Create function signature matching flexi.h DMAStore macro:
    // DMAStore(ofc_addr, onc_addr, row_numel, total_row, stride, trans, dtype, zero_pad)
    SmallVector<Type> argTypes;
    argTypes.push_back(rewriter.getI64Type());    // ofc_addr
    argTypes.push_back(rewriter.getI64Type());    // onc_addr
    argTypes.push_back(rewriter.getI32Type());    // row_numel
    argTypes.push_back(rewriter.getI32Type());    // total_row
    argTypes.push_back(rewriter.getI32Type());    // stride
    argTypes.push_back(rewriter.getI1Type());     // trans
    argTypes.push_back(rewriter.getI32Type());    // dtype
    argTypes.push_back(rewriter.getI32Type());    // zero_pad

    auto funcType = rewriter.getFunctionType(argTypes, {});
    // auto funcName = "flexinpu_dma_store";

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto base = StringRef("flexinpu_dma_store");
    std::string uniqueName = mangleWithExecAndVariant(op, base);
    auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(),
                                    /*baseName=*/uniqueName, funcType);
    // auto funcOp = getOrCreateFunction(moduleOp, rewriter, op.getLoc(), funcName, funcType);

    // If function body is empty, add implementation using FlexiNPUFuncImpl.h
    if (funcOp.getBody().empty()) {
      auto &funcBody = funcOp.getBody();
      Block *block = &funcBody.emplaceBlock();
      block->addArgument(rewriter.getI64Type(), op.getLoc());  // ofc_addr
      block->addArgument(rewriter.getI64Type(), op.getLoc());  // onc_addr
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // row_numel
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // total_row
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // stride
      block->addArgument(rewriter.getI1Type(), op.getLoc());   // trans
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // dtype
      block->addArgument(rewriter.getI32Type(), op.getLoc());  // zero_pad
      auto bodyBuilder = OpBuilder::atBlockEnd(block);

      // Use FlexiNPUDmaStore implementation
      FlexiNPUDmaStore storeImpl(bodyBuilder, op.getLoc(), elemsPerRow, rows, stride,
                                trans ? 1 : 0, dtype, zeroPad);
      storeImpl.impl();

      bodyBuilder.create<func::ReturnOp>(op.getLoc());
    }

    // Compute actual memref address for ofc_addr
    Value ofcAddr = computeMemRefAddress(rewriter, op.getLoc(), op.getSrcOfcMemref());

    // Create arguments for function call
    SmallVector<Value> args;
    args.push_back(ofcAddr);                                           // ofc_addr
    args.push_back(op.getOncPtr());                                   // onc_addr
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(elemsPerRow))); // row_numel
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(rows)));        // total_row
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(stride)));      // stride
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI1Type(),
                                                       rewriter.getBoolAttr(trans)));             // trans
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(dtype)));       // dtype
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                       rewriter.getI32IntegerAttr(zeroPad)));     // zero_pad

    rewriter.create<func::CallOp>(op.getLoc(), funcOp, ValueRange(args));
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

// Pass implementation
class FlexiNPUConvertToFunctionCallPass
    : public impl::FlexiNPUConvertToFunctionCallPassBase<FlexiNPUConvertToFunctionCallPass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Debug: Check for scf.if operations before transformation
    int scfIfCountBefore = 0;
    funcOp.walk([&](scf::IfOp ifOp) {
      scfIfCountBefore++;
    });

    // Debug: Check for FlexiNPU operations
    int flexinpuOpCountBefore = 0;
    funcOp.walk([&](Operation *op) {
      if (isa<flexinpu::DmaLoadOp, flexinpu::PreloadOp, flexinpu::ExecuteS1Op,
              flexinpu::FlushOp, flexinpu::DmaStoreOp>(op)) {
        flexinpuOpCountBefore++;
      }
    });

    // llvm::errs() << "DEBUG FlexiNPUConvertToFunctionCall: Before transformation:\n";
    // llvm::errs() << "  scf.if operations: " << scfIfCountBefore << "\n";
    // llvm::errs() << "  FlexiNPU operations: " << flexinpuOpCountBefore << "\n";

    RewritePatternSet patterns(&getContext());
    patterns.add<FlexiNPUDmaLoadOpPattern>(&getContext());
    patterns.add<FlexiNPUPreloadOpPattern>(&getContext());
    patterns.add<FlexiNPUExecuteS1OpPattern>(&getContext());
    patterns.add<FlexiNPUFlushOpPattern>(&getContext());
    patterns.add<FlexiNPUDmaStoreOpPattern>(&getContext());

    // applyPatternsGreedily already preserves control flow structures like scf.if
    // It only transforms the matched operations, not the structure containing them
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }

    // Debug: Check for scf.if operations after transformation
    int scfIfCountAfter = 0;
    funcOp.walk([&](scf::IfOp ifOp) {
      scfIfCountAfter++;
    });

    // llvm::errs() << "DEBUG FlexiNPUConvertToFunctionCall: After transformation:\n";
    // llvm::errs() << "  scf.if operations: " << scfIfCountAfter << "\n";
    // llvm::errs() << "  scf.if preserved: " << (scfIfCountBefore == scfIfCountAfter ? "YES" : "NO") << "\n";
  }
};

} // namespace mlir::iree_compiler