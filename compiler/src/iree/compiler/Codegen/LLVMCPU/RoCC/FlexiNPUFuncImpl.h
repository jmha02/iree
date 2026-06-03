//===- FlexiNPUFuncImpl.h - FlexiNPU Function Implementations ----*- C++ -*-===//
//
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_FLEXINPUFUNCIMPL_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_FLEXINPUFUNCIMPL_H_

#include "iree/compiler/Codegen/LLVMCPU/RoCC/Utils/FuncUtils.h"
#include "iree/compiler/Codegen/LLVMCPU/RoCC/Utils/Utils.h"
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace iree_compiler;

namespace mlir::iree_compiler {

// FlexiNPU operation codes (from flexi.h) - exactly matching macro definitions
#define OP_DMA_LOAD 12
#define OP_GEMM_PRELOAD 3
#define OP_GEMM_EXECUTE_S1 5
#define OP_GEMM_FLUSH 7
#define OP_DMA_STORE 13

// FlexiNPU ROCC Function Codes
enum class FlexiNPURoccFunct : int32_t {
  DMA_LOAD = OP_DMA_LOAD,
  GEMM_PRELOAD = OP_GEMM_PRELOAD,
  GEMM_EXECUTE_S1 = OP_GEMM_EXECUTE_S1,
  GEMM_FLUSH = OP_GEMM_FLUSH,
  DMA_STORE = OP_DMA_STORE
};

// Forward declaration
void injectFlexiNPURocc(OpBuilder &builder, Location loc, Value rs1, Value rs2,
                        int32_t funct, bool extend);

//===----------------------------------------------------------------------===//
// FlexiNPU DMA Load Implementation
//===----------------------------------------------------------------------===//

class FlexiNPUDmaLoad : public RoCCFuncParams<FlexiNPUDmaLoad> {
public:
  FlexiNPUDmaLoad(OpBuilder &builder, Location loc)
      : RoCCFuncParams(builder, loc) {}

  FlexiNPUDmaLoad(OpBuilder &builder, Location loc, int32_t rowNumel, int32_t totalRow,
                  int32_t stride, int32_t trans, int32_t dtype, int32_t zeroPad)
      : RoCCFuncParams(builder, loc) {
    this->rowNumel = rowNumel;
    this->totalRow = totalRow;
    this->stride = stride;
    this->trans = trans;
    this->dtype = dtype;
    this->zeroPad = zeroPad;
  }

  std::string getFuncName() override { return "flexinpu_dma_load"; }

  ConstParam<int32_t> rowNumel{64};
  ConstParam<int32_t> totalRow{64};
  ConstParam<int32_t> stride{64};
  ConstParam<int32_t> trans{0};
  ConstParam<int32_t> dtype{1}; // i32
  ConstParam<int32_t> zeroPad{0};

  auto as_tuple() {
    return std::tie(rowNumel, totalRow, stride, trans, dtype, zeroPad);
  }

  /*
   * DMALoad macro from flexi.h:
   * #define DMALoad(ofc_addr, onc_addr, row_numel, total_row, stride, trans, dtype, zero_pad) \
   *   ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
   *     ((uint64_t)(onc_addr) << 38) | ((uint64_t)(ofc_addr) >> 10), \
   *     ((uint64_t)(ofc_addr)  << 54) | \
   *     ((uint64_t)(zero_pad)  << 41) | \
   *     ((uint64_t)(dtype)     << 38) | \
   *     ((uint64_t)(trans)     << 37) | \
   *     ((uint64_t)(stride)    << 27) | \
   *     ((uint64_t)(total_row) << 17) | \
   *     ((uint64_t)(row_numel) << 7)  | \
   *     ((uint64_t)(OP_DMA_LOAD)), OP_DMA_LOAD);
   *
   * Note: For simplicity, we'll use dummy addresses and focus on the lower 32 bits of rs2
   */
  void impl() override {
    // Get block arguments (parameters passed to function)
    Block *block = builder.getInsertionBlock();
    Value ofcAddr = block->getArgument(0);     // ofc_addr
    Value oncAddr = block->getArgument(1);     // onc_addr
    Value rowNumelVal = block->getArgument(2); // row_numel
    Value totalRowVal = block->getArgument(3); // total_row
    Value strideVal = block->getArgument(4);   // stride
    Value transVal = block->getArgument(5);    // trans
    Value dtypeVal = block->getArgument(6);    // dtype
    Value zeroPadVal = block->getArgument(7);  // zero_pad

    // Ensure all values are i64
    Value ofcAddr64 = ensureI64(builder, loc, ofcAddr);
    Value oncAddr64 = ensureI64(builder, loc, oncAddr);

    // rs1: ((uint64_t)(onc_addr) << 38) | ((uint64_t)(ofc_addr) >> 10)
    Value oncAddrShifted = builder.create<arith::ShLIOp>(loc, oncAddr64,
                                                         builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(38)));
    Value ofcAddrShifted = builder.create<arith::ShRUIOp>(loc, ofcAddr64,
                                                          builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(10)));
    Value rs1 = builder.create<arith::OrIOp>(loc, oncAddrShifted, ofcAddrShifted);

    // rs2: ((uint64_t)(ofc_addr) << 54) | other_fields
    Value ofcAddrUpper = builder.create<arith::ShLIOp>(loc, ofcAddr64,
                                                       builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(54)));

    // Build rs2 bitfield from function arguments, including ofc_addr upper bits
    Value rs2Lower = BitFieldBuilder(builder, loc)
                        .cat(OP_DMA_LOAD, 0)
                        .cat(rowNumelVal, 7)
                        .cat(totalRowVal, 17)
                        .cat(strideVal, 27)
                        .cat(transVal, 37)
                        .cat(dtypeVal, 38)
                        .cat(zeroPadVal, 41);

    Value rs2 = builder.create<arith::OrIOp>(loc, ofcAddrUpper, rs2Lower);

    injectFlexiNPURocc(builder, loc, rs1, rs2, OP_DMA_LOAD);
  }
};

//===----------------------------------------------------------------------===//
// FlexiNPU GEMM Preload Implementation
//===----------------------------------------------------------------------===//

class FlexiNPUGemmPreload : public RoCCFuncParams<FlexiNPUGemmPreload> {
public:
  FlexiNPUGemmPreload(OpBuilder &builder, Location loc)
      : RoCCFuncParams(builder, loc) {}

  FlexiNPUGemmPreload(OpBuilder &builder, Location loc, int32_t rowNumel, int32_t totalRow,
                      int32_t matType, int32_t dtype)
      : RoCCFuncParams(builder, loc) {
    this->rowNumel = rowNumel;
    this->totalRow = totalRow;
    this->matType = matType;
    this->dtype = dtype;
  }

  std::string getFuncName() override { return "flexinpu_gemm_preload"; }

  ConstParam<int32_t> rowNumel{64};
  ConstParam<int32_t> totalRow{64};
  ConstParam<int32_t> matType{0}; // A matrix
  ConstParam<int32_t> dtype{1}; // i32

  auto as_tuple() {
    return std::tie(rowNumel, totalRow, matType, dtype);
  }

  /*
   * GEMMPreload macro from flexi.h:
   * #define GEMMPreload(addr, row_numel, total_row, mat_type, dtype) \
   *   ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
   *     ((uint64_t)(addr) << 38), \
   *     ((uint64_t)(dtype)     << 38) | \
   *     ((uint64_t)(mat_type)  << 27) | \
   *     ((uint64_t)(total_row) << 17) | \
   *     ((uint64_t)(row_numel) << 7)  | \
   *     ((uint64_t)(OP_GEMM_PRELOAD)), OP_GEMM_PRELOAD);
   */
  void impl() override {
    // Get block arguments (parameters passed to function)
    Block *block = builder.getInsertionBlock();
    Value addr = block->getArgument(0);        // addr
    Value rowNumelVal = block->getArgument(1); // row_numel
    Value totalRowVal = block->getArgument(2); // total_row
    Value matTypeVal = block->getArgument(3);  // mat_type
    Value dtypeVal = block->getArgument(4);    // dtype

    // rs1: address shifted left by 38 bits (matching flexi.h GEMMPreload macro)
    Value addr64 = ensureI64(builder, loc, addr);
    Value rs1 = builder.create<arith::ShLIOp>(loc, addr64,
                                              builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(38)));

    // rs2: bitfield exactly matching flexi.h GEMMPreload macro
    Value rs2 = BitFieldBuilder(builder, loc)
                    .cat(OP_GEMM_PRELOAD, 0)
                    .cat(rowNumelVal, 7)
                    .cat(totalRowVal, 17)
                    .cat(matTypeVal, 27)
                    .cat(dtypeVal, 38);

    injectFlexiNPURocc(builder, loc, rs1, rs2, OP_GEMM_PRELOAD);
  }
};

//===----------------------------------------------------------------------===//
// FlexiNPU GEMM Execute S1 Implementation
//===----------------------------------------------------------------------===//

class FlexiNPUGemmExecuteS1 : public RoCCFuncParams<FlexiNPUGemmExecuteS1> {
public:
  FlexiNPUGemmExecuteS1(OpBuilder &builder, Location loc)
      : RoCCFuncParams(builder, loc) {}

  FlexiNPUGemmExecuteS1(OpBuilder &builder, Location loc, int32_t rowNumel, int32_t totalRow,
                        int32_t matType, int32_t dtype, int32_t resDtype)
      : RoCCFuncParams(builder, loc) {
    this->rowNumel = rowNumel;
    this->totalRow = totalRow;
    this->matType = matType;
    this->dtype = dtype;
    this->resDtype = resDtype;
  }

  std::string getFuncName() override { return "flexinpu_gemm_execute_s1"; }

  ConstParam<int32_t> rowNumel{64};
  ConstParam<int32_t> totalRow{64};
  ConstParam<int32_t> matType{2}; // C matrix
  ConstParam<int32_t> dtype{1}; // i32
  ConstParam<int32_t> resDtype{1}; // i32

  auto as_tuple() {
    return std::tie(rowNumel, totalRow, matType, dtype, resDtype);
  }

  /*
   * GEMMExecuteS1 macro from flexi.h:
   * #define GEMMExecuteS1(addr, row_numel, total_row, mat_type, dtype, res_dtype) \
   *   ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
   *     ((uint64_t)(addr) << 38), \
   *     ((uint64_t)(dtype)     << 38) | \
   *     ((uint64_t)(res_dtype) << 30) | \
   *     ((uint64_t)(mat_type)  << 27) | \
   *     ((uint64_t)(total_row) << 17) | \
   *     ((uint64_t)(row_numel) << 7)  | \
   *     ((uint64_t)(OP_GEMM_EXECUTE_S1)), OP_GEMM_EXECUTE_S1);
   */
  void impl() override {
    // Get block arguments (parameters passed to function)
    Block *block = builder.getInsertionBlock();
    Value addr = block->getArgument(0);        // addr
    Value rowNumelVal = block->getArgument(1); // row_numel
    Value totalRowVal = block->getArgument(2); // total_row
    Value matTypeVal = block->getArgument(3);  // mat_type
    Value dtypeVal = block->getArgument(4);    // dtype
    Value resDtypeVal = block->getArgument(5); // res_dtype

    // rs1: address shifted left by 38 bits (matching flexi.h GEMMExecuteS1 macro)
    Value addr64 = ensureI64(builder, loc, addr);
    Value rs1 = builder.create<arith::ShLIOp>(loc, addr64,
                                              builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(38)));

    // rs2: bitfield exactly matching flexi.h GEMMExecuteS1 macro
    Value rs2 = BitFieldBuilder(builder, loc)
                    .cat(OP_GEMM_EXECUTE_S1, 0)
                    .cat(rowNumelVal, 7)
                    .cat(totalRowVal, 17)
                    .cat(matTypeVal, 27)
                    .cat(resDtypeVal, 30)
                    .cat(dtypeVal, 38);

    injectFlexiNPURocc(builder, loc, rs1, rs2, OP_GEMM_EXECUTE_S1);
  }
};

//===----------------------------------------------------------------------===//
// FlexiNPU GEMM Flush Implementation
//===----------------------------------------------------------------------===//

class FlexiNPUGemmFlush : public RoCCFuncParams<FlexiNPUGemmFlush> {
public:
  FlexiNPUGemmFlush(OpBuilder &builder, Location loc)
      : RoCCFuncParams(builder, loc) {}

  std::string getFuncName() override { return "flexinpu_gemm_flush"; }

  auto as_tuple() {
    return std::tie(); // No parameters for flush
  }

  /*
   * GEMMFlush macro from flexi.h:
   * #define GEMMFlush(addr, row_numel, total_row, mat_type, dtype) \
   *   ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(addr) << 38), \
   *                                            ((uint64_t)(dtype)     << 38) | \
   *                                            ((uint64_t)(mat_type)  << 27) | \
   *                                            ((uint64_t)(total_row) << 17) | \
   *                                            ((uint64_t)(row_numel) << 7)  | \
   *                                            ((uint64_t)(OP_GEMM_FLUSH)), OP_GEMM_FLUSH);
   */
  void impl() override {
    // Get block arguments (parameters passed to function)
    Block *block = builder.getInsertionBlock();
    Value addr = block->getArgument(0);        // addr
    Value rowNumelVal = block->getArgument(1); // row_numel
    Value totalRowVal = block->getArgument(2); // total_row
    Value matTypeVal = block->getArgument(3);  // mat_type
    Value dtypeVal = block->getArgument(4);    // dtype

    // rs1: address shifted left by 38 bits (matching flexi.h GEMMFlush macro)
    Value addr64 = ensureI64(builder, loc, addr);
    Value rs1 = builder.create<arith::ShLIOp>(loc, addr64,
                                              builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(38)));

    Value rs2 = BitFieldBuilder(builder, loc)
                    .cat(OP_GEMM_FLUSH, 0)
                    .cat(rowNumelVal, 7)
                    .cat(totalRowVal, 17)
                    .cat(matTypeVal, 27)
                    .cat(dtypeVal, 38);

    injectFlexiNPURocc(builder, loc, rs1, rs2, OP_GEMM_FLUSH);
  }
};

//===----------------------------------------------------------------------===//
// FlexiNPU DMA Store Implementation
//===----------------------------------------------------------------------===//

class FlexiNPUDmaStore : public RoCCFuncParams<FlexiNPUDmaStore> {
public:
  FlexiNPUDmaStore(OpBuilder &builder, Location loc)
      : RoCCFuncParams(builder, loc) {}

  FlexiNPUDmaStore(OpBuilder &builder, Location loc, int32_t rowNumel, int32_t totalRow,
                   int32_t stride, int32_t trans, int32_t dtype, int32_t zeroPad)
      : RoCCFuncParams(builder, loc) {
    this->rowNumel = rowNumel;
    this->totalRow = totalRow;
    this->stride = stride;
    this->trans = trans;
    this->dtype = dtype;
    this->zeroPad = zeroPad;
  }

  std::string getFuncName() override { return "flexinpu_dma_store"; }

  ConstParam<int32_t> rowNumel{64};
  ConstParam<int32_t> totalRow{64};
  ConstParam<int32_t> stride{64};
  ConstParam<int32_t> trans{0};
  ConstParam<int32_t> dtype{1}; // i32
  ConstParam<int32_t> zeroPad{0};

  auto as_tuple() {
    return std::tie(rowNumel, totalRow, stride, trans, dtype, zeroPad);
  }

  /*
   * DMAStore macro from flexi.h:
   * #define DMAStore(ofc_addr, onc_addr, row_numel, total_row, stride, trans, dtype, zero_pad) \
   *   ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
   *     ((uint64_t)(onc_addr) << 38) | ((uint64_t)(ofc_addr) >> 10), \
   *     ((uint64_t)(ofc_addr)  << 54) | \
   *     ((uint64_t)(zero_pad)  << 41) | \
   *     ((uint64_t)(dtype)     << 38) | \
   *     ((uint64_t)(trans)     << 37) | \
   *     ((uint64_t)(stride)    << 27) | \
   *     ((uint64_t)(total_row) << 17) | \
   *     ((uint64_t)(row_numel) << 7)  | \
   *     ((uint64_t)(OP_DMA_STORE)), OP_DMA_STORE);
   */
  void impl() override {
    // Get block arguments (parameters passed to function)
    Block *block = builder.getInsertionBlock();
    Value ofcAddr = block->getArgument(0);     // ofc_addr
    Value oncAddr = block->getArgument(1);     // onc_addr
    Value rowNumelVal = block->getArgument(2); // row_numel
    Value totalRowVal = block->getArgument(3); // total_row
    Value strideVal = block->getArgument(4);   // stride
    Value transVal = block->getArgument(5);    // trans
    Value dtypeVal = block->getArgument(6);    // dtype
    Value zeroPadVal = block->getArgument(7);  // zero_pad

    // Ensure all values are i64
    Value ofcAddr64 = ensureI64(builder, loc, ofcAddr);
    Value oncAddr64 = ensureI64(builder, loc, oncAddr);

    // rs1: ((uint64_t)(onc_addr) << 38) | ((uint64_t)(ofc_addr) >> 10)
    Value oncAddrShifted = builder.create<arith::ShLIOp>(loc, oncAddr64,
                                                         builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(38)));
    Value ofcAddrShifted = builder.create<arith::ShRUIOp>(loc, ofcAddr64,
                                                          builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(10)));
    Value rs1 = builder.create<arith::OrIOp>(loc, oncAddrShifted, ofcAddrShifted);

    // rs2: ((uint64_t)(ofc_addr) << 54) | other_fields
    Value ofcAddrUpper = builder.create<arith::ShLIOp>(loc, ofcAddr64,
                                                       builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(54)));

    // Build rs2 bitfield from function arguments, including ofc_addr upper bits
    Value rs2Lower = BitFieldBuilder(builder, loc)
                        .cat(OP_DMA_STORE, 0)
                        .cat(rowNumelVal, 7)
                        .cat(totalRowVal, 17)
                        .cat(strideVal, 27)
                        .cat(transVal, 37)
                        .cat(dtypeVal, 38)
                        .cat(zeroPadVal, 41);

    Value rs2 = builder.create<arith::OrIOp>(loc, ofcAddrUpper, rs2Lower);

    injectFlexiNPURocc(builder, loc, rs1, rs2, OP_DMA_STORE);
  }
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_FLEXINPUFUNCIMPL_H_
