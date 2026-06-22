// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FlexiNPU
#define FlexiNPU

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/OpBase.td"

def LLVM_PointerHost: LLVM_PointerInAddressSpace<0>;
def LLVM_PointerNPU: LLVM_PointerInAddressSpace<1>;

def FlexiNPU_Dialect : Dialect {
  let name = "flexinpu";
  let cppNamespace = "::flexinpu";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let description = [{
    TODO
  }];
  // TODO: extraClassDeclaration
}

//===----------------------------------------------------------------------===//
// FlexiNPU Attribute Definitions
//===----------------------------------------------------------------------===//
def FlexiNPUTypeI4 : I32EnumAttrCase<"i4", 0>;
def FlexiNPUTypeI8 : I32EnumAttrCase<"i8", 1>;
def FlexiNPUTypeI32 : I32EnumAttrCase<"i32", 2>;
def FlexiNPUTypeI64 : I32EnumAttrCase<"i64", 3>;
def FlexiNPUTypeF16: I32EnumAttrCase<"f16", 4>;
def FlexiNPUTypeF32: I32EnumAttrCase<"f32", 5>;
def FlexiNPUTypeF64: I32EnumAttrCase<"f64", 6>;
def FlexiNPUTypeBF16: I32EnumAttrCase<"bf16", 7>;

def FlexiNPUMatTypeA: I32EnumAttrCase<"A", 0>;
def FlexiNPUMatTypeB: I32EnumAttrCase<"B", 1>;
def FlexiNPUMatTypeC: I32EnumAttrCase<"C", 2>;
def FlexiNPUMatTypeD: I32EnumAttrCase<"D", 3>;
//===----------------------------------------------------------------------===//
// FlexiNPU Type Definitions
//===----------------------------------------------------------------------===//

def FlexiNPUTypesAttr: I32EnumAttr<"FlexiNPUTypes", "FlexiNPU Types", [
  FlexiNPUTypeI4, FlexiNPUTypeI8, FlexiNPUTypeI32, FlexiNPUTypeI64, FlexiNPUTypeF16,
  FlexiNPUTypeF32, FlexiNPUTypeF64, FlexiNPUTypeBF16
]> {
  let cppNamespace = "::flexinpu";
}

def FlexiNPUMatTypesAttr: I32EnumAttr<"FlexiNPUMatTypes", "FlexiNPU Matrix Types", [
  FlexiNPUMatTypeA, FlexiNPUMatTypeB, FlexiNPUMatTypeC, FlexiNPUMatTypeD
]> {
  let cppNamespace = "::flexinpu";
}

//===----------------------------------------------------------------------===//
// FlexiNPU Op Definitions
//===----------------------------------------------------------------------===//

class FlexiNPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<FlexiNPU_Dialect, mnemonic, traits> {}

class FlexiLLVM_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<FlexiNPU_Dialect, mnemonic, traits> {}

def FlexiNPU_DmaLoadOp : FlexiNPU_Op<"dmaload", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "DMA load operation for loading a variable-sized tile";
  let description = [{
    Loads a variable-sized tile from memory into the on-chip buffer.
    The operation requires the memory address, row, column indices, stride, and the tile size.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$srcOfcMemref,
    Arg<I64, "", []>:$oncPtr,
    I32Attr:$elems_per_row,
    I32Attr:$rows,
    I32Attr:$stride,
    BoolAttr:$trans, // make it optional
    FlexiNPUTypesAttr:$type,
    I32Attr:$zero_pad
  );
  // let results = (outs Optional<I1>:$completed);

  let builders =[
    OpBuilder<(ins
      "::mlir::Value":$srcOfcMemref,
      "::mlir::Value":$oncPtr,
      "uint32_t":$elems_per_row,
      "uint32_t":$rows,
      "uint32_t":$stride,
      "bool":$trans,
      "FlexiNPUTypesAttr":$type,
      "uint32_t":$zero_pad
    )>
  ];
}

def FlexiNPU_PreloadOp : FlexiNPU_Op<"preload", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Preload operation for preparing a variable-sized tile for computation";
  let description = [{
    Prepares the given tile in the on-chip buffer for further computation.
    This operation is a no-op for most accelerators but may involve data reorganization.
  }];

  let arguments = (ins
    Arg<I64, "", []>:$oncPtr,
    I32Attr:$elems_per_row,
    I32Attr:$rows,
    FlexiNPUMatTypesAttr:$mat_type,
    FlexiNPUTypesAttr:$type
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$oncPtr,
      "uint32_t":$elems_per_row,
      "uint32_t":$rows,
      "FlexiNPUMatTypesAttr":$mat_type,
      "FlexiNPUTypesAttr":$type
    )>
  ];

}

def FlexiNPU_ExecuteS1Op : FlexiNPU_Op<"execute_s1", [MemoryEffects<[MemRead, MemWrite]>]> { // GEMM
  let summary = "Matrix multiplication operation for variable-sized tiles";
  let description = [{
    Computes the product of two variable-sized tiles and produces a result with the same size.
    The dimensions of the input tiles must match for valid multiplication.
  }];

  let arguments = (ins
    Arg<I64, "", []>:$oncPtr,
    I32Attr:$elems_per_row,
    I32Attr:$rows,
    FlexiNPUMatTypesAttr:$mat_type,
    FlexiNPUTypesAttr:$type,
    FlexiNPUTypesAttr:$res_type
  );

  let results = (outs AnyType:$out);

  let builders =[
    OpBuilder<(ins
      "::mlir::Value":$oncPtr,
      "uint32_t":$elems_per_row,
      "uint32_t":$rows,
      "FlexiNPUMatTypesAttr":$mat_type,
      "FlexiNPUTypesAttr":$type,
      "FlexiNPUTypesAttr":$res_type
    )>
  ];

}

def FlexiNPU_DmaStoreOp : FlexiNPU_Op<"dmastore", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "DMA store operation for storing a variable-sized tile";
  let description = [{
    Stores a variable-sized tile from the on-chip buffer back to memory.
    The operation requires the destination pointer, row, column indices, stride, and the tile size.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$srcOfcMemref,
    Arg<I64, "", []>:$oncPtr,
    I32Attr:$elems_per_row,
    I32Attr:$rows,
    I32Attr:$stride,
    BoolAttr:$trans,
    FlexiNPUTypesAttr:$type,
    I32Attr:$zero_pad
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$srcOfcMemref,
      "::mlir::Value":$oncPtr,
      "uint32_t":$elems_per_row,
      "uint32_t":$rows,
      "uint32_t":$stride,
      "bool":$trans,
      "FlexiNPUTypesAttr":$type,
      "uint32_t":$zero_pad
    )>
  ];
}

def FlexiNPU_FlushOp: FlexiNPU_Op<"flush", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Flush the on-chip buffer";
  let description = [{
    Flushes the on-chip buffer to memory.
    This operation is a no-op for most accelerators but may involve data reorganization.
  }];

  let arguments = (ins
    Arg<I64, "", []>:$onc_ptr,
    I32Attr:$elems_per_row,
    I32Attr:$rows,
    FlexiNPUMatTypesAttr:$mat_type,
    FlexiNPUTypesAttr:$type
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$onc_ptr,
      "uint32_t":$elems_per_row,
      "uint32_t":$rows,
      "FlexiNPUMatTypesAttr":$mat_type,
      "FlexiNPUTypesAttr":$type
    )>
  ];
}

// Single generic ROCC instruction operation for FlexiNPU
def FlexiNPU_RoccInstructionOp : FlexiNPU_Op<"rocc_instruction", [MemoryEffects<[MemWrite]>]> {
  let summary = "Generic FlexiNPU ROCC instruction";
  let description = [{
    Generates a generic ROCC instruction for FlexiNPU.
    Maps to: ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, funct)

    The funct parameter determines the specific ROCC instruction:
    - Based on flexi.h implementations
  }];

  let arguments = (ins
    I64:$rs1,    // First source register value
    I64:$rs2,    // Second source register value
    I32:$funct   // Function code (determines instruction type)
  );

  let assemblyFormat = "$rs1 `,` $rs2 `,` $funct attr-dict";
}

def FlexiNPU_DummyMemrefOp : FlexiNPU_Op<"dummy_memref", [Pure]> {
  let summary = "Creates a dummy memref for special pointer values";
  let description = [{
    Creates a dummy memref with a special pointer value:
    - value = 0: represents null pointer
    - value = 1: represents dummy pointer (void*)1
  }];

  let arguments = (ins
    Index:$value  // 0 for null, 1 for dummy
  );

  let results = (outs
    AnyMemRef:$result
  );

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

#endif // FlexiNPU
