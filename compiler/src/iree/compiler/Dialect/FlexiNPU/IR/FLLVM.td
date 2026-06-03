// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FLLVM
#define FLLVM

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

def LLVM_PointerHost: LLVM_PointerInAddressSpace<0>;
def LLVM_PointerNPU: LLVM_PointerInAddressSpace<1>;

def FLLVM_Dialect : Dialect {
  let name = "fllvm";
  let cppNamespace = "::fllvm"; // TODO
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let description = [{
    TODO
  }];
  // TODO: extraClassDeclaration
}

//===----------------------------------------------------------------------===//
// FLLVM Attribute Definitions
//===----------------------------------------------------------------------===//
def FLLVMTypeI8 : I32EnumAttrCase<"i8", 0>;
def FLLVMTypeI32 : I32EnumAttrCase<"i32", 1>;
def FLLVMTypeI64 : I32EnumAttrCase<"i64", 2>;
def FLLVMTypeF16: I32EnumAttrCase<"f16", 3>;
def FLLVMTypeF32: I32EnumAttrCase<"f32", 4>;
def FLLVMTypeF64: I32EnumAttrCase<"f64", 5>;
def FLLVMTypeBF16: I32EnumAttrCase<"bf16", 6>;

def FLLVMMatTypeA: I32EnumAttrCase<"A", 0>;
def FLLVMMatTypeB: I32EnumAttrCase<"B", 1>;
def FLLVMMatTypeC: I32EnumAttrCase<"C", 2>;
def FLLVMMatTypeD: I32EnumAttrCase<"D", 3>;
//===----------------------------------------------------------------------===//
// FLLVM Type Definitions
//===----------------------------------------------------------------------===//

def FLLVMTypes: I32EnumAttr<"FLLVMTypes", "FLLVM Types", [
  FLLVMTypeI8, FLLVMTypeI32, FLLVMTypeI64, FLLVMTypeF16,
  FLLVMTypeF32, FLLVMTypeF64, FLLVMTypeBF16
]> {
  let cppNamespace = "::fllvm";
}
def FLLVMTypesAttr: EnumAttr<FLLVM_Dialect, FLLVMTypes, "fllvm_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def FLLVMMatTypes: I32EnumAttr<"FLLVMMatTypes", "FLLVM Matrix Types", [
  FLLVMMatTypeA, FLLVMMatTypeB, FLLVMMatTypeC, FLLVMMatTypeD
]> {
  let cppNamespace = "::fllvm";
}
def FLLVMMatTypesAttr: EnumAttr<FLLVM_Dialect, FLLVMMatTypes, "fllvm_mat_type"> {
  let assemblyFormat = "`<` $value `>`";
}


//===----------------------------------------------------------------------===//
// FLLVM Op Definitions
//===----------------------------------------------------------------------===//

class FLLVM_Op<string mnemonic, list<Trait> traits = []> :
  Op<FLLVM_Dialect, mnemonic, traits> {}

def FLLVM_DmaLoadOp : FLLVM_Op<"dmaload", [MemoryEffects<[MemRead]>]> {
  let summary = "DMA load operation for loading a variable-sized tile";
  let description = [{
    Loads a variable-sized tile from memory into the on-chip buffer.
    The operation requires the memory address, row, column indices, stride, and the tile size.
  }];

  let arguments = (ins
    LLVM_PointerHost:$ofc_ptr,
    LLVM_PointerNPU:$onc_ptr,
    I32:$elems_per_row,
    I32:$rows,
    I32:$stride,
    BoolAttr:$trans, // make it optional
    FLLVMTypesAttr:$type,
    I32:$zero_pad
  );

  // TODO: results, llvmBuilder
  // string llvmBuilder = [{
  //   mlir::Type hostPtrType = opInst.getOperand(0)->getType();
  //   mlir::Type npuPtrType = opInst.getOperand(1)->getType();
  //   llvm::Type *hostLLVMPtrTy = moduleTranslation.translateType(hostPtrType).cast<llvm::Type *>();
  //   llvm::Type *npuLLVMPtrTy = moduleTranslation.translateType(npuPtrType).cast<llvm::Type *>();

  //   llvm::Function *memcpyFunc = moduleTranslation.getModule().getFunction("llvm.memcpy.p0i8.p0i8.i32");
  //   if (!memcpyFunc) {
  //     llvm::FunctionType *memcpyType = llvm::FunctionType::get(
  //       llvm::Type::getVoidTy(moduleTranslation.getLLVMContext()),
  //       {llvm::Type::getInt8PtrTy(moduleTranslation.getLLVMContext()),
  //        llvm::Type::getInt8PtrTy(moduleTranslation.getLLVMContext()),
  //        llvm::Type::getInt32Ty(moduleTranslation.getLLVMContext()),
  //        llvm::Type::getInt1Ty(moduleTranslation.getLLVMContext())},
  //       false);
  //     memcpyFunc = llvm::Function::Create(memcpyType, llvm::Function::ExternalLinkage, "llvm.memcpy.p0i8.p0i8.i32", &moduleTranslation.getModule());
  //   }
  //   llvm::Value *hostPtr = builder.CreateBitCast($ofc_ptr, llvm::Type::getInt8PtrTy(moduleTranslation.getLLVMContext()));
  //   llvm::Value *npuPtr = builder.CreateBitCast($onc_ptr, llvm::Type::getInt8PtrTy(moduleTranslation.getLLVMContext()));

  //   llvm::Value *size = builder.getInt32($elems_per_row);
  //   builder.CreateCall(memcpyFunc, {hostPtr, npuPtr, size, builder.getInt1(false)});
  // }];
}

def FLLVM_PreloadOp : FLLVM_Op<"preload", [MemoryEffects<[MemRead]>]> {
  let summary = "Preload operation for preparing a variable-sized tile for computation";
  let description = [{
    Prepares the given tile in the on-chip buffer for further computation.
    This operation is a no-op for most accelerators but may involve data reorganization.
  }];

  let arguments = (ins
    LLVM_PointerNPU:$ptr, // PointerNPU right?
    I32:$elems_per_row,
    I32:$rows,
    I32:$mat_type, // ?
    FLLVMTypesAttr:$type
  );

  // TODO: results, llvmBuilder
}

def FLLVM_ExecuteS1Op : FLLVM_Op<"execute_s1"> { // GEMM
  let summary = "Matrix multiplication operation for variable-sized tiles";
  let description = [{
    Computes the product of two variable-sized tiles and produces a result with the same size.
    The dimensions of the input tiles must match for valid multiplication.
  }];

  let arguments = (ins
    LLVM_PointerNPU:$ptr,
    I32:$elems_per_row,
    I32:$rows,
    I32:$mat_type,
    FLLVMTypesAttr:$type,
    FLLVMTypesAttr:$res_type
  );

  // TODO: results, llvmBuilder
}


def FLLVM_DmaStoreOp : FLLVM_Op<"dmastore"> {
  let summary = "DMA store operation for storing a variable-sized tile";
  let description = [{
    Stores a variable-sized tile from the on-chip buffer back to memory.
    The operation requires the destination pointer, row, column indices, stride, and the tile size.
  }];

  let arguments = (ins
    LLVM_PointerHost:$ofc_ptr,
    LLVM_PointerNPU:$onc_ptr,
    I32:$elems_per_row,
    I32:$rows,
    I32:$stride,
    BoolAttr:$trans,
    FLLVMTypesAttr:$type,
    I32:$zero_pad
  );

  // TODO: results, llvmBuilder
}

def FLLVM_FlushOp: FLLVM_Op<"flush"> {
  let summary = "Flush the on-chip buffer";
  let description = [{
    Flushes the on-chip buffer to memory.
    This operation is a no-op for most accelerators but may involve data reorganization.
  }];

  let arguments = (ins
    LLVM_PointerNPU:$ptr,
    I32:$elems_per_row,
    I32:$rows,
    I32:$mat_type,
    FLLVMTypesAttr:$type
  );

  // TODO: results, llvmBuilder
}


#endif // FLLVM