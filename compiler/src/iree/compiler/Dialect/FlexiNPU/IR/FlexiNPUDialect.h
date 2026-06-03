//===- FlexiNPUDialect.h - FlexiNPU Dialect -------------------*- C++ -*-===//
//
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FlexiNPU Dialect and its associated operations,
// types, and attributes.
//
//===----------------------------------------------------------------------===//

#ifndef FLEXINPU_DIALECT_H
#define FLEXINPU_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUAttrDefs.h.inc"
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUOps.h.inc"

#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.h.inc"


#endif // FLEXINPU_DIALECT_H