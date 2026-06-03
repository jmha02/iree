# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
iree_add_all_subdirs()
# iree_setup_c_src_root(
#   PACKAGE_ROOT_PREFIX iree::target
#   IMPLICIT_DEFS_TARGET iree_target_flexinpu_compiler_defs
# )
###############################################################################
# FlexiNPU Dialect
###############################################################################

iree_tablegen_library(
  NAME
    FlexiNPUOpsGen
  TBLGEN
    MLIR
  TD_FILE
    FlexiNPU.td
  OUTS
    --gen-dialect-decls --dialect=flexinpu FlexiNPUDialect.h.inc
    --gen-dialect-defs --dialect=flexinpu FlexiNPUDialect.cpp.inc
    --gen-op-decls FlexiNPUOps.h.inc
    --gen-op-defs FlexiNPUOps.cpp.inc
    --gen-attrdef-decls --attrdefs-dialect=flexinpu FlexiNPUAttrDefs.h.inc
    --gen-attrdef-defs --attrdefs-dialect=flexinpu FlexiNPUAttrDefs.cpp.inc
    --gen-enum-decls FlexiNPUEnums.h.inc
    --gen-enum-defs FlexiNPUEnums.cpp.inc
    # --gen-llvmir-conversions FlexiNPUConversions.h.inc
)

iree_cc_library(
  NAME
    IR
  HDRS
    "FlexiNPUDialect.h"
    "FlexiNPUDialect.h.inc"
    "FlexiNPUOps.h.inc"
    "FlexiNPUAttrDefs.h.inc"
    "FlexiNPUEnums.h.inc"
  SRCS
    "FlexiNPUDialect.cpp"
    "FlexiNPUDialect.cpp.inc"
    "FlexiNPUOps.cpp"
    "FlexiNPUOps.cpp.inc"
    "FlexiNPUAttrDefs.cpp.inc"
    "FlexiNPUEnums.cpp.inc"
  DEPS
    ::FlexiNPUOpsGen
    LLVMSupport
    MLIRArithUtils
    MLIRDialectUtils
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRViewLikeInterface
  PUBLIC
)
# FTODO: add FLLVM
