#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_PASSES_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/LLVMCPU/RoCC/Passes.h.inc"

void registerCodegenLLVMCPURoCCPasses();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_PASSES_H_
