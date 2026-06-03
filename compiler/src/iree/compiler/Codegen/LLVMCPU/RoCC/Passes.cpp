#include "iree/compiler/Codegen/LLVMCPU/RoCC/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "iree-llvmcpu-rocc-pass-pipelines"

namespace mlir::iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/LLVMCPU/RoCC/Passes.h.inc"
} // namespace

void registerCodegenLLVMCPURoCCPasses() {
  // Generated.
  registerPasses();
}

} //namespace mlir::iree_compiler
