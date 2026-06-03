include "mlir/Pass/PassBase.td"

def FlexiNPUConvertToFunctionCallPass :
  Pass<"iree-flexinpu-convert-to-function-call", "::mlir::func::FuncOp"> {
  let summary = "Pass to convert FlexiNPU ops to function calls with implementations.";
  let dependentDialects = [
    "flexinpu::FlexiNPUDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
  ];
}

def FlexiNPUTensorizePass :
  Pass<"iree-flexinpu-tensorize", "::mlir::func::FuncOp"> {
  let summary = "Pass to lower linalg matmul to FlexiNPU ops with new kernel code structure.";
  let dependentDialects = [
    "flexinpu::FlexiNPUDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
  ];
}
