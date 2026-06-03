#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_UTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_UTILS_H_

#include <tuple> // For std::tuple
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace iree_compiler {

template <typename T>
Value createConst(OpBuilder &builder, Location loc, T value) {
  if constexpr (std::is_same_v<T, int32_t>) {
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getI32IntegerAttr(value));
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getI64IntegerAttr(value));
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getI64IntegerAttr(static_cast<int64_t>(value)));
  } else if constexpr (std::is_same_v<T, float>) {
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getF32FloatAttr(value));
  } else if constexpr (std::is_same_v<T, bool>) {
    return builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(value));
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getI8IntegerAttr(value));
  } else {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      return builder.create<arith::ConstantOp>(
          loc, builder.getI32IntegerAttr(static_cast<int32_t>(value)));
    }
    llvm_unreachable(
        "Unsupported constant type passed to createConst that is not integral");
  }
}

Value ensureI64(OpBuilder &builder, Location loc, Value value);

Value extractMemrefPtr(OpBuilder &builder, Location loc, Value memref,
                              Value offset);

void injectFlexiNPURocc(OpBuilder &builder, Location loc, Value rs1, Value rs2,
                        int32_t funct, bool extend = false);

bool isConstantZero(Value val, Value outputBuffer);

flexinpu::FlexiNPUTypes convertFlexiNPUType(Type type);

Operation *getOutermostLoop(Operation *op);

std::tuple<Value, Value, Value>
extractStridedMetadata(OpBuilder &builder, Location loc, Value memref,
                       int64_t defaultStride = 0);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_UTILS_H_
