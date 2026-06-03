#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_FUNCUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_FUNCUTILS_H_

#include <string>
#include <tuple>
#include <vector>
#include "Utils.h"
#include "ParamUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {

class BitFieldBuilder {
private:
  OpBuilder &builder;
  Location loc;
  Value result;

public:
  BitFieldBuilder(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc), result(nullptr) {}

  template <typename T>
  BitFieldBuilder &cat(T value, int shift) {
    Value val;
    if constexpr (std::is_same_v<T, Value>) {
      val = ensureI64(builder, loc,value);
    } else {
      val = ensureI64(builder, loc, createConst(builder, loc, value));
    }

    Value shifted = builder.create<mlir::arith::ShLIOp>(
        loc, val, createConst<int64_t>(builder, loc, shift));

    if (result) {
      result = builder.create<mlir::arith::OrIOp>(loc, result, shifted);
    } else {
      result = shifted;
    }
    return *this;
  }

  // Overload for ConstParam
  template <typename T>
  BitFieldBuilder &cat(const ConstParam<T> &param, int shift) {
    return cat(param.getArg(), shift);
  }

  operator Value() const {
    return result ? result : createConst<int64_t>(builder, loc, 0);
  }
};

// Base class for FlexiNPU helper function parameter builders.
template <typename Derived>
class RoCCFuncParams {
protected:
  OpBuilder &builder;
  Location loc;
  func::FuncOp funcOp;

public:
  RoCCFuncParams(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}
  virtual ~RoCCFuncParams() = default;
  virtual std::string getFuncName() = 0;
  virtual void impl() = 0;

  func::FuncOp getOrDeclareFunc() {
    if (funcOp)
      return funcOp;

    ModuleOp module =
        builder.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>();

    std::string funcName = getFuncName();
    funcOp = module.lookupSymbol<func::FuncOp>(funcName);

    if (!funcOp) {
      std::vector<Type> argTypes = getArgTypes(builder);
      auto funcType = builder.getFunctionType(argTypes, {});

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(module.getBody());
      funcOp = builder.create<func::FuncOp>(loc, funcName, funcType);
      funcOp.setPrivate();

      Block *entryBlock = funcOp.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);
      bindArgs(entryBlock->getArguments());

      impl();
      builder.create<func::ReturnOp>(loc);
    }

    return funcOp;
  }

  void for_each_param(std::function<void(BaseParam &)> fn) {
    auto &self = static_cast<Derived &>(*this);
    std::apply([&](auto &...param) { (fn(param), ...); }, self.as_tuple());
  }

  std::vector<Type> getArgTypes(OpBuilder &builder) {
    std::vector<Type> argTypes;
    for_each_param(
        [&](BaseParam &param) { argTypes.push_back(param.getType(builder)); });
    return argTypes;
  }

  void bindArgs(ArrayRef<BlockArgument> args) {
    int idx = 0;
    for_each_param([&](BaseParam &param) { param.setArg(args[idx++]); });
  }

  func::CallOp call() {
    func::FuncOp funcOp = getOrDeclareFunc();
    std::vector<Value> callArgs;
    for_each_param([&](BaseParam &param) {
      if (!param.getValue()) {
        callArgs.push_back(param.build(builder, loc));
      } else {
        callArgs.push_back(param.getValue());
      }
    });
    return builder.create<func::CallOp>(loc, funcOp, callArgs);
  }
};
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_FUNCUTILS_H_
