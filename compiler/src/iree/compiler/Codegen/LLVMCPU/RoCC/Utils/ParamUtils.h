#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_PARAMUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_PARAMUTILS_H_

#include <string>
#include <tuple>
#include <vector>
#include "Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {

struct BaseParam {
  virtual std::string name() const = 0;
  virtual std::string value_as_string() const = 0;
  virtual ~BaseParam() = default;
  virtual Value build(OpBuilder &builder, Location loc) = 0;
  virtual mlir::Type getType(mlir::OpBuilder &builder) const = 0;
  virtual void setValue(Value ssr) = 0;
  virtual Value getValue() const = 0;
  virtual void setArg(mlir::Value arg) = 0;
  virtual mlir::Value getArg() const = 0;
  // decltype(auto) getVal() const;
};

template <typename T>
class ConstParam : public BaseParam {
  T val;
  Value ssr;
  Value arg;
  std::string param_name;

public:
  ConstParam() = default;
  ConstParam(T val, std::string name = "")
      : val(val), param_name(std::move(name)) {}

  ConstParam &operator=(T newVal) {
    val = newVal;
    return *this;
  }

  std::string value_as_string() const override {
    if constexpr (std::is_same_v<T, float>)
      return std::to_string(val);
    else
      return std::to_string(static_cast<int64_t>(val));
  }

  mlir::Type getType(mlir::OpBuilder &builder) const override {
    using namespace mlir;
    if constexpr (std::is_same_v<T, int32_t>) {
      return builder.getI32Type();
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return builder.getI64Type();
    } else if constexpr (std::is_same_v<T, float>) {
      return builder.getF32Type();
    } else if constexpr (std::is_same_v<T, double>) {
      return builder.getF64Type();
    } else if constexpr (std::is_same_v<T, bool>) {
      return builder.getI1Type();
    } else {
      static_assert(sizeof(T) == 0,
                    "Unsupported ConstParam<T>::getType specialization");
    }
  }

  std::string name() const override { return this->param_name; }

  Value build(OpBuilder &builder, Location loc) override {
    using namespace mlir;
    if (ssr) {
      return ssr;
    }
    return createConst<T>(builder, loc, val);
  }

  void setValue(Value ssr) override { this->ssr = ssr; }

  mlir::Value getValue() const override { return ssr; }

  mlir::Value getArg() const override { return arg; }
  void setArg(mlir::Value arg) override { this->arg = arg; }
};

class ValueParam : public BaseParam {
  Value val;
  Value arg;
  std::string param_name;

public:
  ValueParam() = default;
  ValueParam(Value val, std::string name = "")
      : val(val), param_name(std::move(name)) {}

  ValueParam &operator=(Value newVal) {
    val = newVal;
    return *this;
  }

  operator Value() const { return val; }
  std::string name() const override { return param_name; }
  std::string value_as_string() const override { return "value"; }
  mlir::Type getType(mlir::OpBuilder &builder) const override {
    return val.getType();
  }
  Value build(OpBuilder &builder, Location loc) override { return val; }
  void setValue(Value ssr) override {}
  Value getValue() const override { return val; }

  mlir::Value getArg() const override { return arg; }
  void setArg(mlir::Value arg) override { this->arg = arg; }
};
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_ROCC_UTILS_PARAMUTILS_H_
