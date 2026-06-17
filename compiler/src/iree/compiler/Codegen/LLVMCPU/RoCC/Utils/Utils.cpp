#include "Utils.h"

#include "iree/compiler/Codegen/LLVMCPU/RoCC/FlexiNPUFuncImpl.h"
#include "llvm/Support/Debug.h" // For LLVM_DEBUG
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "iree-llvmcpu-flexinpu-utils"

namespace mlir {
namespace iree_compiler {

bool isConstantZero(Value val, Value outputBuffer) {
  auto constOp = dyn_cast_or_null<arith::ConstantOp>(val.getDefiningOp());
  if (!constOp)
    return false;

  Attribute constValueAttr = constOp.getValue();
  if (auto floatAttr = dyn_cast<FloatAttr>(constValueAttr)) {
    return floatAttr.getValue().isZero();
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(constValueAttr)) {
    auto shapedType = dyn_cast<ShapedType>(outputBuffer.getType());
    if (!shapedType)
      return false;

    Type elementType = shapedType.getElementType();
    return (elementType.isSignlessInteger() || elementType.isSignedInteger() ||
            elementType.isUnsignedInteger() || elementType.isIndex()) &&
           intAttr.getInt() == 0;
  }
  return false;
}

Operation *getOutermostLoop(Operation *op) {
  Operation *currentOp = op;
  Operation *outermostLoop = nullptr;

  while (Operation *parentOp = currentOp->getParentOp()) {
    if (isa<scf::ForOp, scf::ForallOp>(parentOp)) {
      outermostLoop = parentOp;
    } else if (isa<mlir::func::FuncOp>(parentOp)) {
      break;
    }
    currentOp = parentOp;
  }
  return outermostLoop;
}

std::tuple<Value, Value, Value> extractStridedMetadata(
    OpBuilder &builder, Location loc, Value memref,
    int64_t defaultStride) { // defaultStride에 기본값 제거 (헤더에 있음)
  auto stridedMetadata =
      builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
  auto base = stridedMetadata.getBaseBuffer();
  auto offset = stridedMetadata.getOffset();

  Value stride = createConst(builder, loc, static_cast<int32_t>(defaultStride));
  if (auto memrefType = dyn_cast<MemRefType>(memref.getType())) {
    if (memrefType.getRank() == 2) {
      auto stridesArray = stridedMetadata.getStrides();
      if (!stridesArray.empty() && stridesArray[0]) {
        if (auto strideValOp = getConstantIntValue(stridesArray[0])) {
          if (strideValOp.has_value()) {
            stride = createConst(builder, loc,
                                 static_cast<int32_t>(strideValOp.value()));
          }
        }
      }
    }
  }
  return {base, offset, stride};
}

Value ensureI64(OpBuilder &builder, Location loc, Value value) {
    if (value.getType().isInteger(64)) {
      return value;
    } else if (value.getType().isInteger(32)) {
      return builder.create<mlir::arith::ExtUIOp>(loc, builder.getI64Type(),
                                                  value);
    } else if (value.getType().isInteger(1) || value.getType().isInteger(8) ||
               value.getType().isInteger(16)) {
      return builder.create<mlir::arith::ExtUIOp>(loc, builder.getI64Type(),
                                                  value);
    } else if (value.getType().isF32()) {
      // Handle float by bitcast to i32 then extend to i64
      Value bitcasted = builder.create<mlir::arith::BitcastOp>(
          loc, builder.getI32Type(), value);
      return builder.create<mlir::arith::ExtUIOp>(loc, builder.getI64Type(),
                                                  bitcasted);
    } else {
      llvm_unreachable("Unsupported type in BitFieldBuilder");
    }
  }

Value extractMemrefPtr(OpBuilder &builder, Location loc, Value memref,
                              Value offset) {
  Value base_index =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
  Value base_i64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), base_index);
  Value offset_i64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), offset);
  return builder.create<arith::AddIOp>(loc, base_i64, offset_i64);
}

template Value createConst<long>(OpBuilder &builder, Location loc, long value);
template Value createConst<float>(OpBuilder &builder, Location loc,
                                  float value);
template Value createConst<int>(OpBuilder &builder, Location loc, int value);
template Value createConst<uint64_t>(OpBuilder &builder, Location loc, uint64_t value);

//===----------------------------------------------------------------------===//
// FlexiNPU Utility Functions
//===----------------------------------------------------------------------===//

// Utility function to inject FlexiNPU ROCC instruction
void injectFlexiNPURocc(OpBuilder &builder, Location loc, Value rs1, Value rs2,
                        int32_t funct, bool extend) {
  if (extend) {
    rs1 = ensureI64(builder, loc, rs1);
    rs2 = ensureI64(builder, loc, rs2);
  }
  Value functVal = createConst<int32_t>(builder, loc, funct);

  builder.create<::flexinpu::RoccInstructionOp>(loc, rs1, rs2, functVal);
}

flexinpu::FlexiNPUTypes convertFlexiNPUType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() == 4)
      return flexinpu::FlexiNPUTypes::i4;
    if (intType.getWidth() == 8)
      return flexinpu::FlexiNPUTypes::i8; // INT8
  }
  if (type.isInteger(32))
    return flexinpu::FlexiNPUTypes::i32;   // INT32
  if (type.isInteger(64))
    return flexinpu::FlexiNPUTypes::i64;   // INT64
  if (type.isF16())
    return flexinpu::FlexiNPUTypes::f16;   // FP16
  if (type.isF32())
    return flexinpu::FlexiNPUTypes::f32;   // FP32
  if (type.isF64())
    return flexinpu::FlexiNPUTypes::f64;   // FP64
  if (type.isBF16())
    return flexinpu::FlexiNPUTypes::bf16;   // BF16
  // Default to FP32 if type not recognized
  return flexinpu::FlexiNPUTypes::f32;     // FP32
}

} // namespace iree_compiler
} // namespace mlir
