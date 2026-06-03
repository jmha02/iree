#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace flexinpu {

void DmaLoadOp::build(OpBuilder &builder, OperationState &state,
                      Value srcOfcMemref, Value oncPtr, uint32_t elems_per_row,
                      uint32_t rows, uint32_t stride, bool trans,
                      FlexiNPUTypesAttr type, uint32_t zero_pad) {
  DmaLoadOp::build(builder, state, TypeRange{}, srcOfcMemref, oncPtr,
                   builder.getI32IntegerAttr(elems_per_row),
                   builder.getI32IntegerAttr(rows),
                   builder.getI32IntegerAttr(stride),
                   builder.getBoolAttr(trans), type,
                   builder.getI32IntegerAttr(zero_pad));
}

void DmaStoreOp::build(OpBuilder &builder, OperationState &state,
                       Value srcOfcMemref, Value oncPtr,
                       uint32_t elems_per_row, uint32_t rows, uint32_t stride,
                       bool trans, FlexiNPUTypesAttr type,
                       uint32_t zero_pad) {
  DmaStoreOp::build(builder, state, TypeRange{}, srcOfcMemref, oncPtr,
                    builder.getI32IntegerAttr(elems_per_row),
                    builder.getI32IntegerAttr(rows),
                    builder.getI32IntegerAttr(stride),
                    builder.getBoolAttr(trans), type,
                    builder.getI32IntegerAttr(zero_pad));
}

void ExecuteS1Op::build(OpBuilder &builder, OperationState &state, Value oncPtr,
                        uint32_t elems_per_row, uint32_t rows,
                        FlexiNPUMatTypesAttr mat_type,
                        FlexiNPUTypesAttr type, FlexiNPUTypesAttr res_type) {
  ExecuteS1Op::build(builder, state, builder.getI32Type(), oncPtr,
                     builder.getI32IntegerAttr(elems_per_row),
                     builder.getI32IntegerAttr(rows), mat_type, type, res_type);
}

void FlushOp::build(OpBuilder &builder, OperationState &state, Value onc_ptr,
                    uint32_t elems_per_row, uint32_t rows,
                    FlexiNPUMatTypesAttr mat_type, FlexiNPUTypesAttr type) {
  FlushOp::build(builder, state, TypeRange{}, onc_ptr,
                 builder.getI32IntegerAttr(elems_per_row),
                 builder.getI32IntegerAttr(rows), mat_type, type);
}

void PreloadOp::build(OpBuilder &builder, OperationState &state, Value oncPtr,
                      uint32_t elems_per_row, uint32_t rows,
                      FlexiNPUMatTypesAttr mat_type, FlexiNPUTypesAttr type) {
  PreloadOp::build(builder, state, TypeRange{}, oncPtr,
                   builder.getI32IntegerAttr(elems_per_row),
                   builder.getI32IntegerAttr(rows), mat_type, type);
}

}
