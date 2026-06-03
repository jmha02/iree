#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

// Enum and Attribute Definitions
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUEnums.cpp.inc"
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUAttrDefs.cpp.inc"

// Operation Definitions
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUOps.cpp.inc"

#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUDialect.cpp.inc"

using namespace mlir;
namespace flexinpu {
void FlexiNPUDialect::initialize() {
  // Dialect-specific initialization here
  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Dialect/FlexiNPU/IR/FlexiNPUOps.cpp.inc"
    >();
}

mlir::Attribute FlexiNPUDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // Add your attribute parsing logic here.
  parser.emitError(parser.getCurrentLocation(), "attribute parsing not implemented");
  return {};
}

// Printing attributes.
void FlexiNPUDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // Add your attribute printing logic here.
  printer << "<attribute printing not implemented>";
}

// Parsing types.
mlir::Type FlexiNPUDialect::parseType(DialectAsmParser &parser) const {
  // Add your type parsing logic here.
  parser.emitError(parser.getCurrentLocation(), "type parsing not implemented");
  return {};
}

// Printing types.
void FlexiNPUDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Add your type printing logic here.
  printer << "<type printing not implemented>";
}
} // namespace flexinpu
// FlexiNPUDialect::initialize
// flexinpu::FlexiNPUDialect::parseAttribute(mlir::DialectAsmParser&, mlir::Type) const
// flexinpu::FlexiNPUDialect::printAttribute(mlir::Attribute, mlir::DialectAsmPrinter&) const
// flexinpu::FlexiNPUDialect::parseType(mlir::DialectAsmParser&) const
// flexinpu::FlexiNPUDialect::printType(mlir::Type, mlir::DialectAsmPrinter&) const


/*
class FlexiNPUDialect : public ::mlir::Dialect {
  explicit FlexiNPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~FlexiNPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("flexinpu");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
};
} // namespace flexinpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::flexinpu::FlexiNPUDialect)

*/