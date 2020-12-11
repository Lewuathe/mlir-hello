#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Sample/SampleDialect.h"
#include "Sample/SampleOps.h"

using namespace mlir;
using namespace sample;

//===----------------------------------------------------------------------===//
// Sample dialect.
//===----------------------------------------------------------------------===//

void SampleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Sample/SampleOps.cpp.inc"
      >();
}

void sample::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  sample::ConstantOp::build(builder, state, dataType, dataAttribute);
}

//static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
//                                         mlir::OperationState &result) {
//  mlir::DenseElementsAttr value;
//  if (parser.parseOptionalAttrDict(result.attributes) ||
//      parser.parseAttribute(value, "value", result.attributes))
//    return failure();
//
//  result.addTypes(value.getType());
//  return success();
//}

//
///// The 'OpAsmPrinter' class is a stream that allows for formatting
///// strings, attributes, operands, types, etc.
//static void print(mlir::OpAsmPrinter &printer, sample::ConstantOp op) {
//  printer << "sample.constant ";
//  printer.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});
//  printer << op.value();
//}

