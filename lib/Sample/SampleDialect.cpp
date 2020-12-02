#include "Sample/SampleDialect.h"
#include "Sample/SampleOps.h"

using namespace mlir;
using namespace mlir::sample;

//===----------------------------------------------------------------------===//
// Sample dialect.
//===----------------------------------------------------------------------===//

void SampleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Sample/SampleOps.cpp.inc"
      >();
}
