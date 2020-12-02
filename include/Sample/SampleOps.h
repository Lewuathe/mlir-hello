#ifndef SAMPLE_SAMPLEOPS_H
#define SAMPLE_SAMPLEOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Sample/SampleOps.h.inc"

#endif // SAMPLE_SAMPLEOPS_H
