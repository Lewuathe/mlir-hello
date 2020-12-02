#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "Sample/SampleDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  // TODO: Register sample translations here.

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
