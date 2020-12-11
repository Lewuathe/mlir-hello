#ifndef SAMPLE_DIALECT
#define SAMPLE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Sample dialect definition.
//===----------------------------------------------------------------------===//

def Sample_Dialect : Dialect {
    let name = "sample";
    let summary = "A sample out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::sample";
}

//===----------------------------------------------------------------------===//
// Base sample operation definition.
//===----------------------------------------------------------------------===//

class Sample_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Sample_Dialect, mnemonic, traits>;

#endif // SAMPLE_DIALECT
