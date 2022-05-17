#ifndef HELLO_DIALECT
#define HELLO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Hello dialect definition.
//===----------------------------------------------------------------------===//

def Hello_Dialect : Dialect {
    let name = "hello";
    let summary = "A hello out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement hello-world kind of sample code
        for MLIR.
    }];
    let cppNamespace = "::hello";
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base hello operation definition.
//===----------------------------------------------------------------------===//

class Hello_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hello_Dialect, mnemonic, traits>;

#endif // HELLO_DIALECT
