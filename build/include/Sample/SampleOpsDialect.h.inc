/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace sample {

class SampleDialect : public ::mlir::Dialect {
  explicit SampleDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<SampleDialect>()) {
    
    initialize();
  }
  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static ::llvm::StringRef getDialectNamespace() { return "sample"; }
};
} // namespace sample
} // namespace mlir
