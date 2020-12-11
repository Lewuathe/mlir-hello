#include <iostream>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Sample/SampleDialect.h"
#include "Sample/SamplePasses.h"

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int dumpLLVMIR(mlir::ModuleOp module) {
  llvm::LLVMContext llvmContext;

  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  return 0;
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  if (int error = loadMLIR(context, module)) {
    return error;
  }

  std::cout << "CCCCCC" << std::endl;
  mlir::PassManager passManager(&context);
  passManager.addPass(sample::createLowerToLLVMPass());

  if (mlir::failed(passManager.run(*module))) {
    return 4;
  }

  return 0;
}

int main(int argc, char **argv) {
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "Sample compiler\n");
  mlir::registerAllPasses();
  // TODO: Register sample passes here.

  mlir::DialectRegistry registry;
  registry.insert<sample::SampleDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  mlir::registerAllDialects(registry);
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  mlir::MLIRContext context;
  context.getOrLoadDialect<sample::SampleDialect>();
  mlir::OwningModuleRef module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  return 0;
}
