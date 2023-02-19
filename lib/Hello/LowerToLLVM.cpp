// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"
#include "Hello/HelloPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

namespace hello {
class PrintOpLowering : public mlir::ConversionPattern {
public:
  explicit PrintOpLowering(mlir::MLIRContext *context)
    : mlir::ConversionPattern(hello::PrintOp::getOperationName(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto memRefType = (*op->operand_type_begin()).cast<mlir::MemRefType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    mlir::Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", mlir::StringRef("%f \0", 4), parentModule);
    mlir::Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", mlir::StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    mlir::SmallVector<mlir::Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upperBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
      for (mlir::Operation &nested : *loop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1) {
        rewriter.create<mlir::func::CallOp>(loc, printfRef, rewriter.getIntegerType(32), newLineCst);
      }
      rewriter.create<mlir::scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = mlir::cast<hello::PrintOp>(op);
    auto elementLoad = rewriter.create<mlir::memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    rewriter.create<mlir::func::CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                            mlir::ArrayRef<mlir::Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  static mlir::FlatSymbolRefAttr getOrInsertPrintf(mlir::PatternRewriter &rewriter,
                                                   mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
      return mlir::SymbolRefAttr::get(context, "printf");
    }

    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  static mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder &builder,
                                       mlir::StringRef name, mlir::StringRef value,
                                       mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              mlir::LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
            loc, mlir::IntegerType::get(builder.getContext(), 64),
            builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)),
        globalPtr,
        mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};
}

namespace {
class HelloToLLVMLoweringPass
        : public mlir::PassWrapper<HelloToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloToLLVMLoweringPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final;
};
}

void HelloToLLVMLoweringPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());
  mlir::RewritePatternSet patterns(&getContext());

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<hello::PrintOpLowering>(&getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> hello::createLowerToLLVMPass() {
  return std::make_unique<HelloToLLVMLoweringPass>();
}
