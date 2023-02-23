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

namespace hello {
    class WorldOpLowering : public mlir::ConversionPattern {
    public:
        explicit WorldOpLowering(mlir::MLIRContext *context) : mlir::ConversionPattern(
                hello::WorldOp::getOperationName(), 1, context) {}

        mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                            mlir::ArrayRef <mlir::Value> operands,
                                            mlir::ConversionPatternRewriter &rewriter) const override {

        }

    private:
        static mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder &builder,
                                                   mlir::StringRef name, mlir::StringRef value,
                                                   mlir::ModuleOp module) {
            // Create the global at the entry of the module.
            mlir::memref::GlobalOp global;
            if (!(global = module.lookupSymbol<mlir::memref::GlobalOp>(name))) {
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
    }
}