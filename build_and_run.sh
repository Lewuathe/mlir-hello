# This script helps you set up MLIR
# Prerequisites: cmake, c compiler, make
# also, make sure you do `git submodule update --init --recursive`
# in this repo to get llvm-project under thirdparty

# Set up MLIR
LLVM_REPO=./thirdparty/llvm-project
BUILD_DIR=$LLVM_REPO/build
INSTALL_DIR=$LLVM_REPO/install

rm -r $BUILD_DIR
mkdir $BUILD_DIR
rm -r $INSTALL_DIR
mkdir $INSTALL_DIR
set -e
cmake "-H$LLVM_REPO/llvm" \
     "-B$BUILD_DIR" \
     -DLLVM_INSTALL_UTILS=ON \
     -DLLVM_ENABLE_PROJECTS="mlir;clang" \
     -DLLVM_INCLUDE_TOOLS=ON \
     -DLLVM_BUILD_EXAMPLES=ON \
     -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_ENABLE_RTTI=ON \
     -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
                 #  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

cmake --build $BUILD_DIR --target check-mlir -j 10
#cmake --build $BUILD_DIR --target install -j 10
pushd $BUILD_DIR
make lli # lli needs to be build separately for testing
popd


# set up mlir-hello
mkdir build && cd build
cmake -G Ninja .. \
  -DLLVM_DIR=$LLVM_REPO/build/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_REPO/build/lib/cmake/mlir \

cmake --build . --target hello-opt

# test run

./build/bin/hello-opt ./test/Hello/print.mlir > print.ll
$BUILD_DIR/bin/lli print.ll
