get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSample
        )
add_llvm_executable(sample-opt sample-opt.cpp)

llvm_update_compile_flags(sample-opt)
target_link_libraries(sample-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(sample-opt)
