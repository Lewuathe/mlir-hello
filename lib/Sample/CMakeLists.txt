add_mlir_dialect_library(MLIRSample
		SampleDialect.cpp
		SampleOps.cpp
		LowerToAffine.cpp
		LowerToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Sample

        DEPENDS
        MLIRSampleOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
