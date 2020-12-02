configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SAMPLE_TEST_DEPENDS
        FileCheck count not
        sample-opt
        sample-translate
        )

add_lit_testsuite(check-sample "Running the sample regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SAMPLE_TEST_DEPENDS}
        )
set_target_properties(check-sample PROPERTIES FOLDER "Tests")

add_lit_testsuites(SAMPLE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SAMPLE_TEST_DEPENDS})
