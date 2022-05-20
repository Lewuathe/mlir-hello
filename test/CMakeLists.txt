configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(HELLO_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        hello-opt
        )

add_lit_testsuite(check-hello "Running the hello regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${HELLO_TEST_DEPENDS}
        )
set_target_properties(check-hello PROPERTIES FOLDER "Tests")

add_lit_testsuites(HELLO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${HELLO_TEST_DEPENDS})
