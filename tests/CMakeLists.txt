add_executable(
    test
    test.cpp
    quadrature_rule.cpp
    basis.cpp
    gmres.cpp
    stiffness.cpp
    mass.cpp
    load_unstructured_square.cpp
    linalg.cpp
)
target_compile_options(test PRIVATE -x=cu -dc)
target_link_libraries(test PUBLIC cuddh)