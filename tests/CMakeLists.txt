add_executable(
    test
    test.cpp
    quadrature_rule.cpp
)
target_link_libraries(test PUBLIC cuddh)