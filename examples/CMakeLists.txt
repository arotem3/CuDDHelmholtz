##### --- examples --- #####

##### Helmholtz #####
add_executable(Helmholtz Helmholtz.cpp)
target_link_libraries(Helmholtz PUBLIC cuddh)
add_custom_command(TARGET Helmholtz POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### Poisson #####
add_executable(Poisson Poisson.cpp)
target_link_libraries(Poisson PUBLIC cuddh)
add_custom_command(TARGET Poisson POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)
