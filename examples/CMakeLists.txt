##### --- examples --- #####

##### DDH #####
add_executable(DDH DDH.cpp)
target_link_libraries(DDH PUBLIC cuddh)
add_custom_command(TARGET DDH POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### Helmholtz #####
add_executable(Helmholtz Helmholtz.cpp)
target_link_libraries(Helmholtz PUBLIC cuddh)
add_custom_command(TARGET Helmholtz POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### Poisson #####
add_executable(Poisson Poisson.cpp)
target_link_libraries(Poisson PUBLIC cuddh)
add_custom_command(TARGET Poisson POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### WaveHoltz #####
add_executable(WaveHoltz WaveHoltz.cpp)
target_link_libraries(WaveHoltz PUBLIC cuddh)
add_custom_command(TARGET WaveHoltz POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)
