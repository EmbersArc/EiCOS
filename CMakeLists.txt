cmake_minimum_required(VERSION 3.5)
project(ecos_eigen)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt)

set(ECOS_INCLUDE
    include
    test
)

set(ECOS_SOURCES
    src/ecos.cpp
    test/ecostester.cpp
)

add_library(ecos_eigen SHARED ${ECOS_SOURCES})
target_include_directories(ecos_eigen PUBLIC ${ECOS_INCLUDE})

set(DEBUG_OPTIONS -Wall -Wextra -Wpedantic)
set(RELEASE_OPTIONS -O2 -march=native)
target_compile_options(ecos_eigen PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options(ecos_eigen PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")

target_link_libraries(ecos_eigen Eigen3::Eigen fmt::fmt)

add_executable(run src/run.cpp)
target_link_libraries(run ecos_eigen)

add_executable(testecos test/ecostester.cpp)
target_link_libraries(testecos ecos_eigen)
