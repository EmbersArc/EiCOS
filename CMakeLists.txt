cmake_minimum_required(VERSION 3.5)
project(eicos)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt)

set(EICOS_INCLUDE
    include
    test
)

set(EICOS_SOURCES
    src/eicos.cpp
    test/ecostester.cpp
)

add_library(eicos SHARED ${EICOS_SOURCES})
target_include_directories(eicos PUBLIC ${EICOS_INCLUDE})

set(DEBUG_OPTIONS -Wall -Wextra -Wpedantic)
set(RELEASE_OPTIONS -O2 -march=native)
target_compile_options(eicos PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options(eicos PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")

target_link_libraries(eicos Eigen3::Eigen fmt::fmt)

add_executable(run src/run.cpp)
target_link_libraries(run eicos)

add_executable(test test/ecostester.cpp)
target_link_libraries(test eicos)
