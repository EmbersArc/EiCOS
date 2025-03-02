cmake_minimum_required(VERSION 3.5)
project(eicos)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt QUIET)
find_package(cuDSS 0.5.0 REQUIRED)

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

target_link_libraries(eicos Eigen3::Eigen)
target_link_libraries(eicos cuDSS::cuDSS)

IF (${fmt_FOUND})
   MESSAGE(STATUS "Found fmt.")
   target_link_libraries(eicos fmt::fmt)
ELSE (${fmt_FOUND})
   MESSAGE(STATUS "Could not locate fmt. Printing has been disabled.")
ENDIF (${fmt_FOUND})

add_executable(eicos_test_problem src/run.cpp)
target_link_libraries(eicos_test_problem eicos)

add_executable(eicos_run_tests test/ecostester.cpp)
target_link_libraries(eicos_run_tests eicos)
