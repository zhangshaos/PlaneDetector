cmake_minimum_required(VERSION 3.12)
project(PlaneDetector)

if (MSVC)
    add_compile_options(/MP /utf-8 /bigobj)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(PlaneDetector STATIC
        src/PlaneDetector.cpp)
target_include_directories(PlaneDetector PUBLIC include)
target_link_libraries(PlaneDetector)

add_executable(test example/test.cpp)
target_link_libraries(test PRIVATE PlaneDetector)