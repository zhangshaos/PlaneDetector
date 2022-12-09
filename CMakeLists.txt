cmake_minimum_required(VERSION 3.12)
project(PlaneDetector)

add_subdirectory(vendor/ED_Lib)
add_subdirectory(vendor/ELSED)

# 使用两个线段检测库
#add_compile_definitions(USE_EDLib)
add_compile_definitions(USE_ELSED)#实际测试，ELSED稍微好一丢丢
if (MSVC)
    add_compile_options(/MP /utf-8 /bigobj)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(glog CONFIG REQUIRED)

add_library(PlaneDetector STATIC
        src/Detector.cpp
        src/Tools.cpp
        src/DetectLine.cpp)
target_include_directories(PlaneDetector PUBLIC include)
target_link_libraries(PlaneDetector
        PUBLIC
        ${OpenCV_LIBS}
        glog::glog
        PRIVATE
        EDLib
        ELSED)

add_executable(DetectPlaneTest example/Test.cpp)
target_link_libraries(DetectPlaneTest PRIVATE PlaneDetector)