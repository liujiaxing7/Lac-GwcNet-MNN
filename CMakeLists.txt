cmake_minimum_required(VERSION 3.20)
project(Lac_GwcNet)

set(CMAKE_CXX_STANDARD 14)

#MNN
set(MNN_DIR /home/ljx/Downloads/MNN/MNN-2.2.0)
include_directories(${MNN_DIR}/include)
include_directories(${MNN_DIR}/include/MNN)
include_directories(${MNN_DIR}/3rd_party/imageHelper/)

#opencv
find_package(OpenCV REQUIRED)

add_executable(Lac_GwcNet main2.2.0.cpp utils.cpp utils.h)
target_link_libraries(Lac_GwcNet ${MNN_DIR}/build/libMNN.so)
target_link_libraries(Lac_GwcNet ${OpenCV_LIBS} )