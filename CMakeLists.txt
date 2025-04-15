cmake_minimum_required(VERSION 3.10)
project(bench-matmul)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -mavx2 -march=native -O2")
set(SOURCES main.cpp)
add_executable(bench-matmul ${SOURCES})

target_link_libraries(bench-matmul atomic pthread)
