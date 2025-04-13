project(bench-matmul)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -mavx2 -O2")
set(SOURCES main.cpp)
add_executable(bemch-matmul ${SOURCES})

target_link_libraries(bench-matmul atomic pthread)
