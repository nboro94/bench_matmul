cmake_minimum_required(VERSION 3.15)
project(bench-matmul LANGUAGES CXX C)

# -- Basic configuration ---------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_TBB "Enable TBB-backed matrix multiplication if available" ON)
option(FETCH_TBB "Fetch oneTBB via FetchContent when TBB not found" OFF)
option(ENABLE_WARNINGS "Enable reasonable compiler warnings" ON)
option(USE_CUDA "Enable CUDA-backed matrix multiplication if available" OFF)
option(USE_DEBUG_MODE "Enable additional debug trace logs" OFF)
set(SIMD_ISA "AVX2" CACHE STRING "SIMD ISA level (AVX2 or AVX512)")
set_property(CACHE SIMD_ISA PROPERTY STRINGS AVX2 AVX512)
string(TOUPPER "${SIMD_ISA}" SIMD_ISA_UPPER)
if(NOT SIMD_ISA_UPPER STREQUAL "AVX2" AND NOT SIMD_ISA_UPPER STREQUAL "AVX512")
    message(FATAL_ERROR "SIMD_ISA must be AVX2 or AVX512 (got: ${SIMD_ISA})")
endif()

# Compiler flags (non-invasive, controlled per-compiler)
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:/W4>
        $<$<COMPILE_LANGUAGE:C>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    if(ENABLE_WARNINGS)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:C>:-Wall>
            $<$<COMPILE_LANGUAGE:C>:-Wextra>
            $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        )
    endif()
endif()

# -- Sources ---------------------------------------------------------------
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cc
    ${PROJECT_SOURCE_DIR}/src/utils.cc
    ${PROJECT_SOURCE_DIR}/src/benchmark.cc
    ${PROJECT_SOURCE_DIR}/src/matmul/naive.cc
    ${PROJECT_SOURCE_DIR}/src/matmul/blocked.cc
    ${PROJECT_SOURCE_DIR}/src/matmul/transposed.cc
    ${PROJECT_SOURCE_DIR}/src/matmul/simd.cc
    ${PROJECT_SOURCE_DIR}/src/matmul/unrolled.cc
    ${PROJECT_SOURCE_DIR}/src/matmul/pthread.cc
)

# Optional TBB-backed implementation
if(USE_TBB)
    find_package(TBB CONFIG QUIET)
    if(NOT TBB_FOUND AND FETCH_TBB)
        include(FetchContent)
        FetchContent_Declare(
            tbb
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG        v2023.0.0
        )
        FetchContent_MakeAvailable(tbb)
    endif()

    if(TBB_FOUND OR TARGET TBB::tbb)
        list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/src/matmul/tbb.cc)
        add_compile_definitions(HAVE_TBB)
        set(TBB_AVAILABLE TRUE)
        message(STATUS "TBB enabled for matrix multiplication")
    else()
        message(STATUS "TBB not available; falling back to std::thread implementation")
    endif()
endif()

# Optional CUDA-backed implementation
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/src/matmul/cuda.cu)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(HAVE_CUDA)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA enabled for matrix multiplication")
endif()

# -- Target ----------------------------------------------------------------
add_executable(bench-matmul ${SOURCES})

if(MSVC)
    target_compile_options(bench-matmul PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/arch:${SIMD_ISA_UPPER}>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    if(SIMD_ISA_UPPER STREQUAL "AVX512")
        target_compile_options(bench-matmul PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512f>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512dq>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512vl>
            $<$<COMPILE_LANGUAGE:CXX>:-mfma>
        )
    else()
        target_compile_options(bench-matmul PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
            $<$<COMPILE_LANGUAGE:CXX>:-mfma>
        )
    endif()
endif()

# Link Threads (pthread on POSIX, handled automatically on MSVC)
find_package(Threads REQUIRED)
target_link_libraries(bench-matmul PRIVATE Threads::Threads)

if(CMAKE_USE_PTHREADS_INIT)
    target_compile_definitions(bench-matmul PRIVATE HAVE_PTHREAD)
endif()

# Link TBB when available
if(TBB_AVAILABLE)
    if(TARGET TBB::tbb_static)
        target_link_libraries(bench-matmul PRIVATE TBB::tbb_static)
    elseif(TARGET TBB::tbb)
        target_link_libraries(bench-matmul PRIVATE TBB::tbb)
    else()
        target_link_libraries(bench-matmul PRIVATE tbb)
    endif()
endif()

# Link CUDA when available
if(CUDA_AVAILABLE)
    target_link_libraries(bench-matmul PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

# Include directories (project-local headers)
target_include_directories(bench-matmul PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Helpful compile-time options
target_compile_definitions(bench-matmul PRIVATE
    $<$<BOOL:${TBB_AVAILABLE}>:HAVE_TBB>
    $<$<BOOL:${CUDA_AVAILABLE}>:HAVE_CUDA>
    USE_DEBUG_MODE=$<IF:$<BOOL:${USE_DEBUG_MODE}>,1,0>
)

# Installation and packaging (small conveniences)
install(TARGETS bench-matmul RUNTIME DESTINATION bin)

# -- Summary ---------------------------------------------------------------
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "TBB support: ${TBB_AVAILABLE}")
message(STATUS "CUDA support: ${CUDA_AVAILABLE}")
message(STATUS "Debug mode: ${USE_DEBUG_MODE}")
if(MSVC)
    message(STATUS "SIMD ISA: ${SIMD_ISA_UPPER}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    message(STATUS "SIMD ISA: ${SIMD_ISA_UPPER}")
endif()
