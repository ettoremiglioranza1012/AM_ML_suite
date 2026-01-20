cmake_minimum_required(VERSION 3.20)

# =============================================================================
# AM C++ HPC Engine - Data Factory for Topology Optimization
# =============================================================================
# Purpose: High-performance solver for massive dataset generation
# Uses: C++20, Eigen, OpenMP
# =============================================================================

project(am_cpp_engine
    VERSION 0.1.0
    DESCRIPTION "HPC Topology Optimization Solver for AM"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Build Options
# -----------------------------------------------------------------------------
option(AM_USE_MPI "Enable MPI for distributed computing" OFF)
option(AM_USE_OPENMP "Enable OpenMP for shared-memory parallelism" ON)
option(AM_BUILD_TESTS "Build unit tests" ON)

# -----------------------------------------------------------------------------
# C++ Standard
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable optimizations for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# Eigen (header-only linear algebra) - Required
find_package(Eigen3 3.4...5.1 REQUIRED)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

# OpenMP (optional but recommended)
# Fix per macOS/Homebrew - AppleClang needs help finding libomp
if(AM_USE_OPENMP)
    if(APPLE)
        if(EXISTS "/opt/homebrew/opt/libomp")
            # Apple Silicon Macs
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
        elseif(EXISTS "/usr/local/opt/libomp")
            # Intel Macs
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
        endif()
    endif()

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found! Parallelism enabled.")
    else()
        message(WARNING "OpenMP not found, running in serial mode")
        set(AM_USE_OPENMP OFF)
    endif()
endif()

# MPI (optional, for future distributed computing)
if(AM_USE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(AM_USE_MPI)
    message(STATUS "MPI found")
endif()

# -----------------------------------------------------------------------------
# Library Target
# -----------------------------------------------------------------------------

add_library(am_engine STATIC
    src/element_kernel.cpp
    src/global_assembler.cpp
    src/linear_solver.cpp
    src/topology_optimizer.cpp
)

target_include_directories(am_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(am_engine PUBLIC Eigen3::Eigen)

if(AM_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(am_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

if(AM_USE_MPI)
    target_link_libraries(am_engine PUBLIC MPI::MPI_CXX)
endif()

# -----------------------------------------------------------------------------
# Test Executable
# -----------------------------------------------------------------------------

add_executable(am_element_test
    src/main.cpp
)

target_link_libraries(am_element_test PRIVATE am_engine)

# -----------------------------------------------------------------------------
# Standalone Topology Optimization Executable
# -----------------------------------------------------------------------------

add_executable(am_topopt
    src/am_topopt.cpp
)

target_link_libraries(am_topopt PRIVATE am_engine)

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

install(TARGETS am_engine
    EXPORT am_engine-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(TARGETS am_element_test am_topopt
    RUNTIME DESTINATION bin
)

# -----------------------------------------------------------------------------
# Status Messages
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "AM C++ Engine Configuration:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP:       ${AM_USE_OPENMP}")
message(STATUS "  MPI:          ${AM_USE_MPI}")
message(STATUS "  Compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
