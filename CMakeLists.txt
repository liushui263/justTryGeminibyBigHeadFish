cmake_minimum_required(VERSION 3.15)
project(EM3D CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages (use modern CUDAToolkit instead of CUDA)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json 3.2.0 QUIET)

# If nlohmann_json not found, try to use header-only version
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package, using header-only")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/config_loader.cpp
    src/assembler.cpp
    src/solver.cpp
    src/exporter.cpp
    src/analytical.cpp
)

# Create executable
add_executable(em3d_solver ${SOURCES})

target_include_directories(em3d_solver PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
# Link libraries using modern CUDAToolkit
#target_link_libraries(em3d_solver 
#    PUBLIC OpenMP::OpenMP_CXX
#    PUBLIC CUDA::cudart
#    PUBLIC CUDA::cusolver
#    PUBLIC CUDA::cusparse
#)
# Critical: Link the CUDA math libraries
#target_link_libraries(em3d_solver 
#    PRIVATE 
#    nlohmann_json::nlohmann_json
#    OpenMP::OpenMP_CXX
#    ${CUDA_LIBRARIES}
#    cublas
#    cusparse
#)
# Link libraries using modern CUDAToolkit targets
target_link_libraries(em3d_solver 
    PUBLIC 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cublas      # <--- Use the namespaced version
    nlohmann_json::nlohmann_json
)
# Optional: Link nlohmann_json if found
if(nlohmann_json_FOUND)
    target_link_libraries(em3d_solver PUBLIC nlohmann_json::nlohmann_json)
endif()

# Compiler flags
if(MSVC)
    target_compile_options(em3d_solver PRIVATE /W4)
else()
    target_compile_options(em3d_solver PRIVATE -Wall -Wextra -O3)
endif()

# Set output directory
set_target_properties(em3d_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)