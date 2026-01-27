cmake_minimum_required(VERSION 3.10)
project(LWD_Solver CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

# 查找 CUDA 包
find_package(CUDA REQUIRED)

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 自动查找 src 下的所有 .cpp 文件

file(GLOB SOURCES "src/*.cpp")
add_executable(lwd_solver ${SOURCES})
target_link_libraries(lwd_solver ${CUDA_LIBRARIES} cusparse cusolver)