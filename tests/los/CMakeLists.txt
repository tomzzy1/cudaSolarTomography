cmake_minimum_required(VERSION 3.20)
include(FetchContent)

project(los VERSION 1.0 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall -O3 -g")
# set(CMAKE_CXX_FLAGS "-Wall -g")

include_directories("./include")
include_directories("./GeometricTools/GTE")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(WCSLIB)
find_package(CFITSIO)
find_package(MATIO)

# FetchContent_Declare(
#   argparse
#   GIT_REPOSITORY https://github.com/p-ranav/argparse.git
# )
# FetchContent_MakeAvailable(argparse)


add_executable(los los.cpp coordinates.cpp HollowSphere.cpp scatter.cpp)
# target_link_libraries(los argparse)

add_executable(fts2mat fts2mat.cpp FITS_Image.cpp coordinates.cpp HollowSphere.cpp scatter.cpp MAT_File.cpp Config.cpp)
target_link_libraries(fts2mat MATIO::MATIO)
