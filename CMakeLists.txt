cmake_minimum_required(VERSION 3.20)
include(CheckLanguage)

project(build_A_matrix VERSION 1.0.0 LANGUAGES CXX C)

#set(CMAKE_CXX_COMPILER "g++-11")
#set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -O3 -g -fno-exceptions")

add_library(libwcs STATIC IMPORTED)
set_target_properties(libwcs PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libwcs/libwcs.a)

#add_library(lbfgsb STATIC IMPORTED)
#set_target_properties(lbfgsb PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lbfgsb-gpu/build/culbfgsb/libcuLBFGSB.a)

add_executable(build_A_matrix build_A_matrix.cpp)
target_link_libraries(build_A_matrix libwcs)

option(BUILD_SOLVER "Build the Solver" ON)

if (BUILD_SOLVER)
  add_library(lbfgs SHARED IMPORTED)
  set_target_properties(lbfgs PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/solver/build/liblbfgs.so)
  add_compile_definitions(BUILD_SOLVER=1)
  message(STATUS "solver enabled")
else()
  add_compile_definitions(BUILD_SOLVER=0)
  message(STATUS "solver disabled")
endif()

find_package(pybind11)

include_directories(include)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)

  if (CUDA_TOOLKIT_VERSION EQUAL 11)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "-std=c++17 --expt-relaxed-constexpr -O3")
  elseif (CUDA_TOOLKIT_VERSION EQUAL 12)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS "-std=c++20 --expt-relaxed-constexpr -O3")
  else()
    message(FATAL_ERROR "CUDA Toolkit Version under 11 not supported")
  endif()

  add_compile_definitions(CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION})
  
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 70 80)

  if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
    set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
  endif()

  if(NOT DEFINED CMAKE_CUDA17_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "")
    set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "")
  endif()

  include_directories("${CMAKE_SOURCE_DIR}/libwcs"
    ${CUDA_INCLUDE_DIRS}
    #"${CMAKE_SOURCE_DIR}/lbfgsb-gpu/culbfgsb"
    "${CMAKE_SOURCE_DIR}/solver/include/CudaLBFGS")

  if (false)

  add_executable(gpu_build_A_matrix build_A_matrix.cu)

  if (CMAKE_CUDA_STANDARD EQUAL 20)
  add_executable(old_gpu_build_A_matrix build_A_matrix_old.cu)
  endif()

  set_target_properties(gpu_build_A_matrix PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(gpu_build_A_matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(gpu_build_A_matrix libwcs ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  if (CMAKE_CUDA_STANDARD EQUAL 20)
    set_target_properties(old_gpu_build_A_matrix PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(old_gpu_build_A_matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(old_gpu_build_A_matrix libwcs ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})
  endif()
  
  endif() # disable (old_)gpu_build_A_matrix

  add_executable(reconstruct reconstruct.cu)

  set_target_properties(reconstruct PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(reconstruct PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #target_link_libraries(reconstruct libwcs lbfgsb lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})
  target_link_libraries(reconstruct libwcs lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  add_executable(cross_validation cross_validation.cu)

  set_target_properties(cross_validation PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(cross_validation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #target_link_libraries(cross_validation libwcs lbfgsb lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})
  target_link_libraries(cross_validation libwcs lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  add_executable(sim_validation sim_validation.cu)
  set_target_properties(sim_validation PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(sim_validation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(sim_validation libwcs lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  add_executable(projection projection.cu)
  set_target_properties(projection PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(projection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(projection libwcs lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  add_executable(projection_validation projection_validation.cu)
  set_target_properties(projection_validation PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(projection_validation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(projection_validation libwcs lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  add_executable(projection_reconstruct projection_reconstruct.cu)
  set_target_properties(projection_reconstruct PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(projection_reconstruct PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(projection_reconstruct libwcs lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  add_executable(reconstruct_template reconstruct_template.cu)
  set_target_properties(reconstruct_template PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(reconstruct_template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(reconstruct_template libwcs lbfgs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})

  if (pybind11_FOUND)
    message(STATUS "enable Pybind11 build")
    pybind11_add_module(py_cuda_solartomography wrapper.cu)
    set_target_properties(py_cuda_solartomography PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(py_cuda_solartomography PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(py_cuda_solartomography PRIVATE libwcs cublas cusparse ${CUDA_LIBRARIES} ${CUDA_cudaevrt_LIBRARY})
    if (BUILD_SOLVER)
      target_link_libraries(py_cuda_solartomography PRIVATE lbfgs)
    endif()
  else()
    message(STATUS "Pybind11 not find, skip Python module build")
  endif()
else()
  include_directories("${CMAKE_SOURCE_DIR}/libwcs")
endif()