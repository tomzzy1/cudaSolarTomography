cmake_minimum_required(VERSION 3.20)
include(CheckLanguage)

project(build_A_matrix VERSION 0.1 LANGUAGES C)

set(CMAKE_CFLAGS "-Wall -O3 -g -fno-exceptions")


add_library(libwcs STATIC IMPORTED)
set_target_properties(libwcs PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libwcs/libwcs.a)


add_executable(test_get_orbit test_get_orbit.c get_orbit.c r3misc.c)
target_link_libraries(test_get_orbit libwcs)
