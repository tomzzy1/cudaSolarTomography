# https://github.com/cschreib/vif/blob/master/cmake/FindWCSLib.cmake

# - Try to find WCSLIB: the FITS "World Coordinate System" library
# Variables used by this module:
#  WCSLIB_ROOT_DIR     - WCSLIB root directory
# Variables defined by this module:
#  WCSLIB_FOUND        - system has WCSLIB
#  WCSLIB_INCLUDE_DIR  - the WCSLIB include directory (cached)
#  WCSLIB_INCLUDE_DIRS - the WCSLIB include directories
#                        (identical to WCSLIB_INCLUDE_DIR)
#  WCSLIB_LIBRARY      - the WCSLIB library (cached)
#  WCSLIB_LIBRARIES    - the WCSLIB libraries
#                        (identical to WCSLIB_LIBRARY)

# Copyright (C) 2009
# ASTRON (Netherlands Institute for Radio Astronomy)
# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands
#
# This file is part of the LOFAR software suite.
# The LOFAR software suite is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The LOFAR software suite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
#
# $Id$

if(NOT WCSLIB_FOUND)

  find_path(WCSLIB_INCLUDE_DIR wcslib/wcs.h
    HINTS ${WCSLIB_ROOT_DIR} PATH_SUFFIXES include)
  find_library(WCSLIB_LIBRARY wcs
    HINTS ${WCSLIB_ROOT_DIR} PATH_SUFFIXES lib)
  find_library(M_LIBRARY m)

  if(EXISTS ${WCSLIB_INCLUDE_DIR})
    file(STRINGS "${WCSLIB_INCLUDE_DIR}/wcslib/wcsconfig.h" TLINE
           REGEX "^#[\t ]*define[\t ]+WCSLIB_VERSION[\t ]+[0-9.]+$")

    string(REGEX REPLACE "^#[\t ]*define[\t ]+WCSLIB_VERSION[\t ]+([0-9.]+)$" "\\1"
           WCSLIB_VERSION_STRING "${TLINE}")

    if(NOT "${WCSLIB_VERSION_STRING}" STREQUAL "")
      message(STATUS "Found WCSLIB version ${WCSLIB_VERSION_STRING}")
    endif()
  endif()

  mark_as_advanced(WCSLIB_INCLUDE_DIR WCSLIB_LIBRARY WCSLIB_VERSION_STRING M_LIBRARY)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(WCSLIB DEFAULT_MSG
    WCSLIB_LIBRARY M_LIBRARY WCSLIB_INCLUDE_DIR WCSLIB_VERSION_STRING)

  set(WCSLIB_INCLUDE_DIRS ${WCSLIB_INCLUDE_DIR})
  set(WCSLIB_LIBRARIES ${WCSLIB_LIBRARY} ${M_LIBRARY})

  include_directories(${WCSLIB_INCLUDE_DIR})
  link_libraries(${WCSLIB_LIBRARIES})
endif(NOT WCSLIB_FOUND)
