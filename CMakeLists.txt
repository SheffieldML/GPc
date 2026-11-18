# CIP-0020: thin CMake for libgp + CLI smoke (+ optional Python via scikit-build).
# Blessed native/CI path remains the makefile (make.linux / make.osx).
# See docs/build-and-ci.md and cip/cip0020.md.

cmake_minimum_required(VERSION 3.16)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PreferHomebrewGcc.cmake)

project(GPc LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GPC_USE_MATIO "Enable libmatio numeric .mat I/O (_NDLMATIO)" OFF)
option(GPC_BUILD_EXTRA_CLIS "Also build ivm, gplvm, bgplvm, ppa" ON)
option(GPC_BUILD_GP_CLI "Build the gp executable" ON)
option(GPC_BUILD_PYTHON "Build the pybind11 gpc._gpc extension" OFF)
option(GPC_BUILD_TESTS "Build CTest mirrors of makefile unit/matio-gate binaries" OFF)

# scikit-build-core sets SKBUILD when configuring for a wheel / editable install.
if(DEFINED SKBUILD)
  set(GPC_BUILD_PYTHON ON)
  set(GPC_BUILD_GP_CLI OFF)
  set(GPC_BUILD_EXTRA_CLIS OFF)
  set(GPC_BUILD_TESTS OFF)
endif()

set(GPC_ROOT ${CMAKE_SOURCE_DIR})

if(APPLE)
  set(GPC_PLAT "_OSX")
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/openblas /usr/local/opt/openblas)
else()
  set(GPC_PLAT "_LINUX")
endif()

add_compile_definitions(${GPC_PLAT})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_definitions(NDEBUG GPC_CHECKS=0)
else()
  add_compile_definitions(DBG GPC_CHECKS=1)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
  endif()
endif()

if(GPC_USE_MATIO)
  add_compile_definitions(_NDLMATIO)
  find_path(MATIO_INCLUDE_DIR matio.h
    HINTS $ENV{MATIOROOT}/include /opt/homebrew/opt/libmatio/include /usr/local/opt/libmatio/include)
  find_library(MATIO_LIBRARY matio
    HINTS $ENV{MATIOROOT}/lib /opt/homebrew/opt/libmatio/lib /usr/local/opt/libmatio/lib)
  if(NOT MATIO_INCLUDE_DIR OR NOT MATIO_LIBRARY)
    message(FATAL_ERROR "GPC_USE_MATIO=ON but libmatio was not found")
  endif()
endif()

set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include(${GPC_ROOT}/cmake/GpcLib.cmake)

function(gpc_add_cli name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE gpc_lib)
  if(APPLE)
    set_target_properties(${name} PROPERTIES
      BUILD_RPATH "/opt/homebrew/opt/openblas/lib;/opt/homebrew/opt/gcc/lib/gcc/current")
  endif()
endfunction()

if(GPC_BUILD_GP_CLI)
  gpc_add_cli(gp ${GPC_ROOT}/gp.cpp)
endif()

if(GPC_BUILD_EXTRA_CLIS)
  gpc_add_cli(ivm ${GPC_ROOT}/ivm.cpp ${GPC_IVM_EXTRA_SOURCES})
  gpc_add_cli(gplvm ${GPC_ROOT}/gplvm.cpp ${GPC_GPLVM_EXTRA_SOURCES})
  gpc_add_cli(bgplvm ${GPC_ROOT}/bgplvm.cpp ${GPC_BGPLVM_EXTRA_SOURCES})
  gpc_add_cli(ppa ${GPC_ROOT}/ppa.cpp ${GPC_PPA_EXTRA_SOURCES})
endif()

if(GPC_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(GPC_BUILD_TESTS)
  include(${GPC_ROOT}/cmake/GpcTests.cmake)
endif()

message(STATUS "GPc CMake (CIP-0020): build=${CMAKE_BUILD_TYPE} plat=${GPC_PLAT} matio=${GPC_USE_MATIO}")
message(STATUS "  CXX=${CMAKE_CXX_COMPILER}")
message(STATUS "  Fortran=${CMAKE_Fortran_COMPILER}")
message(STATUS "  BLAS=${BLAS_LIBRARIES}")
message(STATUS "  python=${GPC_BUILD_PYTHON} gp_cli=${GPC_BUILD_GP_CLI} tests=${GPC_BUILD_TESTS}")
