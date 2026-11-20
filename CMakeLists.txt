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
# CIP-0025: optional OpenMP over bulk K(r) workspace fills.
# cmake -DGPC_OPENMP=ON ...  or  cmake -DGPC_OPENMP=1 ...
# Requires libomp / libgomp; set OPENBLAS_NUM_THREADS=1 when enabled.
option(GPC_OPENMP "Enable OpenMP for parallel stationary workspace fills" OFF)

# scikit-build-core sets SKBUILD when configuring for a wheel / editable install.
if(DEFINED SKBUILD)
  set(GPC_BUILD_PYTHON ON)
  set(GPC_BUILD_GP_CLI OFF)
  set(GPC_BUILD_EXTRA_CLIS OFF)
  set(GPC_BUILD_TESTS OFF)
endif()

set(GPC_ROOT ${CMAKE_SOURCE_DIR})

# CIP-002B: distinguish Windows from Linux (do not fall through to _LINUX).
if(APPLE)
  set(GPC_PLAT "_OSX")
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/openblas /usr/local/opt/openblas)
elseif(WIN32)
  set(GPC_PLAT "_WIN32")
else()
  set(GPC_PLAT "_LINUX")
endif()

add_compile_definitions(${GPC_PLAT})
# CIP-002D QW1: default RelWithDebInfo (-O2 -g) with cheap checks; Release = RELEASE=1.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
# Optional override: -DGPC_CHECKS=0|1|2 (2 = makefile CHECKS=full). Empty = from build type.
set(GPC_CHECKS "" CACHE STRING "Check level override: 0 (off), 1 (cheap), 2 (full); empty=from CONFIG")
# Generator expressions cover Ninja (CMAKE_BUILD_TYPE) and multi-config VS (CIP-002B).
add_compile_definitions(
  $<$<CONFIG:Release>:NDEBUG>
  $<$<NOT:$<CONFIG:Release>>:DBG>
)
if(GPC_CHECKS STREQUAL "")
  add_compile_definitions(
    $<$<CONFIG:Release>:GPC_CHECKS=0>
    $<$<NOT:$<CONFIG:Release>>:GPC_CHECKS=1>
  )
else()
  add_compile_definitions(GPC_CHECKS=${GPC_CHECKS})
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

if(GPC_OPENMP)
  find_package(OpenMP REQUIRED)
  add_compile_definitions(GPC_OPENMP=1)
endif()

# Unix: FindBLAS/FindLAPACK with BLA_VENDOR=OpenBLAS (makefile parity).
# Windows (CIP-002B): vcpkg openblas is NOFORTRAN + BUILD_WITHOUT_LAPACK, and
# FindBLAS's Fortran sgemm probe fails under MSVC+ifx. Prefer OpenBLAS CONFIG
# for BLAS; pair with a separate LAPACK (lapack-reference) from vcpkg.
set(GPC_BLAS_LINK_LIBS "")
if(WIN32)
  find_package(OpenBLAS CONFIG QUIET)
  if(OpenBLAS_FOUND)
    list(APPEND GPC_BLAS_LINK_LIBS OpenBLAS::OpenBLAS)
    message(STATUS "OpenBLAS via CONFIG (Windows/vcpkg)")
  endif()
  # Netlib LAPACK (Fortran ABI with trailing underscores, matching lapack.h).
  find_library(GPC_LAPACK_LIB NAMES lapack liblapack)
  if(GPC_LAPACK_LIB)
    list(APPEND GPC_BLAS_LINK_LIBS ${GPC_LAPACK_LIB})
    message(STATUS "LAPACK library: ${GPC_LAPACK_LIB}")
  elseif(OpenBLAS_FOUND)
    # Some OpenBLAS builds embed LAPACK; still try FindLAPACK as a fallback.
    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
      list(APPEND GPC_BLAS_LINK_LIBS ${LAPACK_LIBRARIES})
    else()
      message(WARNING "OpenBLAS found but no separate LAPACK; link may fail on dpotrf_/dsyev_")
    endif()
  endif()
endif()
if(NOT GPC_BLAS_LINK_LIBS)
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  list(APPEND GPC_BLAS_LINK_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()
if(WIN32 AND OpenBLAS_FOUND AND NOT GPC_LAPACK_LIB AND NOT LAPACK_FOUND)
  message(FATAL_ERROR
    "Windows build needs LAPACK alongside vcpkg OpenBLAS "
    "(e.g. vcpkg install lapack-reference:x64-windows).")
endif()

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
message(STATUS "  BLAS_LINK=${GPC_BLAS_LINK_LIBS}")
message(STATUS "  python=${GPC_BUILD_PYTHON} gp_cli=${GPC_BUILD_GP_CLI} tests=${GPC_BUILD_TESTS}")
