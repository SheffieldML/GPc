# CIP-0020 Phase A: thin CMake for libgp + CLI smoke targets.
# Blessed native/CI path remains the makefile (make.linux / make.osx).
# See docs/build-and-ci.md and cip/cip0020.md.

cmake_minimum_required(VERSION 3.16)

# On macOS prefer Homebrew GCC so C++ and gfortran share an ABI (matches make.osx).
# Skip when the user already set CXX/FC or CMAKE_*_COMPILER on the command line.
option(GPC_PREFER_HOMEBREW_GCC "Prefer Homebrew g++/gfortran on Apple (CIP-0020)" ON)
if(APPLE AND GPC_PREFER_HOMEBREW_GCC AND NOT DEFINED ENV{CXX} AND NOT DEFINED ENV{FC})
  if(NOT CMAKE_CXX_COMPILER AND NOT CMAKE_Fortran_COMPILER)
    execute_process(
      COMMAND brew --prefix gcc
      OUTPUT_VARIABLE _GPC_GCC_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_GPC_GCC_PREFIX)
      # Use /bin/ls so shell aliases (ls -F) cannot append type suffixes.
      execute_process(
        COMMAND sh -c "/bin/ls '${_GPC_GCC_PREFIX}/bin'/g++-[0-9]* 2>/dev/null | sort -V | tail -1"
        OUTPUT_VARIABLE _GPC_GXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      execute_process(
        COMMAND sh -c "/bin/ls '${_GPC_GCC_PREFIX}/bin'/gfortran-[0-9]* 2>/dev/null | sort -V | tail -1"
        OUTPUT_VARIABLE _GPC_GFORTRAN
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(_GPC_GXX AND _GPC_GFORTRAN)
        set(CMAKE_CXX_COMPILER "${_GPC_GXX}" CACHE FILEPATH "C++ compiler" FORCE)
        set(CMAKE_Fortran_COMPILER "${_GPC_GFORTRAN}" CACHE FILEPATH "Fortran compiler" FORCE)
        set(CMAKE_C_COMPILER "${_GPC_GXX}" CACHE FILEPATH "C compiler" FORCE)
      endif()
    endif()
  endif()
endif()

project(GPc LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GPC_USE_MATIO "Enable libmatio numeric .mat I/O (_NDLMATIO)" OFF)
option(GPC_BUILD_EXTRA_CLIS "Also build ivm, gplvm, bgplvm" ON)

include(${CMAKE_SOURCE_DIR}/cmake/GpcSources.cmake)

if(APPLE)
  set(GPC_PLAT "_OSX")
else()
  set(GPC_PLAT "_LINUX")
endif()

# Debug matches bare make (checks on); Release matches make RELEASE=1.
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

# OpenBLAS (CIP-0013). On Apple, Homebrew openblas is preferred over Accelerate.
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(gp_fortran OBJECT ${GPC_LIBGP_FORTRAN_SOURCES})
# Match make FCFLAGS; -std=legacy quiets F2018 deleted-feature DO warnings in vendored L-BFGS/PSI.
target_compile_options(gp_fortran PRIVATE -std=legacy -funroll-all-loops -O3 -fPIC)

# Target name gpc_lib avoids clashing with the gp executable (CMP0002).
add_library(gpc_lib STATIC ${GPC_LIBGP_CXX_SOURCES} $<TARGET_OBJECTS:gp_fortran>)
set_target_properties(gpc_lib PROPERTIES OUTPUT_NAME gp)
target_include_directories(gpc_lib PUBLIC ${CMAKE_SOURCE_DIR})
if(GPC_USE_MATIO)
  target_include_directories(gpc_lib PUBLIC ${MATIO_INCLUDE_DIR})
endif()
target_link_libraries(gpc_lib PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
if(GPC_USE_MATIO)
  target_link_libraries(gpc_lib PUBLIC ${MATIO_LIBRARY})
endif()
# gfortran runtime (Fortran objects in the static archive need it at link time).
target_link_libraries(gpc_lib PUBLIC ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})

function(gpc_add_cli name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE gpc_lib)
  # Ensure rpath picks up Homebrew OpenBLAS / gfortran on macOS.
  if(APPLE)
    set_target_properties(${name} PROPERTIES BUILD_RPATH "/opt/homebrew/opt/openblas/lib;/opt/homebrew/opt/gcc/lib/gcc/current")
  endif()
endfunction()

gpc_add_cli(gp ${CMAKE_SOURCE_DIR}/gp.cpp)

if(GPC_BUILD_EXTRA_CLIS)
  gpc_add_cli(ivm ${CMAKE_SOURCE_DIR}/ivm.cpp ${GPC_IVM_EXTRA_SOURCES})
  gpc_add_cli(gplvm ${CMAKE_SOURCE_DIR}/gplvm.cpp ${GPC_GPLVM_EXTRA_SOURCES})
  gpc_add_cli(bgplvm ${CMAKE_SOURCE_DIR}/bgplvm.cpp ${GPC_BGPLVM_EXTRA_SOURCES})
endif()

message(STATUS "GPc CMake (CIP-0020): build=${CMAKE_BUILD_TYPE} plat=${GPC_PLAT} matio=${GPC_USE_MATIO}")
message(STATUS "  CXX=${CMAKE_CXX_COMPILER}")
message(STATUS "  Fortran=${CMAKE_Fortran_COMPILER}")
message(STATUS "  BLAS=${BLAS_LIBRARIES}")
