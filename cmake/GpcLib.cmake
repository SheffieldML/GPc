# Create static target gpc_lib (OUTPUT_NAME gp) from GpcSources.cmake.
# Caller must: set GPC_ROOT, enable CXX+Fortran, find BLAS/LAPACK, set plat/check defs.

if(NOT DEFINED GPC_ROOT)
  message(FATAL_ERROR "GpcLib.cmake requires GPC_ROOT (repository root)")
endif()

include(${GPC_ROOT}/cmake/GpcSources.cmake)

# STATIC (not OBJECT): MSVC does not reliably re-export $<TARGET_OBJECTS:…>
# from a C++ static archive when the objects were compiled by ifx (CIP-002B).
add_library(gp_fortran STATIC ${GPC_LIBGP_FORTRAN_SOURCES})
set_target_properties(gp_fortran PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME gp_fortran
)
# GNU gfortran flags from the makefile path.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(gp_fortran PRIVATE -std=legacy -funroll-all-loops -O3)
  if(UNIX)
    target_compile_options(gp_fortran PRIVATE -fPIC)
  endif()
endif()
# Intel ifx / classic: match ndlfortran.h extern "C" lowercase + trailing underscore
# (default Windows ifx decoration otherwise leaves lbfgs_/psi_/… unresolved).
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  if(WIN32)
    target_compile_options(gp_fortran PRIVATE /names:lowercase /assume:underscore)
  else()
    target_compile_options(gp_fortran PRIVATE -names lowercase -assume underscore)
  endif()
endif()

add_library(gpc_lib STATIC ${GPC_LIBGP_CXX_SOURCES})
set_target_properties(gpc_lib PROPERTIES OUTPUT_NAME gp)
target_include_directories(gpc_lib PUBLIC ${GPC_ROOT})
if(GPC_USE_MATIO)
  target_include_directories(gpc_lib PUBLIC ${MATIO_INCLUDE_DIR})
  target_link_libraries(gpc_lib PUBLIC ${MATIO_LIBRARY})
endif()
if(NOT GPC_BLAS_LINK_LIBS)
  message(FATAL_ERROR "GpcLib.cmake requires GPC_BLAS_LINK_LIBS (set by root CMakeLists)")
endif()
target_link_libraries(gpc_lib PUBLIC gp_fortran ${GPC_BLAS_LINK_LIBS})
target_link_libraries(gpc_lib PUBLIC ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
if(GPC_OPENMP)
  target_link_libraries(gpc_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
