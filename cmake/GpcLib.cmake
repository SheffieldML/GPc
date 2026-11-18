# Create static target gpc_lib (OUTPUT_NAME gp) from GpcSources.cmake.
# Caller must: set GPC_ROOT, enable CXX+Fortran, find BLAS/LAPACK, set plat/check defs.

if(NOT DEFINED GPC_ROOT)
  message(FATAL_ERROR "GpcLib.cmake requires GPC_ROOT (repository root)")
endif()

include(${GPC_ROOT}/cmake/GpcSources.cmake)

add_library(gp_fortran OBJECT ${GPC_LIBGP_FORTRAN_SOURCES})
target_compile_options(gp_fortran PRIVATE -std=legacy -funroll-all-loops -O3 -fPIC)

add_library(gpc_lib STATIC ${GPC_LIBGP_CXX_SOURCES} $<TARGET_OBJECTS:gp_fortran>)
set_target_properties(gpc_lib PROPERTIES OUTPUT_NAME gp)
target_include_directories(gpc_lib PUBLIC ${GPC_ROOT})
if(GPC_USE_MATIO)
  target_include_directories(gpc_lib PUBLIC ${MATIO_INCLUDE_DIR})
  target_link_libraries(gpc_lib PUBLIC ${MATIO_LIBRARY})
endif()
target_link_libraries(gpc_lib PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(gpc_lib PUBLIC ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
if(GPC_OPENMP)
  target_link_libraries(gpc_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
