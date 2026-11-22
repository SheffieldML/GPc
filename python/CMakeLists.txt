# CIP-0020 Phase B / CIP-001D: pybind11 module gpc._gpc linked against gpc_lib.
# Built when GPC_BUILD_PYTHON=ON or when scikit-build-core configures the repo root.

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(GPC_PY_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/module.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_numpy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/wrap_gp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/wrap_gplvm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/wrap_deepgp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/wrap_likelihood.cpp
)

pybind11_add_module(_gpc MODULE ${GPC_PY_SOURCES})
target_include_directories(_gpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${GPC_ROOT})
target_link_libraries(_gpc PRIVATE gpc_lib)

if(APPLE)
  if(DEFINED SKBUILD)
    # CIP-002C Phase B: wheel builds — relative RPATH only; delocate vendors dylibs.
    set_target_properties(_gpc PROPERTIES
      BUILD_RPATH "@loader_path"
      INSTALL_RPATH "@loader_path")
  else()
    # Contributor editable / plain CMake: Homebrew absolute RPATH for local runs.
    set_target_properties(_gpc PROPERTIES
      BUILD_RPATH "/opt/homebrew/opt/openblas/lib;/opt/homebrew/opt/gcc/lib/gcc/current"
      INSTALL_RPATH "@loader_path;/opt/homebrew/opt/openblas/lib;/opt/homebrew/opt/gcc/lib/gcc/current")
  endif()
endif()

# scikit-build-core installs into the wheel; plain CMake installs beside the build tree.
install(TARGETS _gpc DESTINATION gpc)
