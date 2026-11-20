# CIP-0020 Phase C: optional CMake/CTest mirrors of makefile unit / matio-gate binaries.
# Does not replace make run-matio-tests as the CI hard gate.

include(CTest)
if(NOT BUILD_TESTING)
  return()
endif()

function(gpc_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE gpc_lib)
  if(APPLE)
    set_target_properties(${name} PROPERTIES
      BUILD_RPATH "/opt/homebrew/opt/openblas/lib;/opt/homebrew/opt/gcc/lib/gcc/current")
  endif()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY ${GPC_ROOT})
endfunction()

# Units that do not require libmatio at link time (still part of the make matio-tests gate).
gpc_add_test(testNdlutil ${GPC_ROOT}/testNdlutil.cpp)
gpc_add_test(testTransform ${GPC_ROOT}/testTransform.cpp)
gpc_add_test(testOptimisable ${GPC_ROOT}/testOptimisable.cpp)
gpc_add_test(testParamPaths ${GPC_ROOT}/testParamPaths.cpp)
gpc_add_test(testCholAdjoint ${GPC_ROOT}/testCholAdjoint.cpp)
gpc_add_test(testLQAdjoint ${GPC_ROOT}/testLQAdjoint.cpp)
gpc_add_test(testDist ${GPC_ROOT}/testDist.cpp)
gpc_add_test(testKern ${GPC_ROOT}/testKern.cpp)
gpc_add_test(testDiagGradParams ${GPC_ROOT}/testDiagGradParams.cpp)
gpc_add_test(testInducingGradZ ${GPC_ROOT}/testInducingGradZ.cpp)
gpc_add_test(testNoise ${GPC_ROOT}/testNoise.cpp)
gpc_add_test(testMatrix ${GPC_ROOT}/testMatrix.cpp)
gpc_add_test(testCoregionalize ${GPC_ROOT}/testCoregionalize.cpp)
gpc_add_test(testPsi ${GPC_ROOT}/testPsi.cpp)

add_library(gpc_mltools STATIC ${GPC_ROOT}/CMltools.cpp)
target_link_libraries(gpc_mltools PUBLIC gpc_lib)

gpc_add_test(testGp ${GPC_ROOT}/testGp.cpp)
target_link_libraries(testGp PRIVATE gpc_mltools)

gpc_add_test(testCGp ${GPC_ROOT}/testCGp.cpp)
target_link_libraries(testCGp PRIVATE gpc_mltools)

add_library(gpc_bayes_gplvm STATIC ${GPC_ROOT}/CBayesGplvm.cpp)
target_link_libraries(gpc_bayes_gplvm PUBLIC gpc_lib)

gpc_add_test(testBayesGplvm ${GPC_ROOT}/testBayesGplvm.cpp)
target_link_libraries(testBayesGplvm PRIVATE gpc_bayes_gplvm)

add_library(gpc_mrd STATIC ${GPC_ROOT}/CMrd.cpp)
target_link_libraries(gpc_mrd PUBLIC gpc_bayes_gplvm)

gpc_add_test(testMrd ${GPC_ROOT}/testMrd.cpp)
target_link_libraries(testMrd PRIVATE gpc_mrd)

# Fixture-backed suites need _NDLMATIO for numeric .mat loads (same as make USE_MATIO=1).
if(NOT GPC_USE_MATIO)
  set_tests_properties(
    testDist testKern testNoise testMatrix testGp testCGp testPsi testBayesGplvm testMrd
    PROPERTIES DISABLED TRUE)
endif()
