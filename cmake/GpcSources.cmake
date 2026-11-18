# Shared source lists for libgp (CIP-0020). Keep in sync with makefile libgp objects.
# Fortran entry is ndlfortran.f (not the archived f2c ndlfortran.c).

set(GPC_LIBGP_CXX_SOURCES
  ${CMAKE_SOURCE_DIR}/CClctrl.cpp
  ${CMAKE_SOURCE_DIR}/CGp.cpp
  ${CMAKE_SOURCE_DIR}/CInducing.cpp
  ${CMAKE_SOURCE_DIR}/CCholAdjoint.cpp
  ${CMAKE_SOURCE_DIR}/CLQAdjoint.cpp
  ${CMAKE_SOURCE_DIR}/CMatrix.cpp
  ${CMAKE_SOURCE_DIR}/CNoise.cpp
  ${CMAKE_SOURCE_DIR}/ndlutil.cpp
  ${CMAKE_SOURCE_DIR}/ndlstrutil.cpp
  ${CMAKE_SOURCE_DIR}/CTransform.cpp
  ${CMAKE_SOURCE_DIR}/COptimisable.cpp
  ${CMAKE_SOURCE_DIR}/CKern.cpp
  ${CMAKE_SOURCE_DIR}/CDist.cpp
  ${CMAKE_SOURCE_DIR}/ndlassert.cpp
)

set(GPC_LIBGP_FORTRAN_SOURCES
  ${CMAKE_SOURCE_DIR}/ndlfortran.f
)

# Extra translation units linked into specific CLIs (not part of libgp.a).
set(GPC_IVM_EXTRA_SOURCES ${CMAKE_SOURCE_DIR}/CIvm.cpp)
set(GPC_GPLVM_EXTRA_SOURCES ${CMAKE_SOURCE_DIR}/CGplvm.cpp)
set(GPC_BGPLVM_EXTRA_SOURCES ${CMAKE_SOURCE_DIR}/CBayesGplvm.cpp)
