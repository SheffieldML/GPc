# Shared source lists for libgp (CIP-0020). Keep in sync with makefile libgp objects.
# Fortran entry is ndlfortran.f (not the archived f2c ndlfortran.c).
# Paths are relative to GPC_ROOT when set, else CMAKE_SOURCE_DIR.

if(NOT DEFINED GPC_ROOT)
  set(GPC_ROOT ${CMAKE_SOURCE_DIR})
endif()

set(GPC_LIBGP_CXX_SOURCES
  ${GPC_ROOT}/CClctrl.cpp
  ${GPC_ROOT}/CGp.cpp
  ${GPC_ROOT}/CInducing.cpp
  ${GPC_ROOT}/CCholAdjoint.cpp
  ${GPC_ROOT}/CLQAdjoint.cpp
  ${GPC_ROOT}/CMatrix.cpp
  ${GPC_ROOT}/CNoise.cpp
  ${GPC_ROOT}/ndlutil.cpp
  ${GPC_ROOT}/ndlstrutil.cpp
  ${GPC_ROOT}/CTransform.cpp
  ${GPC_ROOT}/COptimisable.cpp
  ${GPC_ROOT}/CKern.cpp
  ${GPC_ROOT}/CDist.cpp
  ${GPC_ROOT}/ndlassert.cpp
)

set(GPC_LIBGP_FORTRAN_SOURCES
  ${GPC_ROOT}/ndlfortran.f
)

set(GPC_IVM_EXTRA_SOURCES ${GPC_ROOT}/CIvm.cpp)
set(GPC_GPLVM_EXTRA_SOURCES ${GPC_ROOT}/CGplvm.cpp)
# Match makefile bgplvm link line: collapsed + SVI (CIP-0030) extras.
set(GPC_BGPLVM_EXTRA_SOURCES
  ${GPC_ROOT}/CBayesGplvm.cpp
  ${GPC_ROOT}/CSviBayesGplvm.cpp
  ${GPC_ROOT}/CSvgp.cpp
  ${GPC_ROOT}/CAdam.cpp
  ${GPC_ROOT}/CStochasticOptimisable.cpp
  ${GPC_ROOT}/CMltools.cpp
)
# Match makefile ppa link line: CPpa + CSparsePpa + CMltools (CIP-0026 sparse path).
set(GPC_PPA_EXTRA_SOURCES
  ${GPC_ROOT}/CPpa.cpp
  ${GPC_ROOT}/CSparsePpa.cpp
  ${GPC_ROOT}/CMltools.cpp
)
