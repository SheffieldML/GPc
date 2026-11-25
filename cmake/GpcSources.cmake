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
  ${GPC_ROOT}/CEpInference.cpp
  ${GPC_ROOT}/CLaplaceInference.cpp
  ${GPC_ROOT}/ndlutil.cpp
  ${GPC_ROOT}/ndlstrutil.cpp
  ${GPC_ROOT}/CTransform.cpp
  ${GPC_ROOT}/COptimisable.cpp
  ${GPC_ROOT}/CKern.cpp
  ${GPC_ROOT}/CMultioutput.cpp
  ${GPC_ROOT}/CDist.cpp
  ${GPC_ROOT}/ndlassert.cpp
  # CIP-001D follow-on: MAP / collapsed Bayesian GPLVM / MRD for the Python façade.
  ${GPC_ROOT}/CMltools.cpp
  ${GPC_ROOT}/CGplvm.cpp
  ${GPC_ROOT}/CBayesGplvm.cpp
  ${GPC_ROOT}/CMrd.cpp
  # CIP-0035 / CIP-002A: SVGP stack + Adam (needed by gpc.models.DeepGP).
  ${GPC_ROOT}/CSvgp.cpp
  ${GPC_ROOT}/CAdam.cpp
  ${GPC_ROOT}/CStochasticOptimisable.cpp
  ${GPC_ROOT}/CDeepGp.cpp
  ${GPC_ROOT}/CHmc.cpp
  ${GPC_ROOT}/CVariationalTarget.cpp
)

set(GPC_LIBGP_FORTRAN_SOURCES
  ${GPC_ROOT}/ndlfortran.f
)

set(GPC_IVM_EXTRA_SOURCES ${GPC_ROOT}/CIvm.cpp)
# CGplvm / CBayesGplvm / CMltools / SVGP / Adam / CDeepGp live in gpc_lib.
set(GPC_GPLVM_EXTRA_SOURCES)
# SVI Bayesian GPLVM only — collapsed Bayes GPLVM and SVGP/Adam are in gpc_lib.
set(GPC_BGPLVM_EXTRA_SOURCES
  ${GPC_ROOT}/CSviBayesGplvm.cpp
)
# Match makefile ppa link line: CPpa + CSparsePpa (CMltools is in gpc_lib).
set(GPC_PPA_EXTRA_SOURCES
  ${GPC_ROOT}/CPpa.cpp
  ${GPC_ROOT}/CSparsePpa.cpp
)
