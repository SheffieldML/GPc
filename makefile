#
#  Top Level Makefile for GPLVM
#  Version 0.11
#  July 6, 2005
#  Dec 23, 2008
# dependencies created with gcc -MM XXX.cpp

# Platform flags: copy make.linux (Ubuntu/CI) or make.osx (Homebrew macOS) to
# make.inc for a local override. Without make.inc, make.linux is the default.
ifeq ($(wildcard make.inc),make.inc)
include make.inc
else
include make.linux
endif

all: gplvm bgplvm deepgp ivm gp ppa libgp$(LIBSEXT) libgp$(LIBDEXT)

# CIP-001D: Python façade (optional; needs python/.venv with pybind11+numpy).
.PHONY: python-gpc python-gpc-test
python-gpc: libgp$(LIBSEXT)
	$(MAKE) -C python
python-gpc-test: python-gpc
	$(MAKE) -C python test

# CIP-0013: objective+gradient timing harness (use RELEASE=1 for timed builds).
bench: benchGPc
benchGPc: bench/benchGPc.o CGp.o CSvgp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o benchGPc bench/benchGPc.o CGp.o CSvgp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o $(LDFLAGS)

bench/benchGPc.o: bench/benchGPc.cpp CGp.h CSvgp.h CKern.h CNoise.h
	$(CC) -c bench/benchGPc.cpp -o bench/benchGPc.o $(CCFLAGS)

# CIP-0015 API demos (docs/tutorials/) — Stage 1 + Phase B
demos: demo_regression_1d demo_sparse_dtcvar demo_compound_paths demo_coregionalization_toy demo_bayes_gplvm demo_mrd

demo_regression_1d: demo_regression_1d.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o demo_regression_1d demo_regression_1d.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

demo_regression_1d.o: examples/demo_regression_1d.cpp CGp.h CKern.h CNoise.h CClctrl.h
	$(CC) -c examples/demo_regression_1d.cpp -o demo_regression_1d.o $(CCFLAGS)

demo_sparse_dtcvar: demo_sparse_dtcvar.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o demo_sparse_dtcvar demo_sparse_dtcvar.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

demo_sparse_dtcvar.o: examples/demo_sparse_dtcvar.cpp CGp.h CKern.h CNoise.h CClctrl.h
	$(CC) -c examples/demo_sparse_dtcvar.cpp -o demo_sparse_dtcvar.o $(CCFLAGS)

demo_compound_paths: demo_compound_paths.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o demo_compound_paths demo_compound_paths.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

demo_compound_paths.o: examples/demo_compound_paths.cpp CGp.h CKern.h CNoise.h CClctrl.h
	$(CC) -c examples/demo_compound_paths.cpp -o demo_compound_paths.o $(CCFLAGS)

demo_coregionalization_toy: demo_coregionalization_toy.o CMultioutput.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o demo_coregionalization_toy demo_coregionalization_toy.o CMultioutput.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

demo_coregionalization_toy.o: examples/demo_coregionalization_toy.cpp CMultioutput.h CGp.h CKern.h CNoise.h
	$(CC) -c examples/demo_coregionalization_toy.cpp -o demo_coregionalization_toy.o $(CCFLAGS)

# CIP-0017 Phase 2: Bayesian GP-LVM demo.
demo_bayes_gplvm: demo_bayes_gplvm.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o demo_bayes_gplvm demo_bayes_gplvm.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

demo_bayes_gplvm.o: examples/demo_bayes_gplvm.cpp CBayesGplvm.h CKern.h
	$(CC) -c examples/demo_bayes_gplvm.cpp -o demo_bayes_gplvm.o $(CCFLAGS)

gplvm: gplvm.o CClctrl.o CGplvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o gplvm gplvm.o CGplvm.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

gplvm.o: gplvm.cpp gplvm.h ndlexceptions.h ndlstrutil.h CMatrix.h \
  ndlassert.h CNdlInterfaces.h ndlutil.h ndlfortran.h lapack.h CKern.h \
  CTransform.h CDataModel.h CDist.h CGplvm.h CMltools.h COptimisable.h \
  CNoise.h CClctrl.h
	$(CC) -c gplvm.cpp -o gplvm.o $(CCFLAGS)

# CIP-0017: Bayesian GP-LVM CLI.
bgplvm: bgplvm.o CClctrl.o CBayesGplvm.o CSviBayesGplvm.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o bgplvm bgplvm.o CBayesGplvm.o CSviBayesGplvm.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o $(LDFLAGS)

bgplvm.o: bgplvm.cpp bgplvm.h CBayesGplvm.h CSviBayesGplvm.h CAdam.h CClctrl.h CKern.h
	$(CC) -c bgplvm.cpp -o bgplvm.o $(CCFLAGS)

# CIP-0035: deep GP CLI.
deepgp: deepgp.o CClctrl.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o deepgp deepgp.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o $(LDFLAGS)

deepgp.o: deepgp.cpp deepgp.h CDeepGp.h CAdam.h CClctrl.h CKern.h
	$(CC) -c deepgp.cpp -o deepgp.o $(CCFLAGS)

ivm: ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o ivm  ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

ivm.o: ivm.cpp CIvm.h CKern.h CMatrix.h ivm.h CClctrl.h
	$(CC) -c ivm.cpp -o ivm.o $(CCFLAGS)

# CIP-0023 dense PPA + CIP-0026 sparse (CSparsePpa / CGp) learn path.
ppa: ppa.o CClctrl.o CPpa.o CSparsePpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o ppa ppa.o CClctrl.o CPpa.o CSparsePpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o $(LDFLAGS)

ppa.o: ppa.cpp CPpa.h CSparsePpa.h CKern.h CMatrix.h ppa.h CClctrl.h
	$(CC) -c ppa.cpp -o ppa.o $(CCFLAGS)

gp: gp.o CHmc.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o gp gp.o CHmc.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

# CIP-001D / CIP-0035: GPLVM façades + DeepGP (SVGP/Adam) in libgp.
LIBGP_OBJS = CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CMultioutput.o CDist.o ndlassert.o CMltools.o CGplvm.o CBayesGplvm.o CMrd.o CSvgp.o CAdam.o CStochasticOptimisable.o CDeepGp.o CHmc.o CVariationalTarget.o

libgp$(LIBSEXT): CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CMultioutput.o CDist.o ndlassert.o CMltools.o CGplvm.o CBayesGplvm.o CMrd.o CSvgp.o CAdam.o CStochasticOptimisable.o CDeepGp.o CHmc.o CVariationalTarget.o
	$(LIBSCOMMAND) libgp$(LIBSEXT) $(LIBGP_OBJS)

libgp$(LIBDEXT): CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CMultioutput.o CDist.o ndlassert.o CMltools.o CGplvm.o CBayesGplvm.o CMrd.o CSvgp.o CAdam.o CStochasticOptimisable.o CDeepGp.o CHmc.o CVariationalTarget.o
	$(LIBDCOMMAND) libgp$(LIBDEXT) $(LIBGP_OBJS) $(LDFLAGS)

gp.o: gp.cpp gp.h ndlexceptions.h ndlstrutil.h CMatrix.h ndlassert.h \
  CNdlInterfaces.h ndlutil.h ndlfortran.h lapack.h CKern.h CTransform.h \
  CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h CClctrl.h CHmc.h CPosteriorTarget.h CVariationalTarget.h
	$(CC) -c gp.cpp -o gp.o $(CCFLAGS)


# Full historical suite still needs the MATLAB interface (_NDLMATLAB) for object I/O
# (kernels/noise/models as structs). Numeric fixtures can use libmatio instead:
#   make USE_MATIO=1 matio-tests
tests: testDist testGp testIvm testKern testMatrix testMltools testNdlutil testNoise testNoiseBatched testNoiseInLikelihood testSkewNoise testLikelihoodZoo testCip0027Perf testIvmSoftener testEp testTransform testOptimisable testParamPaths testCGp testCoregionalize testPsi testBayesGplvm testMrd testCholAdjoint testLQAdjoint testClctrl testPpa testSparsePpa testSvgp testSviBayesGplvm testDeepGp testDiagGradParams testInducingGradZ testInducingHermite testNoisePrior testHmc testVariationalHmc

# Reference tests runnable without MATLAB (CIP-0001 / REQ-0002): numeric .mat via matio.
# testOptimisable needs no .mat fixtures but is part of the hard gate (CIP-0010).
# testParamPaths is CIP-0014 Phase A (no .mat). testCGp is CIP-0011.
# testCoregionalize is CIP-000D (no .mat). testPsi / testBayesGplvm are CIP-0017.
# testMrd is CIP-0018.
# testCholAdjoint is CIP-0012 (no .mat); must run under coverage so CCholAdjoint.cpp is hit.
# testClctrl is CIP-0022 (CSV/SVML data I/O; no .mat).
# testNoiseBatched / testNoiseInLikelihood / testCip0027Perf / testIvmSoftener are CIP-0027 (no .mat).
matio-tests: testDist testKern testMatrix testNdlutil testNoise testNoiseBatched testNoiseInLikelihood testSkewNoise testLikelihoodZoo testCip0027Perf testIvmSoftener testEp testTransform testGp testOptimisable testParamPaths testCGp testCholAdjoint testLQAdjoint testCoregionalize testPsi testBayesGplvm testMrd testClctrl testDiagGradParams testInducingGradZ testInducingHermite testNoisePrior testHmc testVariationalHmc

# Run matio reference binaries (CIP-0004 coverage workflow uses this after a COVERAGE=1 build).
run-matio-tests: matio-tests
	./testNdlutil
	./testTransform
	./testDist
	./testNoise
	./testNoiseBatched
	./testNoiseInLikelihood
	./testSkewNoise
	./testLikelihoodZoo
	./testCip0027Perf
	./testIvmSoftener
	./testEp
	./testKern
	./testDiagGradParams
	./testInducingGradZ
	./testInducingHermite
	./testNoisePrior
	./testHmc
	./testVariationalHmc
	./testMatrix
	./testOptimisable
	./testParamPaths
	./testGp
	./testCGp
	./testCholAdjoint
	./testLQAdjoint
	./testCoregionalize
	./testClctrl
	./testPsi
	./testBayesGplvm
	./testMrd

# CIP-0017: CLI smoke for ./bgplvm (also used under COVERAGE=1 for Codecov).
run-bgplvm-smoke: bgplvm
	chmod +x scripts/smoke_bgplvm.sh
	./scripts/smoke_bgplvm.sh

# CIP-0035: CLI smoke for ./deepgp.
run-deepgp-smoke: deepgp
	chmod +x scripts/smoke_deepgp.sh
	./scripts/smoke_deepgp.sh

# CIP-0019: adversarial / extreme-input numerical stability (separate from matio-tests).
testNumericalStability: numerical_stability/testNumericalStability.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testNumericalStability numerical_stability/testNumericalStability.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

numerical_stability/testNumericalStability.o: numerical_stability/testNumericalStability.cpp CGp.h CKern.h CMatrix.h CNoise.h
	$(CC) -c numerical_stability/testNumericalStability.cpp -o numerical_stability/testNumericalStability.o $(CCFLAGS)

run-numerical-stability: testNumericalStability
	./testNumericalStability

# CIP-0012 dual-path: StableFactor vs Legacy under the numerical-stability suite.
compare-sparse-modes: testNumericalStability
	python3 numerical_stability/compare_sparse_modes.py --skip-build


testDist: testDist.o CMatrix.o ndlfortran.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testDist testDist.o CMatrix.o ndlfortran.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS) 

testDist.o: testDist.cpp CDist.h CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testDist.cpp -o testDist.o $(CCFLAGS)

testGp: testGp.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testGp  testGp.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testGp.o: testGp.cpp CGp.h CKern.h CMatrix.h CClctrl.h
	$(CC) -c testGp.cpp -o testGp.o $(CCFLAGS)

# Maintainer diagnostics / fixture stampers live under tools/ (binaries still at repo root).
# DTC/DTCVAR gradient + GPy-aligned reference dump (backlog 2026-11-15_dtcvar-numerical-diagnostic).
diagDtcvar: tools/diagDtcvar.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagDtcvar tools/diagDtcvar.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/diagDtcvar.o: tools/diagDtcvar.cpp CGp.h CKern.h CMatrix.h
	$(CC) -c tools/diagDtcvar.cpp -o tools/diagDtcvar.o $(CCFLAGS)

# CIP-0012: Legacy vs StableFactor γ×M stress sweep (pair with scripts/sparse_mp_reference.py).
diagSparseStable: tools/diagSparseStable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagSparseStable tools/diagSparseStable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/diagSparseStable.o: tools/diagSparseStable.cpp CGp.h CKern.h CMatrix.h
	$(CC) -c tools/diagSparseStable.cpp -o tools/diagSparseStable.o $(CCFLAGS)

# CIP-001C pilot: white-noise inducing vs point inducing (hand closed-form sweep).
diagWhiteNoisePilot: tools/diagWhiteNoisePilot.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagWhiteNoisePilot tools/diagWhiteNoisePilot.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o $(LDFLAGS)

tools/diagWhiteNoisePilot.o: tools/diagWhiteNoisePilot.cpp CMatrix.h
	$(CC) -c tools/diagWhiteNoisePilot.cpp -o tools/diagWhiteNoisePilot.o $(CCFLAGS)

# CIP-001C pilot: full SCG — point DTCVAR (CGp) vs VIK white-noise features.
diagVikCompare: tools/diagVikCompare.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagVikCompare tools/diagVikCompare.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/diagVikCompare.o: tools/diagVikCompare.cpp CGp.h CKern.h CMatrix.h COptimisable.h
	$(CC) -c tools/diagVikCompare.cpp -o tools/diagVikCompare.o $(CCFLAGS)

diagPitc: tools/diagPitc.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagPitc tools/diagPitc.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/diagPitc.o: tools/diagPitc.cpp CGp.h CKern.h CMatrix.h
	$(CC) -c tools/diagPitc.cpp -o tools/diagPitc.o $(CCFLAGS)

genTestGpPitc: tools/genTestGpPitc.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o genTestGpPitc tools/genTestGpPitc.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/genTestGpPitc.o: tools/genTestGpPitc.cpp CGp.h CKern.h CMatrix.h
	$(CC) -c tools/genTestGpPitc.cpp -o tools/genTestGpPitc.o $(CCFLAGS)

# Fixture generator for matfiles/testGpdtcvar.mat (scripts/gen_testGpdtcvar.py).
genTestGpDtcvar: tools/genTestGpDtcvar.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o genTestGpDtcvar tools/genTestGpDtcvar.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/genTestGpDtcvar.o: tools/genTestGpDtcvar.cpp CGp.h CKern.h CMatrix.h
	$(CC) -c tools/genTestGpDtcvar.cpp -o tools/genTestGpDtcvar.o $(CCFLAGS)
 
testIvm: testIvm.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testIvm  testIvm.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testIvm.o: testIvm.cpp CIvm.h CKern.h CMatrix.h CClctrl.h
	$(CC) -c testIvm.cpp -o testIvm.o $(CCFLAGS)

# CIP-0023: PPA vs Octave Phase 0 fixtures (no matio required).
testPpa: testPpa.o CPpa.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testPpa testPpa.o CPpa.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o $(LDFLAGS)

testPpa.o: testPpa.cpp CPpa.h CKern.h CMatrix.h
	$(CC) -c testPpa.cpp -o testPpa.o $(CCFLAGS)

# CIP-0026 prototype: CSparsePpa on CGp (DTC M=N vs dense). Not a release gate yet.
testSparsePpa: testSparsePpa.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testSparsePpa testSparsePpa.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testSparsePpa.o: testSparsePpa.cpp CSparsePpa.h CPpa.h CGp.h CKern.h CMatrix.h
	$(CC) -c testSparsePpa.cpp -o testSparsePpa.o $(CCFLAGS)

CSparsePpa.o: CSparsePpa.cpp CSparsePpa.h CGp.h CKern.h CNoise.h
	$(CC) -c CSparsePpa.cpp -o CSparsePpa.o $(CCFLAGS)

# CIP-002A: SVGP q(u) natural gradients, ELBO grads, Adam (Gaussian).
testSvgp: testSvgp.o CSvgp.o CAdam.o CStochasticOptimisable.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testSvgp testSvgp.o CSvgp.o CAdam.o CStochasticOptimisable.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testSvgp.o: testSvgp.cpp CSvgp.h CAdam.h CGp.h CKern.h CMatrix.h CNoise.h CSparsePpa.h
	$(CC) -c testSvgp.cpp -o testSvgp.o $(CCFLAGS)

# CIP-0030 Phase 1: SVI Bayesian GP-LVM (sampled q(X) on CSvgp).
testSviBayesGplvm: testSviBayesGplvm.o CSviBayesGplvm.o CSvgp.o CAdam.o CStochasticOptimisable.o CBayesGplvm.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testSviBayesGplvm testSviBayesGplvm.o CSviBayesGplvm.o CSvgp.o CAdam.o CStochasticOptimisable.o CBayesGplvm.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testSviBayesGplvm.o: testSviBayesGplvm.cpp CSviBayesGplvm.h CBayesGplvm.h CSvgp.h CAdam.h CKern.h CMatrix.h
	$(CC) -c testSviBayesGplvm.cpp -o testSviBayesGplvm.o $(CCFLAGS)

CSviBayesGplvm.o: CSviBayesGplvm.cpp CSviBayesGplvm.h CSvgp.h
	$(CC) -c CSviBayesGplvm.cpp -o CSviBayesGplvm.o $(CCFLAGS)

CSvgp.o: CSvgp.cpp CSvgp.h CAdam.h CStochasticOptimisable.h CKern.h CMatrix.h CCholAdjoint.h CInducing.h
	$(CC) -c CSvgp.cpp -o CSvgp.o $(CCFLAGS)

# CIP-0035 Phase 1: deep GP (DSVI on CSvgp layers).
testDeepGp: testDeepGp.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testDeepGp testDeepGp.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testDeepGp.o: testDeepGp.cpp CDeepGp.h CSvgp.h CKern.h CMatrix.h
	$(CC) -c testDeepGp.cpp -o testDeepGp.o $(CCFLAGS)

CDeepGp.o: CDeepGp.cpp CDeepGp.h CSvgp.h
	$(CC) -c CDeepGp.cpp -o CDeepGp.o $(CCFLAGS)

CAdam.o: CAdam.cpp CAdam.h CStochasticOptimisable.h CMatrix.h
	$(CC) -c CAdam.cpp -o CAdam.o $(CCFLAGS)

CStochasticOptimisable.o: CStochasticOptimisable.cpp CStochasticOptimisable.h
	$(CC) -c CStochasticOptimisable.cpp -o CStochasticOptimisable.o $(CCFLAGS)

testKern: testKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testKern testKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS) 

testKern.o: testKern.cpp CKern.h CDist.h CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testKern.cpp -o testKern.o $(CCFLAGS)

# CIP-002D follow-up: bulk getDiagGradParams (regularise-once + O(1)-in-N allocs).
testDiagGradParams: testDiagGradParams.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testDiagGradParams testDiagGradParams.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testDiagGradParams.o: testDiagGradParams.cpp CKern.h CDist.h CTransform.h CMatrix.h
	$(CC) -c testDiagGradParams.cpp -o testDiagGradParams.o $(CCFLAGS)

# CIP-002D follow-up: batched inducingGradZ + MLP bulk compute.
testInducingGradZ: testInducingGradZ.o CInducing.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testInducingGradZ testInducingGradZ.o CInducing.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testInducingGradZ.o: testInducingGradZ.cpp CKern.h CInducing.h CMatrix.h
	$(CC) -c testInducingGradZ.cpp -o testInducingGradZ.o $(CCFLAGS)

# CIP-003D phase 1: Hermite inducing layout, capacity and audit.
testInducingHermite: testInducingHermite.o CInducing.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testInducingHermite testInducingHermite.o CInducing.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testInducingHermite.o: testInducingHermite.cpp CKern.h CInducing.h CMatrix.h
	$(CC) -c testInducingHermite.cpp -o testInducingHermite.o $(CCFLAGS)

# CIP-0017 Phase 1: psi statistics (MC + finite differences).
testPsi: testPsi.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testPsi testPsi.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testPsi.o: testPsi.cpp CKern.h CDist.h CTransform.h CMatrix.h
	$(CC) -c testPsi.cpp -o testPsi.o $(CCFLAGS)

# CIP-001C Phase 3: Hankel / spectralDensity white-noise K_fu gate.
testWnHankel: testWnHankel.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testWnHankel testWnHankel.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testWnHankel.o: testWnHankel.cpp CKern.h CDist.h CTransform.h CMatrix.h
	$(CC) -c testWnHankel.cpp -o testWnHankel.o $(CCFLAGS)

# CIP-0017 Phase 2: Bayesian GP-LVM.
CBayesGplvm.o: CBayesGplvm.cpp CBayesGplvm.h CKern.h COptimisable.h CTransform.h CMatrix.h
	$(CC) -c CBayesGplvm.cpp -o CBayesGplvm.o $(CCFLAGS)

testBayesGplvm: testBayesGplvm.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testBayesGplvm testBayesGplvm.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testBayesGplvm.o: testBayesGplvm.cpp CBayesGplvm.h CKern.h
	$(CC) -c testBayesGplvm.cpp -o testBayesGplvm.o $(CCFLAGS)

# CIP-0018: Manifold Relevance Determination.
CMrd.o: CMrd.cpp CMrd.h CBayesGplvm.h CKern.h COptimisable.h CTransform.h CMatrix.h
	$(CC) -c CMrd.cpp -o CMrd.o $(CCFLAGS)

testMrd: testMrd.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testMrd testMrd.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testMrd.o: testMrd.cpp CMrd.h CBayesGplvm.h CKern.h
	$(CC) -c testMrd.cpp -o testMrd.o $(CCFLAGS)

demo_mrd: examples/demo_mrd.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o demo_mrd examples/demo_mrd.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS)

examples/demo_mrd.o: examples/demo_mrd.cpp CMrd.h CBayesGplvm.h CKern.h
	$(CC) -c examples/demo_mrd.cpp -o examples/demo_mrd.o $(CCFLAGS)

testCoregionalize: testCoregionalize.o CMultioutput.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testCoregionalize testCoregionalize.o CMultioutput.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testCoregionalize.o: testCoregionalize.cpp CKern.h CMultioutput.h CGp.h CDist.h CTransform.h CMatrix.h
	$(CC) -c testCoregionalize.cpp -o testCoregionalize.o $(CCFLAGS)

CMultioutput.o: CMultioutput.cpp CMultioutput.h CKern.h CGp.h CNoise.h
	$(CC) -c CMultioutput.cpp -o CMultioutput.o $(CCFLAGS)

testMatrix: testMatrix.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testMatrix testMatrix.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o ndlassert.o $(LDFLAGS) 

testMatrix.o: testMatrix.cpp CMatrix.h CClctrl.h
	$(CC) -c testMatrix.cpp  -o testMatrix.o $(CCFLAGS)

# CIP-0022: CSV / SVML data loaders.
testClctrl: testClctrl.o CClctrl.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testClctrl testClctrl.o CClctrl.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o $(LDFLAGS)

testClctrl.o: testClctrl.cpp CClctrl.h CMatrix.h
	$(CC) -c testClctrl.cpp -o testClctrl.o $(CCFLAGS)

# CIP-0012: Murray Cholesky / trsm adjoint primitives.
CCholAdjoint.o: CCholAdjoint.cpp CCholAdjoint.h CMatrix.h
	$(CC) -c CCholAdjoint.cpp -o CCholAdjoint.o $(CCFLAGS)

CLQAdjoint.o: CLQAdjoint.cpp CLQAdjoint.h CMatrix.h lapack.h
	$(CC) -c CLQAdjoint.cpp -o CLQAdjoint.o $(CCFLAGS)

testCholAdjoint: testCholAdjoint.o CCholAdjoint.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testCholAdjoint testCholAdjoint.o CCholAdjoint.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o $(LDFLAGS)

testCholAdjoint.o: testCholAdjoint.cpp CCholAdjoint.h CMatrix.h
	$(CC) -c testCholAdjoint.cpp -o testCholAdjoint.o $(CCFLAGS)

testLQAdjoint: testLQAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testLQAdjoint testLQAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o $(LDFLAGS)

testLQAdjoint.o: testLQAdjoint.cpp CLQAdjoint.h CMatrix.h
	$(CC) -c testLQAdjoint.cpp -o testLQAdjoint.o $(CCFLAGS)

testOptimisable: testOptimisable.o COptimisable.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testOptimisable testOptimisable.o COptimisable.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o ndlassert.o $(LDFLAGS)

testOptimisable.o: testOptimisable.cpp COptimisable.h CMatrix.h
	$(CC) -c testOptimisable.cpp -o testOptimisable.o $(CCFLAGS)

# CIP-0014 Phase A: GPy-like parameter paths (no matio fixtures).
testParamPaths: testParamPaths.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testParamPaths testParamPaths.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o ndlassert.o $(LDFLAGS)

testParamPaths.o: testParamPaths.cpp CKern.h CTransform.h CMatrix.h
	$(CC) -c testParamPaths.cpp -o testParamPaths.o $(CCFLAGS)

testCGp: testCGp.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testCGp testCGp.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testCGp.o: testCGp.cpp CGp.h CKern.h CMatrix.h
	$(CC) -c testCGp.cpp -o testCGp.o $(CCFLAGS)

testNoisePrior: testNoisePrior.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testNoisePrior testNoisePrior.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testNoisePrior.o: testNoisePrior.cpp CPosteriorTarget.h CGp.h CKern.h CNoise.h CDist.h CMatrix.h
	$(CC) -c testNoisePrior.cpp -o testNoisePrior.o $(CCFLAGS)

# CIP-003B: HMC verification (analytic Gaussian, energy error, GP grid check).
testHmc: testHmc.o CHmc.o CGplvm.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testHmc testHmc.o CHmc.o CGplvm.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testVariationalHmc: testVariationalHmc.o CHmc.o CVariationalTarget.o CPpa.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testVariationalHmc testVariationalHmc.o CHmc.o CVariationalTarget.o CPpa.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o $(LDFLAGS)

testVariationalHmc.o: testVariationalHmc.cpp CHmc.h CVariationalTarget.h CPosteriorTarget.h CBayesGplvm.h CMrd.h CPpa.h CKern.h CNoise.h CDist.h CMatrix.h
	$(CC) -c testVariationalHmc.cpp -o testVariationalHmc.o $(CCFLAGS)

testHmc.o: testHmc.cpp CHmc.h CPosteriorTarget.h CGp.h CGplvm.h CKern.h CNoise.h CDist.h CMatrix.h
	$(CC) -c testHmc.cpp -o testHmc.o $(CCFLAGS)

CHmc.o: CHmc.cpp CHmc.h CPosteriorTarget.h COptimisable.h CMatrix.h
	$(CC) -c CHmc.cpp -o CHmc.o $(CCFLAGS)

CVariationalTarget.o: CVariationalTarget.cpp CVariationalTarget.h CPosteriorTarget.h CBayesGplvm.h CMrd.h COptimisable.h CKern.h CDist.h CMatrix.h
	$(CC) -c CVariationalTarget.cpp -o CVariationalTarget.o $(CCFLAGS)

# CIP-0028: FTC vs DTCVAR ridge free/fixed centres (tighter bound demo).
diagWeightSpaceRidge: tools/diagWeightSpaceRidge.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagWeightSpaceRidge tools/diagWeightSpaceRidge.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o $(LDFLAGS)

tools/diagWeightSpaceRidge.o: tools/diagWeightSpaceRidge.cpp CGp.h CKern.h CNoise.h
	$(CC) -c tools/diagWeightSpaceRidge.cpp -o tools/diagWeightSpaceRidge.o $(CCFLAGS)

# CIP-0027 Phase B′: IVM White softener composition matrix (unitsquare).
diagIvmSoftener: tools/diagIvmSoftener.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagIvmSoftener tools/diagIvmSoftener.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/diagIvmSoftener.o: tools/diagIvmSoftener.cpp CIvm.h CKern.h CNoise.h CClctrl.h
	$(CC) -c tools/diagIvmSoftener.cpp -o tools/diagIvmSoftener.o $(CCFLAGS)

testMltools: testMltools.o CMltools.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testMltools  testMltools.o CMltools.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CClctrl.o ndlassert.o $(LDFLAGS)

testMltools.o: testMltools.cpp CMltools.h CKern.h CMatrix.h CClctrl.h 
	$(CC) -c testMltools.cpp -o testMltools.o $(CCFLAGS)

testNdlutil: testNdlutil.o ndlutil.o ndlstrutil.o CMatrix.o ndlfortran.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testNdlutil testNdlutil.o ndlutil.o ndlstrutil.o CMatrix.o ndlfortran.o CClctrl.o ndlassert.o $(LDFLAGS)

testNdlutil.o: testNdlutil.cpp ndlutil.h CClctrl.h
	$(CC) -c testNdlutil.cpp -o testNdlutil.o $(CCFLAGS)

testNoise: testNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testNoise  testNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o $(LDFLAGS)

testNoise.o: testNoise.cpp CNoise.h CMatrix.h CClctrl.h
	$(CC) -c testNoise.cpp -o testNoise.o $(CCFLAGS)

# CIP-0027 Phase 0: batched Gaussian helpers (no .mat).
testNoiseBatched: testNoiseBatched.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testNoiseBatched testNoiseBatched.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o $(LDFLAGS)

testNoiseBatched.o: testNoiseBatched.cpp CNoise.h CMatrix.h
	$(CC) -c testNoiseBatched.cpp -o testNoiseBatched.o $(CCFLAGS)

# CIP-003A Phase 0/1: Mills helpers + skew likelihoods (no .mat).
testSkewNoise: testSkewNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testSkewNoise testSkewNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o $(LDFLAGS)

testSkewNoise.o: testSkewNoise.cpp CNoise.h CMatrix.h
	$(CC) -c testSkewNoise.cpp -o testSkewNoise.o $(CCFLAGS)

# CIP-003A Phase 2–5: likelihood zoo dens/VE + generalised Laplace on CGp.
testLikelihoodZoo: testLikelihoodZoo.o CLaplaceInference.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testLikelihoodZoo testLikelihoodZoo.o CLaplaceInference.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testLikelihoodZoo.o: testLikelihoodZoo.cpp CNoise.h CGp.h CKern.h CLaplaceInference.h CMatrix.h
	$(CC) -c testLikelihoodZoo.cpp -o testLikelihoodZoo.o $(CCFLAGS)

# CIP-0027 Phase A: likelihood.variance on FTC Gram + path aliases.
testNoiseInLikelihood: testNoiseInLikelihood.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testNoiseInLikelihood testNoiseInLikelihood.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testNoiseInLikelihood.o: testNoiseInLikelihood.cpp CGp.h CKern.h CNoise.h CMatrix.h
	$(CC) -c testNoiseInLikelihood.cpp -o testNoiseInLikelihood.o $(CCFLAGS)

# CIP-0027 Phase C: dual-noise harden + FTC/sparse/IVM algorithm performance gates.
testCip0027Perf: testCip0027Perf.o CGp.o CIvm.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testCip0027Perf testCip0027Perf.o CGp.o CIvm.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testCip0027Perf.o: testCip0027Perf.cpp CGp.h CIvm.h CKern.h CNoise.h CMatrix.h CClctrl.h
	$(CC) -c testCip0027Perf.cpp -o testCip0027Perf.o $(CCFLAGS)

# CIP-0027: deeper IVM White softener gates (long budget, getWhite, stream).
testIvmSoftener: testIvmSoftener.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testIvmSoftener testIvmSoftener.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testIvmSoftener.o: testIvmSoftener.cpp CIvm.h CKern.h CNoise.h CMatrix.h CClctrl.h
	$(CC) -c testIvmSoftener.cpp -o testIvmSoftener.o $(CCFLAGS)

# CIP-0033 Phase A: GPy-mirrored Bernoulli–probit ep_moments + dense EP.
testEp: testEp.o CEpInference.o CLaplaceInference.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testEp testEp.o CEpInference.o CLaplaceInference.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

testEp.o: testEp.cpp CEpInference.h CLaplaceInference.h CGp.h CNoise.h CKern.h CMatrix.h
	$(CC) -c testEp.cpp -o testEp.o $(CCFLAGS)

CEpInference.o: CEpInference.cpp CEpInference.h CNoise.h CMatrix.h ndlutil.h
	$(CC) -c CEpInference.cpp -o CEpInference.o $(CCFLAGS)

CLaplaceInference.o: CLaplaceInference.cpp CLaplaceInference.h CNoise.h CMatrix.h ndlutil.h
	$(CC) -c CLaplaceInference.cpp -o CLaplaceInference.o $(CCFLAGS)

testTransform: testTransform.o CMatrix.o ndlfortran.o  CTransform.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testTransform testTransform.o CMatrix.o ndlfortran.o CTransform.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o $(LDFLAGS) 

testTransform.o: testTransform.cpp CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testTransform.cpp -o testTransform.o $(CCFLAGS)

CClctrl.o: CClctrl.cpp CClctrl.h ndlstrutil.h ndlexceptions.h ndlutil.h \
  ndlassert.h ndlfortran.h CMatrix.h CNdlInterfaces.h lapack.h
	$(CC) -c CClctrl.cpp -o CClctrl.o $(CCFLAGS)

CGplvm.o: CGplvm.cpp CGplvm.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CGplvm.cpp -o CGplvm.o $(CCFLAGS)

CNoise.o: CNoise.cpp CNoise.h ndlexceptions.h ndlutil.h ndlassert.h \
  ndlfortran.h ndlstrutil.h CMatrix.h CNdlInterfaces.h lapack.h \
  CTransform.h COptimisable.h CDist.h CKern.h CDataModel.h
	$(CC) -c CNoise.cpp -o CNoise.o $(CCFLAGS)

CKern.o: CKern.cpp CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h \
  CDataModel.h CDist.h
	$(CC) -c CKern.cpp -o CKern.o $(CCFLAGS)

CTransform.o: CTransform.cpp CTransform.h CMatrix.h ndlassert.h \
  ndlexceptions.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h \
  lapack.h
	$(CC) -c CTransform.cpp -o CTransform.o $(CCFLAGS)

COptimisable.o: COptimisable.cpp COptimisable.h CMatrix.h ndlassert.h \
  ndlexceptions.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h \
  lapack.h
	$(CC) -c COptimisable.cpp -o COptimisable.o $(CCFLAGS)

CDist.o: CDist.cpp CDist.h CMatrix.h ndlassert.h ndlexceptions.h \
  CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h \
  CTransform.h
	$(CC) -c CDist.cpp -o CDist.o $(CCFLAGS)

CMatrix.o: CMatrix.cpp CMatrix.h ndlassert.h ndlexceptions.h \
  CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h
	$(CC) -c CMatrix.cpp -o CMatrix.o $(CCFLAGS)

CMltools.o: CMltools.cpp CMltools.h
	$(CC) -c CMltools.cpp -o CMltools.o $(CCFLAGS)

CInducing.o: CInducing.cpp CInducing.h CKern.h CMatrix.h ndlexceptions.h
	$(CC) -c CInducing.cpp -o CInducing.o $(CCFLAGS)

CGp.o: CGp.cpp CGp.h CInducing.h CCholAdjoint.h CLQAdjoint.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h CEpInference.h CLaplaceInference.h
	$(CC) -c CGp.cpp -o CGp.o $(CCFLAGS)

CIvm.o: CIvm.cpp CIvm.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CIvm.cpp -o CIvm.o $(CCFLAGS)

CPpa.o: CPpa.cpp CPpa.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CPpa.cpp -o CPpa.o $(CCFLAGS)

ndlutil.o: ndlutil.cpp ndlutil.h ndlassert.h ndlexceptions.h ndlfortran.h
	$(CC) -c ndlutil.cpp -o ndlutil.o $(CCFLAGS)

ndlstrutil.o: ndlstrutil.cpp ndlstrutil.h ndlexceptions.h
	$(CC) -c ndlstrutil.cpp -o ndlstrutil.o $(CCFLAGS)

ndlassert.o: ndlassert.cpp ndlassert.h ndlexceptions.h
	$(CC) -c ndlassert.cpp -o ndlassert.o $(CCFLAGS)

# Collected FORTRAN utilities.
ndlfortran.o: ndlfortran.f
	$(FC) -c ndlfortran.f -o ndlfortran.o $(FCFLAGS)


clean:
	rm -f *.o *.gcda *.gcno *.gcov coverage.info
	rm -f tools/*.o tools/*.gcda tools/*.gcno
	rm -f demo_regression_1d demo_sparse_dtcvar demo_compound_paths demo_coregionalization_toy
	rm -rf coverage-html

diagLqVsStable: tools/diagLqVsStable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o diagLqVsStable tools/diagLqVsStable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o $(LDFLAGS)

tools/diagLqVsStable.o: tools/diagLqVsStable.cpp CGp.h
	$(CC) -c tools/diagLqVsStable.cpp -o tools/diagLqVsStable.o $(CCFLAGS)
