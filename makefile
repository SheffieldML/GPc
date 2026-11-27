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

# Optional YAML reading (CIP-0016, gp-scuderia CIP-0001): make YAML=1 [YAMLROOT=prefix].
# Off by default: without it a .yaml path fails with a message that says how to convert.
ifeq ($(YAML),1)
YAMLROOT ?= $(BREW)
CCFLAGS += -DGPC_YAML -I$(YAMLROOT)/include
LDFLAGS += -L$(YAMLROOT)/lib -lyaml-cpp -Wl,-rpath,$(YAMLROOT)/lib
endif

# Automatic header dependencies: -MMD -MP writes X.d beside each X.o listing every
# header the compiler actually read, so no hand-maintained header lists can go stale.
# (After first adopting this, run `make clean` once: objects built earlier have no .d.)
CCFLAGS += -MMD -MP

# Compile any C++ source to the object beside it. Objects whose source lives in a
# different directory (examples/demo_*.o -> ./demo_*.o) keep explicit rules below.
# Objects also depend on .buildflags, which is rewritten only when the compiler or
# flags change (USE_MATIO, RELEASE, CHECKS, COVERAGE, GPC_OPENMP, YAML, make.inc ...),
# so switching configuration rebuilds exactly the objects it affects.
BUILDFLAGS = .buildflags
.PHONY: FORCE
FORCE:
$(BUILDFLAGS): FORCE
	@echo '$(CC) $(CCFLAGS) | $(FC) $(FCFLAGS)' | cmp -s - $@ || echo '$(CC) $(CCFLAGS) | $(FC) $(FCFLAGS)' > $@

%.o: %.cpp $(BUILDFLAGS)
	$(CC) -c $< -o $@ $(CCFLAGS)

# Every executable this makefile can link (used by `clean`).
ALL_BINARIES = benchGPc \
  demo_regression_1d \
  demo_sparse_dtcvar \
  demo_compound_paths \
  demo_coregionalization_toy \
  demo_bayes_gplvm \
  gplvm \
  bgplvm \
  deepgp \
  ivm \
  ppa \
  gp \
  testNumericalStability \
  testDist \
  testGp \
  diagDtcvar \
  diagSparseStable \
  diagWhiteNoisePilot \
  diagVikCompare \
  diagPitc \
  genTestGpPitc \
  genTestGpDtcvar \
  testIvm \
  testPpa \
  testSparsePpa \
  testSvgp \
  testInducingPatches \
  testConvWeightSpace \
  testSviBayesGplvm \
  testDeepGp \
  testKern \
  testDiagGradParams \
  testInducingGradZ \
  testConvKern \
  testInducingPatches \
  testConvWeightSpace \
  testInducingHermite \
  testPsi \
  testWnHankel \
  testBayesGplvm \
  testMrd \
  demo_mrd \
  testCoregionalize \
  testMatrix \
  testClctrl \
  testCholAdjoint \
  testLQAdjoint \
  testOptimisable \
  testParamPaths \
  testCGp \
  testInterchange \
  testNoisePrior \
  testHmc \
  testVariationalHmc \
  diagWeightSpaceRidge \
  diagIvmSoftener \
  testMltools \
  testNdlutil \
  testNoise \
  testNoiseBatched \
  testSkewNoise \
  testLikelihoodZoo \
  testNoiseInLikelihood \
  testCip0027Perf \
  testIvmSoftener \
  testEp \
  testTransform \
  diagLqVsStable

.PHONY: all tests matio-tests run-matio-tests run-bgplvm-smoke run-deepgp-smoke run-numerical-stability compare-sparse-modes bench demos clean

# Pull in generated header dependencies (silently absent on a clean tree).
-include $(wildcard *.d tools/*.d examples/*.d bench/*.d numerical_stability/*.d)

all: gplvm bgplvm deepgp ivm gp ppa libgp$(LIBSEXT) libgp$(LIBDEXT)

# CIP-001D: Python façade (optional; needs python/.venv with pybind11+numpy).
.PHONY: python-gpc python-gpc-test
python-gpc: libgp$(LIBSEXT)
	$(MAKE) -C python
python-gpc-test: python-gpc
	$(MAKE) -C python test

# CIP-0013: objective+gradient timing harness (use RELEASE=1 for timed builds).
bench: benchGPc
benchGPc: bench/benchGPc.o CDeepGp.o CAdam.o CStochasticOptimisable.o CMltools.o CGp.o CSvgp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0015 API demos (docs/tutorials/) — Stage 1 + Phase B
demos: demo_regression_1d demo_sparse_dtcvar demo_compound_paths demo_coregionalization_toy demo_bayes_gplvm demo_mrd

demo_regression_1d: demo_regression_1d.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

demo_regression_1d.o: examples/demo_regression_1d.cpp $(BUILDFLAGS)
	$(CC) -c $< -o $@ $(CCFLAGS)

demo_sparse_dtcvar: demo_sparse_dtcvar.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

demo_sparse_dtcvar.o: examples/demo_sparse_dtcvar.cpp $(BUILDFLAGS)
	$(CC) -c $< -o $@ $(CCFLAGS)

demo_compound_paths: demo_compound_paths.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

demo_compound_paths.o: examples/demo_compound_paths.cpp $(BUILDFLAGS)
	$(CC) -c $< -o $@ $(CCFLAGS)

demo_coregionalization_toy: demo_coregionalization_toy.o CMultioutput.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

demo_coregionalization_toy.o: examples/demo_coregionalization_toy.cpp $(BUILDFLAGS)
	$(CC) -c $< -o $@ $(CCFLAGS)

# CIP-0017 Phase 2: Bayesian GP-LVM demo.
demo_bayes_gplvm: demo_bayes_gplvm.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

demo_bayes_gplvm.o: examples/demo_bayes_gplvm.cpp $(BUILDFLAGS)
	$(CC) -c $< -o $@ $(CCFLAGS)

gplvm: gplvm.o CClctrl.o CGplvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0017: Bayesian GP-LVM CLI.
bgplvm: bgplvm.o CClctrl.o CBayesGplvm.o CSviBayesGplvm.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0035: deep GP CLI.
deepgp: deepgp.o CClctrl.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


ivm: ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0023 dense PPA + CIP-0026 sparse (CSparsePpa / CGp) learn path.
ppa: ppa.o CClctrl.o CPpa.o CSparsePpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


gp: gp.o CJson.o CInterchange.o CHmc.o CClctrl.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

# CIP-001D / CIP-0035: GPLVM façades + DeepGP (SVGP/Adam) in libgp.
LIBGP_OBJS = CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CMultioutput.o CDist.o ndlassert.o CMltools.o CGplvm.o CBayesGplvm.o CMrd.o CSvgp.o CAdam.o CStochasticOptimisable.o CDeepGp.o CHmc.o CVariationalTarget.o CJson.o CInterchange.o

libgp$(LIBSEXT): $(LIBGP_OBJS)
	$(LIBSCOMMAND) libgp$(LIBSEXT) $(LIBGP_OBJS)

libgp$(LIBDEXT): $(LIBGP_OBJS)
	$(LIBDCOMMAND) libgp$(LIBDEXT) $(LIBGP_OBJS) $(LDFLAGS)



# Full historical suite still needs the MATLAB interface (_NDLMATLAB) for object I/O
# (kernels/noise/models as structs). Numeric fixtures can use libmatio instead:
#   make USE_MATIO=1 matio-tests
tests: testDist testGp testIvm testKern testMatrix testMltools testNdlutil testNoise testNoiseBatched testNoiseInLikelihood testNoiseBlock testRobustMax testSoftmax testSkewNoise testLikelihoodZoo testCip0027Perf testIvmSoftener testEp testEpBlock testTransform testOptimisable testParamPaths testCGp testCoregionalize testPsi testWnHankel testBayesGplvm testMrd testCholAdjoint testLQAdjoint testClctrl testPpa testSparsePpa testSvgp testSvgpBlock testSviBayesGplvm testDeepGp testDiagGradParams testInducingGradZ testConvKern testInducingPatches testConvWeightSpace testInducingHermite testNoisePrior testHmc testVariationalHmc testInterchange

# Reference tests runnable without MATLAB (CIP-0001 / REQ-0002): numeric .mat via matio.
# testOptimisable needs no .mat fixtures but is part of the hard gate (CIP-0010).
# testParamPaths is CIP-0014 Phase A (no .mat). testCGp is CIP-0011.
# testCoregionalize is CIP-000D (no .mat). testPsi / testBayesGplvm are CIP-0017.
# testMrd is CIP-0018.
# testWnHankel is CIP-001C Phase 3 (no .mat).
# is a fixture-free smoke test. testIvm / testMltools read numeric reference
# matrices through matio (testIvm<Type>Active.mat); with _NDLMATLAB they instead load
# the full MATLAB model structs.
# testCholAdjoint is CIP-0012 (no .mat); must run under coverage so CCholAdjoint.cpp is hit.
# testClctrl is CIP-0022 (CSV/SVML data I/O; no .mat).
# testNoiseBatched / testNoiseInLikelihood / testCip0027Perf / testIvmSoftener are CIP-0027 (no .mat).
matio-tests: testDist testKern testMatrix testMltools testIvm testNdlutil testNoise testNoiseBatched testNoiseInLikelihood testNoiseBlock testRobustMax testSoftmax testSkewNoise testLikelihoodZoo testCip0027Perf testIvmSoftener testEp testEpBlock testTransform testGp testOptimisable testParamPaths testCGp testCholAdjoint testLQAdjoint testCoregionalize testPsi testWnHankel testBayesGplvm testMrd testClctrl testDiagGradParams testInducingGradZ testConvKern testInducingPatches testConvWeightSpace testInducingHermite testNoisePrior testHmc testVariationalHmc testInterchange

# Run matio reference binaries (CIP-0004 coverage workflow uses this after a COVERAGE=1 build).
run-matio-tests: matio-tests
	./testNdlutil
	./testTransform
	./testDist
	./testNoise
	./testNoiseBatched
	./testNoiseInLikelihood
	./testNoiseBlock
	./testRobustMax
	./testSoftmax
	./testSkewNoise
	./testLikelihoodZoo
	./testCip0027Perf
	./testIvmSoftener
	./testEp
	./testEpBlock
	./testKern
	./testDiagGradParams
	./testInducingGradZ
	./testConvKern
	./testInducingPatches
	./testConvWeightSpace
	./testInducingHermite
	./testNoisePrior
	./testHmc
	./testVariationalHmc
	./testMatrix
	./testMltools
	./testIvm
	./testOptimisable
	./testParamPaths
	./testInterchange
	./testGp
	./testCGp
	./testCholAdjoint
	./testLQAdjoint
	./testCoregionalize
	./testClctrl
	./testPsi
	./testWnHankel
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
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


run-numerical-stability: testNumericalStability
	./testNumericalStability

# CIP-0012 dual-path: StableFactor vs Legacy under the numerical-stability suite.
compare-sparse-modes: testNumericalStability
	python3 numerical_stability/compare_sparse_modes.py --skip-build


testDist: testDist.o CMatrix.o ndlfortran.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS) 


testGp: testGp.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# Maintainer diagnostics / fixture stampers live under tools/ (binaries still at repo root).
# DTC/DTCVAR gradient + GPy-aligned reference dump (backlog 2026-11-15_dtcvar-numerical-diagnostic).
diagDtcvar: tools/diagDtcvar.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0012: Legacy vs StableFactor γ×M stress sweep (pair with scripts/sparse_mp_reference.py).
diagSparseStable: tools/diagSparseStable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-001C pilot: white-noise inducing vs point inducing (hand closed-form sweep).
diagWhiteNoisePilot: tools/diagWhiteNoisePilot.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-001C pilot: full SCG — point DTCVAR (CGp) vs VIK white-noise features.
diagVikCompare: tools/diagVikCompare.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


diagPitc: tools/diagPitc.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


genTestGpPitc: tools/genTestGpPitc.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# Fixture generator for matfiles/testGpdtcvar.mat (scripts/gen_testGpdtcvar.py).
genTestGpDtcvar: tools/genTestGpDtcvar.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

 
testIvm: testIvm.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0023: PPA vs Octave Phase 0 fixtures (no matio required).
testPpa: testPpa.o CPpa.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0026 prototype: CSparsePpa on CGp (DTC M=N vs dense). Not a release gate yet.
testSparsePpa: testSparsePpa.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)



# CIP-002A: SVGP q(u) natural gradients, ELBO grads, Adam (Gaussian).
testSvgp: testSvgp.o CSvgp.o CAdam.o CStochasticOptimisable.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testSvgpBlock: testSvgpBlock.o CSvgp.o CAdam.o CStochasticOptimisable.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0030 Phase 1: SVI Bayesian GP-LVM (sampled q(X) on CSvgp).
testSviBayesGplvm: testSviBayesGplvm.o CSviBayesGplvm.o CSvgp.o CAdam.o CStochasticOptimisable.o CBayesGplvm.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)




# CIP-0035 Phase 1: deep GP (DSVI on CSvgp layers).
testDeepGp: testDeepGp.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)





testKern: testKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS) 


# CIP-002D follow-up: bulk getDiagGradParams (regularise-once + O(1)-in-N allocs).
testDiagGradParams: testDiagGradParams.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-002D follow-up: batched inducingGradZ + MLP bulk compute.
# CIP-0040 Phase 1a: convolutional covariance function.
testInducingPatches: testInducingPatches.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)
testConvWeightSpace: testConvWeightSpace.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

testConvKern: testConvKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

testInducingGradZ: testInducingGradZ.o CInducing.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-003D phase 1: Hermite inducing layout, capacity and audit.
testInducingHermite: testInducingHermite.o CDeepGp.o CSvgp.o CAdam.o CStochasticOptimisable.o CSparsePpa.o CPpa.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0017 Phase 1: psi statistics (MC + finite differences).
testPsi: testPsi.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-001C Phase 3: Hankel / spectralDensity white-noise K_fu gate.
testWnHankel: testWnHankel.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0017 Phase 2: Bayesian GP-LVM.

testBayesGplvm: testBayesGplvm.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0018: Manifold Relevance Determination.

testMrd: testMrd.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


demo_mrd: examples/demo_mrd.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testCoregionalize: testCoregionalize.o CMultioutput.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)



testMatrix: testMatrix.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS) 

testMatrix.o: testMatrix.cpp
	$(CC) -c $< -o $@ $(CCFLAGS)

# CIP-0022: CSV / SVML data loaders.
testClctrl: testClctrl.o CClctrl.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0012: Murray Cholesky / trsm adjoint primitives.


testCholAdjoint: testCholAdjoint.o CCholAdjoint.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testLQAdjoint: testLQAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testOptimisable: testOptimisable.o COptimisable.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0014 Phase A: GPy-like parameter paths (no matio fixtures).
testParamPaths: testParamPaths.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testCGp: testCGp.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testInterchange: testInterchange.o $(LIBGP_OBJS)
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)




testNoisePrior: testNoisePrior.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-003B: HMC verification (analytic Gaussian, energy error, GP grid check).
testHmc: testHmc.o CHmc.o CGplvm.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

testVariationalHmc: testVariationalHmc.o CHmc.o CVariationalTarget.o CPpa.o CMrd.o CBayesGplvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)





# CIP-0028: FTC vs DTCVAR ridge free/fixed centres (tighter bound demo).
diagWeightSpaceRidge: tools/diagWeightSpaceRidge.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0027 Phase B′: IVM White softener composition matrix (unitsquare).
diagIvmSoftener: tools/diagIvmSoftener.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testMltools: testMltools.o CMltools.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testNdlutil: testNdlutil.o ndlutil.o ndlstrutil.o CMatrix.o ndlfortran.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testNoise: testNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0027 Phase 0: batched Gaussian helpers (no .mat).
testNoiseBatched: testNoiseBatched.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-003A Phase 0/1: Mills helpers + skew likelihoods (no .mat).
testSkewNoise: testSkewNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-003A Phase 2–5: likelihood zoo dens/VE + generalised Laplace on CGp.
testLikelihoodZoo: testLikelihoodZoo.o CLaplaceInference.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0027 Phase A: likelihood.variance on FTC Gram + path aliases.
testEpBlock: testEpBlock.o CEpInference.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testNoiseBlock: testNoiseBlock.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testSoftmax: testSoftmax.o CEpInference.o CLaplaceInference.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testRobustMax: testRobustMax.o CEpInference.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


testNoiseInLikelihood: testNoiseInLikelihood.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0027 Phase C: dual-noise harden + FTC/sparse/IVM algorithm performance gates.
testCip0027Perf: testCip0027Perf.o CGp.o CIvm.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0027: deeper IVM White softener gates (long budget, getWhite, stream).
testIvmSoftener: testIvmSoftener.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)


# CIP-0033 Phase A: GPy-mirrored Bernoulli–probit ep_moments + dense EP.
testEp: testEp.o CEpInference.o CLaplaceInference.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)




testTransform: testTransform.o CMatrix.o ndlfortran.o CTransform.o ndlutil.o ndlstrutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS) 


















# Collected FORTRAN utilities.
ndlfortran.o: ndlfortran.f $(BUILDFLAGS)
	$(FC) -c ndlfortran.f -o ndlfortran.o $(FCFLAGS)


clean:
	rm -f $(BUILDFLAGS) *.o *.d *.gcda *.gcno *.gcov coverage.info
	rm -f tools/*.o tools/*.d tools/*.gcda tools/*.gcno
	rm -f examples/*.o examples/*.d bench/*.o bench/*.d numerical_stability/*.o numerical_stability/*.d
	rm -f $(ALL_BINARIES) libgp$(LIBSEXT) libgp$(LIBDEXT)
	rm -rf coverage-html

diagLqVsStable: tools/diagLqVsStable.o CGp.o CInducing.o CCholAdjoint.o CLQAdjoint.o CMatrix.o ndlfortran.o CNoise.o CEpInference.o CLaplaceInference.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o $@ $^ $(LDFLAGS)

