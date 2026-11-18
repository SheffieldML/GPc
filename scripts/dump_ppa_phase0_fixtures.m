% DUMP_PPA_PHASE0_FIXTURES  Octave/MATLAB Phase 0 dumps for CIP-0021.
%
% Uses Nat King PPA toolbox (lawrennd/ppa) with Sheffield deps on the path.
% Writes ASCII artefacts under testdata/ppa/ for C++ assert-and-verify.
%
% Run from repo root with DYLD_LIBRARY_PATH set if needed for Octave/hdf5:
%   octave-cli --quiet scripts/dump_ppa_phase0_fixtures.m

warning("off", "all");

ppaRoot = "/Users/neil/lawrennd/ppa";
outDir = fullfile(pwd, "testdata", "ppa");
mkdir(outDir);

addpath(fullfile(ppaRoot, "matlab"));
addpath("/Users/neil/lawrennd/kern/matlab");
addpath("/Users/neil/lawrennd/noise/matlab");
addpath("/Users/neil/lawrennd/ndlutil/matlab");
addpath("/Users/neil/lawrennd/optimi/matlab");
addpath("/Users/neil/lawrennd/prior/matlab");
addpath("/Users/neil/lawrennd/mltools/matlab");
addpath("/Users/neil/lawrennd/netlab");

Xfull = load(fullfile(ppaRoot, "data", "rand_data.asc"));
yfull = load(fullfile(ppaRoot, "data", "rand_labels.asc"));

% Small subset for CI-friendly fixtures (full demPpa1 is N=200).
N = 30;
X = Xfull(1:N, :);
y = yfull(1:N, :);

noiseModel = "probit";
kernelType = {"rbf", "bias", "white"};

options = ppaOptions;
options.display = 0;
options.kernDisplay = 0;
options.BDisplay = 0;
options.scalarB = 1;
options.varKern = 0;
options.doMStep = 1;
options.updateBeta = 0;
options.doDebug = 0;
options.rememLoglikeVals = 0;
options.tol = 1e-6;
options.maxOuterIter = 3;
options.kernIters = 10;

model = ppa(X, y, noiseModel, kernelType);
model = ppaInit(model, options);

ll0 = ppaCalculateLogLike(model);
kp0 = kernExtractParam(model.kern);

addpath(fullfile(pwd, "scripts"));
ppa_save_matrix(fullfile(outDir, "phase0_X.txt"), X);
ppa_save_matrix(fullfile(outDir, "phase0_y.txt"), y);
ppa_save_matrix(fullfile(outDir, "phase0_Kstore_init.txt"), model.kern.Kstore);
ppa_save_matrix(fullfile(outDir, "phase0_invKstore_init.txt"), model.kern.invKstore);
ppa_save_matrix(fullfile(outDir, "phase0_f_init.txt"), model.expectations.f);
ppa_save_matrix(fullfile(outDir, "phase0_ff_init.txt"), model.expectations.ff);
ppa_save_matrix(fullfile(outDir, "phase0_fBar_init.txt"), model.expectations.fBar);
ppa_save_matrix(fullfile(outDir, "phase0_fBarfBar_init.txt"), model.expectations.fBarfBar(:, :, 1));
ppa_save_matrix(fullfile(outDir, "phase0_g_init.txt"), model.g);
ppa_save_matrix(fullfile(outDir, "phase0_nu_init.txt"), model.nu);
ppa_save_matrix(fullfile(outDir, "phase0_gamma_init.txt"), model.gamma);
ppa_save_matrix(fullfile(outDir, "phase0_B_init.txt"), model.B);
ppa_save_matrix(fullfile(outDir, "phase0_kernParams_init.txt"), kp0);
fid = fopen(fullfile(outDir, "phase0_loglike_init.txt"), "w");
fprintf(fid, "%.17g\n", ll0);
fclose(fid);

% Short outer EM (M then E). Avoid ppaMStep.m: upstream file is missing
% 'end' tokens and fails Octave parse; with updateBeta=0 / doDebug=0 the
% live body is just ppaUpdateKernel (see lawrennd/ppa matlab/ppaMStep.m).
for it = 1:options.maxOuterIter
  if options.kernIters
    model = ppaUpdateKernel(model, options);
  end
  model = ppaEStep(model, options);
end

ll1 = ppaCalculateLogLike(model);
kp1 = kernExtractParam(model.kern);

ppa_save_matrix(fullfile(outDir, "phase0_Kstore_after.txt"), model.kern.Kstore);
ppa_save_matrix(fullfile(outDir, "phase0_invKstore_after.txt"), model.kern.invKstore);
ppa_save_matrix(fullfile(outDir, "phase0_f_after.txt"), model.expectations.f);
ppa_save_matrix(fullfile(outDir, "phase0_ff_after.txt"), model.expectations.ff);
ppa_save_matrix(fullfile(outDir, "phase0_fBar_after.txt"), model.expectations.fBar);
ppa_save_matrix(fullfile(outDir, "phase0_fBarfBar_after.txt"), model.expectations.fBarfBar(:, :, 1));
ppa_save_matrix(fullfile(outDir, "phase0_g_after.txt"), model.g);
ppa_save_matrix(fullfile(outDir, "phase0_nu_after.txt"), model.nu);
ppa_save_matrix(fullfile(outDir, "phase0_gamma_after.txt"), model.gamma);
ppa_save_matrix(fullfile(outDir, "phase0_B_after.txt"), model.B);
ppa_save_matrix(fullfile(outDir, "phase0_kernParams_after.txt"), kp1);
fid = fopen(fullfile(outDir, "phase0_loglike_after.txt"), "w");
fprintf(fid, "%.17g\n", ll1);
fclose(fid);

fid = fopen(fullfile(outDir, "phase0_meta.txt"), "w");
fprintf(fid, "source=lawrennd/ppa demPpa1 subset\n");
fprintf(fid, "N=%d\n", N);
fprintf(fid, "inputDim=%d\n", size(X, 2));
fprintf(fid, "outputDim=%d\n", size(y, 2));
fprintf(fid, "noise=%s\n", noiseModel);
fprintf(fid, "kernel=rbf+bias+white\n");
fprintf(fid, "scalarB=%d\n", options.scalarB);
fprintf(fid, "varKern=%d\n", options.varKern);
fprintf(fid, "updateBeta=%d\n", options.updateBeta);
fprintf(fid, "maxOuterIter=%d\n", options.maxOuterIter);
fprintf(fid, "kernIters=%d\n", options.kernIters);
fprintf(fid, "loglike_init=%.17g\n", ll0);
fprintf(fid, "loglike_after=%.17g\n", ll1);
fprintf(fid, "octave=%s\n", version);
fclose(fid);

printf("Phase 0 fixtures written to %s\n", outDir);
printf("loglike_init=%.10g loglike_after=%.10g\n", ll0, ll1);
