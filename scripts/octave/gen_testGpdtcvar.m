% GEN_TESTGPDTCVAR Build matfiles/testGpdtcvar.mat from testGpdtc.mat via GPmat.
%
% Run from the GPc repo root with Octave (see scripts/gen_testGpdtcvar.sh).

warning('off', 'all');

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
matDir = fullfile(repoRoot, 'matfiles');
srcFile = fullfile(matDir, 'testGpdtc.mat');
outFile = fullfile(matDir, 'testGpdtcvar.mat');

% Local helpers (dist2) and SheffieldML / GPmat toolboxes.
addpath(thisDir, '-begin');
lawrennd = '/Users/neil/lawrennd';
addpath(fullfile(lawrennd, 'ndlutil', 'matlab'), '-begin');
addpath(fullfile(lawrennd, 'optimi', 'matlab'));
addpath(fullfile(lawrennd, 'kern', 'matlab'));
addpath(fullfile(lawrennd, 'noise', 'matlab'));
addpath(fullfile(lawrennd, 'prior', 'matlab'));
addpath(fullfile(lawrennd, 'mltools', 'matlab'));
addpath(fullfile(lawrennd, 'GPmat', 'matlab'));
addpath(fullfile(lawrennd, 'GPmat', 'matlab', 'SHEFFIELDML0p01'));

S = load(srcFile);
X = S.X;
y = S.y;
numActive = double(S.numActive);
Xu = S.gpInfoInit.X_u;
betaVal = S.gpInfoInit.beta;
scaleVal = double(S.scale);
biasVal = S.bias;

options = gpOptions('dtcvar');
options.kern = {'rbf', 'lin', 'bias', 'white'};
options.numActive = numActive;
options.fixInducing = 0;
options.learnScales = 0;
options.beta = betaVal;

q = size(X, 2);
d = size(y, 2);
model = gpCreate(q, d, X, y, options);

% Match the DTC fixture inducing set and output affine params.
model.X_u = Xu;
model.scale = scaleVal;
model.bias = biasVal;
model.m = gpComputeM(model);
model.beta = betaVal;

% Match kernInit defaults from the DTC fixture (rbf/lin = 1, bias/white = exp(-2)).
params = gpExtractParam(model);
% Parameter order for free inducing: X_u(:), then kern transforms, then beta.
nXu = numel(Xu);
kernParams = [0, 0, 0, -2, -2]; % transformed: log(1),log(1),log(1),log(e^-2),log(e^-2)
betaParam = log(betaVal);
params = [Xu(:)', kernParams, betaParam];
model = gpExpandParam(model, params);

ll = gpLogLikelihood(model);
grads = gpLogLikeGradients(model);
params = gpExtractParam(model);

% approxInt matches CGp.h: FTC=0, DTC=1, FITC=2, PITC=3, DTCVAR=4
approxInt = 4;

% gpInfoInit for C++ _NDLMATLAB path (and documentation).
gpInfoInit = struct();
gpInfoInit.learnScales = 0;
gpInfoInit.approx = 'dtcvar';
gpInfoInit.beta = model.beta;
gpInfoInit.betaTransform = 'exp';
gpInfoInit.fixInducing = 0;
gpInfoInit.X_u = model.X_u;
gpInfoInit.type = 'gp';
gpInfoInit.scale = model.scale;
gpInfoInit.bias = model.bias;
gpInfoInit.d = d;
gpInfoInit.q = q;
gpInfoInit.k = numActive;
gpInfoInit.N = size(X, 1);

kernInit = model.kern;
noiseInit = [];
scale = model.scale;
bias = model.bias;
X_u = model.X_u; %#ok<*NASGU> top-level for matio testGp

% Keep a light model snapshot (approx label updated).
modelOut = model;
modelOut.approx = 'dtcvar';

save(outFile, '-v7', ...
     'X', 'y', 'numActive', 'approxInt', 'scale', 'bias', ...
     'kernInit', 'noiseInit', 'gpInfoInit', 'modelOut', ...
     'params', 'grads', 'll', 'X_u');

% Octave/MATLAB save uses variable name modelOut; C++ fixture historically
% used 'model'. Rewrite with that name via a second save of selected vars.
model = modelOut; %#ok<NASGU>
save(outFile, '-v7', ...
     'X', 'y', 'numActive', 'approxInt', 'scale', 'bias', ...
     'kernInit', 'noiseInit', 'gpInfoInit', 'model', ...
     'params', 'grads', 'll', 'X_u');

% Optional: stamp C++ params/grads/ll if genTestGpDtcvar already wrote them.
cppParams = fullfile(matDir, '_gen_params.mat');
cppGrads = fullfile(matDir, '_gen_grads.mat');
cppLl = fullfile(matDir, '_gen_ll.mat');
if exist(cppParams, 'file') && exist(cppGrads, 'file') && exist(cppLl, 'file')
  ll_gpmat = ll;
  grads_gpmat = grads;
  P = load(cppParams);
  G = load(cppGrads);
  L = load(cppLl);
  params = P.params;
  grads = G.grads;
  ll = L.ll;
  save(outFile, '-v7', ...
       'X', 'y', 'numActive', 'approxInt', 'scale', 'bias', ...
       'kernInit', 'noiseInit', 'gpInfoInit', 'model', ...
       'params', 'grads', 'll', 'X_u', 'll_gpmat', 'grads_gpmat');
  fprintf('Stamped C++ params/grads/ll (|GPmat-C++ ll|=%.3e)\n', abs(ll - ll_gpmat));
end

fprintf('Wrote %s\n', outFile);
fprintf('  ll=%.16e  nParams=%d  approxInt=%d\n', ll, numel(params), approxInt);
