/* CIP-0013: GPc-side objective+gradient timing (q=2 spherical RBF + white).
 * Prints one key=value line for harness aggregation. Shared data comes from
 * run_bench.py (--data-dir); without it, a portable protocol fill is used.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "CGp.h"
#include "CSvgp.h"
#include "CKern.h"
#include "CNoise.h"
#include "CMatrix.h"
#include "ndlexceptions.h"

using namespace std;

namespace {

// Portable LCG so C++ and Python can regenerate the same toy without files.
unsigned int lcgNext(unsigned int& state)
{
  state = 1664525u * state + 1013904223u;
  return state;
}
double lcgUniform(unsigned int& state)
{
  return (lcgNext(state) & 0x00ffffffu) / double(0x01000000u);
}

void fillProtocol(CMatrix& X, CMatrix& y, CMatrix& Xu,
                  unsigned int N, unsigned int q, unsigned int M, unsigned seed)
{
  unsigned int state = seed;
  X.resize(N, q);
  y.resize(N, 1);
  for(unsigned int n = 0; n < N; n++)
  {
    double s = 0.0;
    for(unsigned int d = 0; d < q; d++)
    {
      // Uniform on [-3, 3].
      double v = -3.0 + 6.0 * lcgUniform(state);
      X.setVal(v, n, d);
      s += sin(v);
    }
    // Observation noise std 0.1 (variance 0.01).
    double eps = 0.1 * (2.0 * lcgUniform(state) - 1.0) * sqrt(3.0); // rough; data-dir preferred
    y.setVal(s + eps, n, 0);
  }
  // Evenly spaced inducing subset (deterministic; run_bench.py overrides via files).
  Xu.resize(M, q);
  for(unsigned int i = 0; i < M; i++)
  {
    unsigned int idx = (M == 1) ? 0 : (i * (N - 1)) / (M - 1);
    for(unsigned int d = 0; d < q; d++)
      Xu.setVal(X.getVal(idx, d), i, d);
  }
}

bool loadTxt(CMatrix& A, const string& path)
{
  ifstream in(path.c_str());
  if(!in)
    return false;
  vector<vector<double> > rows;
  string line;
  while(getline(in, line))
  {
    if(line.empty() || line[0] == '#')
      continue;
    istringstream iss(line);
    vector<double> r;
    double v;
    while(iss >> v)
      r.push_back(v);
    if(!r.empty())
      rows.push_back(r);
  }
  if(rows.empty())
    return false;
  const unsigned int ncols = (unsigned)rows[0].size();
  A.resize((unsigned)rows.size(), ncols);
  for(unsigned int i = 0; i < rows.size(); i++)
  {
    if(rows[i].size() != ncols)
      return false;
    for(unsigned int j = 0; j < ncols; j++)
      A.setVal(rows[i][j], i, j);
  }
  return true;
}

CGp* makeModel(const string& kind, CMatrix* X, CMatrix* y, CKern* kern, CNoise* noise,
               unsigned int M)
{
  if(kind == "ftc")
    return new CGp(kern, noise, X, CGp::FTC, 0, 0);
  if(kind == "dtcvar")
    return new CGp(kern, noise, X, CGp::DTCVAR, M, 0);
  if(kind == "fitc")
    return new CGp(kern, noise, X, CGp::FITC, M, 0);
  throw ndlexceptions::Error("unknown model kind: " + kind);
}

double median(vector<double>& v)
{
  sort(v.begin(), v.end());
  size_t n = v.size();
  if(n % 2)
    return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv)
{
  string kind = "ftc";
  string kernName = "rbf";  // CIP-002D QW8: rbf (default) or mlp
  string dataDir;
  unsigned int N = 1000;
  unsigned int q = 2;
  unsigned int M = 50;
  unsigned int reps = 7;
  unsigned int warm = 2;
  bool inducingFixed = false;
  for(int i = 1; i < argc; i++)
  {
    if(!strcmp(argv[i], "--kind") && i + 1 < argc)
      kind = argv[++i];
    else if(!strcmp(argv[i], "--kern") && i + 1 < argc)
      kernName = argv[++i];
    else if(!strcmp(argv[i], "--N") && i + 1 < argc)
      N = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--q") && i + 1 < argc)
      q = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--M") && i + 1 < argc)
      M = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--reps") && i + 1 < argc)
      reps = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--data-dir") && i + 1 < argc)
      dataDir = argv[++i];
    else if(!strcmp(argv[i], "--inducing-fixed"))
      inducingFixed = true;
    else if(!strcmp(argv[i], "--help"))
    {
      cout << "Usage: benchGPc [--kind ftc|dtcvar|fitc|svgp] [--kern rbf|mlp]\n"
           << "                [--N N] [--q q] [--M M] [--reps R]\n"
           << "                [--data-dir DIR] [--inducing-fixed]\n"
           << "Checked-build timings: make CHECKS=full bench && ./benchGPc ...\n"
           << "  (gpc_checks=on|off is printed; RELEASE=1 is the competitive path)\n";
      return 0;
    }
  }

  CMatrix X, y, Xu;
  if(!dataDir.empty())
  {
    if(!loadTxt(X, dataDir + "/X.txt") || !loadTxt(y, dataDir + "/y.txt"))
    {
      cerr << "benchGPc: failed to load X.txt/y.txt from " << dataDir << endl;
      return 2;
    }
    N = X.getRows();
    q = X.getCols();
    if(kind != "ftc")
    {
      if(!loadTxt(Xu, dataDir + "/Xu.txt"))
      {
        cerr << "benchGPc: failed to load Xu.txt from " << dataDir << endl;
        return 2;
      }
      M = Xu.getRows();
    }
  }
  else
  {
    fillProtocol(X, y, Xu, N, q, M, 13);
  }

  CCmpndKern kern(X);
  if(kernName == "rbf")
  {
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CWhiteKern(X));
    kern.setByPath("rbf.lengthscale", 1.0);
    kern.setByPath("rbf.variance", 1.0);
    kern.setByPath("white.variance", 0.01);
  }
  else if(kernName == "mlp")
  {
    kern.addKern(new CMlpKern(X));
    kern.addKern(new CWhiteKern(X));
    kern.setByPath("mlp.weightVariance", 1.0);
    kern.setByPath("mlp.biasVariance", 1.0);
    kern.setByPath("mlp.variance", 1.0);
    kern.setByPath("white.variance", 0.01);
  }
  else
  {
    cerr << "benchGPc: unknown --kern " << kernName << " (want rbf|mlp)" << endl;
    return 2;
  }

#if defined(NDEBUG)
  const char* flags = "O2-DNDEBUG";
#else
  const char* flags = "ggdb-DDBG";
#endif
#if GPC_CHECKS
  const char* checks = "on";
#else
  const char* checks = "off";
#endif

  // CIP-002D QW6/QW7: SVGP objective+grad path with allocation count.
  if(kind == "svgp")
  {
    CSvgp sv(&kern, &X, &y, Xu, 100.0, 0);
    sv.naturalGradientStep(1.0);
    CMatrix gKern(1, kern.getNumParams());
    CMatrix gZ(M, q);
    double gBeta = 0.0;
    for(unsigned int w = 0; w < warm; w++)
      (void)sv.elboGrad(gKern, gZ, gBeta);

    vector<double> times;
    times.reserve(reps);
    double ll = 0.0;
    unsigned int allocs = 0;
    for(unsigned int r = 0; r < reps; r++)
    {
      CMatrix::resetTotalAlloc();
      auto t0 = chrono::steady_clock::now();
      ll = sv.elboGrad(gKern, gZ, gBeta);
      auto t1 = chrono::steady_clock::now();
      times.push_back(chrono::duration<double, milli>(t1 - t0).count());
      allocs += CMatrix::getTotalAlloc();
    }
    const double med = median(times);
    cout << "kind=svgp"
         << " kern=" << kernName
         << " N=" << N
         << " q=" << q
         << " M=" << M
         << " nparams=" << sv.getOptNumParams()
         << " ll=" << ll
         << " median_ms=" << med
         << " allocs_per_eval=" << (allocs / reps)
         << " reps=" << reps
         << " flags=" << flags
         << " gpc_checks=" << checks
         << " blas=openblas_or_platform"
         << " threads=see_OPENBLAS_NUM_THREADS"
         << endl;
    return 0;
  }

  CGaussianNoise noise(&y);
  noise.setBias(0.0);

  CGp* model = 0;
  try
  {
    model = makeModel(kind, &X, &y, &kern, &noise, M);
  }
  catch(ndlexceptions::Error& e)
  {
    cerr << "benchGPc: construct failed: " << e.what() << endl;
    return 2;
  }
  model->setVerbosity(0);
  // CIP-0027 Phase C: bench compounds include White as observation noise.
  if(kind == "ftc")
    model->fixLikelihoodVariance(true);
  if(kind != "ftc")
  {
    model->setActiveSet(Xu);
    model->setInducingFixed(inducingFixed);
    model->setBetaVals(100.0);
  }

  CMatrix g(1, model->getOptNumParams());
  for(unsigned int w = 0; w < warm; w++)
    (void)model->computeObjectiveGradParams(g);

  vector<double> times;
  times.reserve(reps);
  double ll = 0.0;
  unsigned int allocs = 0;
  for(unsigned int r = 0; r < reps; r++)
  {
    CMatrix::resetTotalAlloc();
    auto t0 = chrono::steady_clock::now();
    ll = -model->computeObjectiveGradParams(g);
    auto t1 = chrono::steady_clock::now();
    times.push_back(chrono::duration<double, milli>(t1 - t0).count());
    allocs += CMatrix::getTotalAlloc();
  }
  double med = median(times);

  cout << "kind=" << kind
       << " kern=" << kernName
       << " N=" << N
       << " q=" << q
       << " M=" << (kind == "ftc" ? 0 : M)
       << " nparams=" << model->getOptNumParams()
       << " ll=" << ll
       << " median_ms=" << med
       << " allocs_per_eval=" << (allocs / reps)
       << " reps=" << reps
       << " flags=" << flags
       << " gpc_checks=" << checks
       << " blas=openblas_or_platform"
       << " threads=see_OPENBLAS_NUM_THREADS"
       << endl;

  delete model;
  return 0;
}
