/* CIP-0013: GPc-side objective+gradient timing (q=2 spherical RBF + white).
 * Prints CSV-ish rows for harness aggregation. GPy compare lives in run_bench.py.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "CGp.h"
#include "CKern.h"
#include "CNoise.h"
#include "ndlexceptions.h"

using namespace std;

namespace {

void fillToy(CMatrix& X, CMatrix& y, unsigned int N, unsigned int q, unsigned seed)
{
  srand(seed);
  X.resize(N, q);
  y.resize(N, 1);
  for(unsigned int n = 0; n < N; n++)
  {
    double s = 0.0;
    for(unsigned int d = 0; d < q; d++)
    {
      double v = (double)rand() / (double)RAND_MAX;
      X.setVal(v, n, d);
      s += v;
    }
    y.setVal(sin(3.0 * s) + 0.05 * ((double)rand() / (double)RAND_MAX), n, 0);
  }
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
  unsigned int N = 1000;
  unsigned int q = 2;
  unsigned int M = 50;
  unsigned int reps = 7;
  unsigned int warm = 2;
  for(int i = 1; i < argc; i++)
  {
    if(!strcmp(argv[i], "--kind") && i + 1 < argc)
      kind = argv[++i];
    else if(!strcmp(argv[i], "--N") && i + 1 < argc)
      N = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--q") && i + 1 < argc)
      q = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--M") && i + 1 < argc)
      M = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--reps") && i + 1 < argc)
      reps = (unsigned)atoi(argv[++i]);
    else if(!strcmp(argv[i], "--help"))
    {
      cout << "Usage: benchGPc [--kind ftc|dtcvar|fitc] [--N N] [--q q] [--M M] [--reps R]\n";
      return 0;
    }
  }

  CMatrix X;
  CMatrix y;
  fillToy(X, y, N, q, 13);

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CWhiteKern(X));
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
  kern.setByPath("white.variance", 0.1);

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
  if(kind != "ftc")
    model->setInducingFixed(true);

  CMatrix g(1, model->getOptNumParams());
  // Warm-up + timed objective+gradient.
  for(unsigned int w = 0; w < warm; w++)
    (void)model->computeObjectiveGradParams(g);

  vector<double> times;
  times.reserve(reps);
  double ll = 0.0;
  for(unsigned int r = 0; r < reps; r++)
  {
    auto t0 = chrono::steady_clock::now();
    ll = -model->computeObjectiveGradParams(g);
    auto t1 = chrono::steady_clock::now();
    times.push_back(chrono::duration<double, milli>(t1 - t0).count());
  }
  double med = median(times);

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

  // One BLAS thread is the protocol; callers should export OPENBLAS_NUM_THREADS=1.
  cout << "kind=" << kind
       << " N=" << N
       << " q=" << q
       << " M=" << (kind == "ftc" ? 0 : M)
       << " nparams=" << model->getOptNumParams()
       << " ll=" << ll
       << " median_ms=" << med
       << " reps=" << reps
       << " flags=" << flags
       << " gpc_checks=" << checks
       << " blas=openblas_or_platform"
       << " threads=see_OPENBLAS_NUM_THREADS"
       << endl;

  delete model;
  return 0;
}
