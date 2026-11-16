/* CIP-0018: tiny two-view MRD smoke (shared + private ARD dims). */
#include <iostream>
#include <vector>
#include "CMrd.h"

using namespace std;

int main()
{
  const unsigned int N = 12;
  const unsigned int Q = 3;
  const unsigned int M = 4;
  CMatrix Y0(N, 2);
  CMatrix Y1(N, 3);
  for(unsigned int n = 0; n < N; n++)
  {
    const double xs = sin(0.6 * n);
    const double xp0 = 0.4 * cos(0.9 * n);
    const double xp1 = 0.5 * sin(1.1 * n + 0.3);
    Y0.setVal(xs + xp0, n, 0);
    Y0.setVal(0.7 * xs + 0.3 * xp0, n, 1);
    Y1.setVal(xs + xp1, n, 0);
    Y1.setVal(0.5 * xs, n, 1);
    Y1.setVal(0.2 * xs + xp1, n, 2);
  }
  vector<CMatrix> Ys;
  Ys.push_back(Y0);
  Ys.push_back(Y1);
  CRbfardKern* k0 = new CRbfardKern(Q);
  k0->setInitParam();
  CRbfardKern* k1 = new CRbfardKern(Q);
  k1->setInitParam();
  vector<CKern*> ks;
  ks.push_back(k0);
  ks.push_back(k1);

  CMrd model(Ys, ks, Q, M, 1);
  cout << "Initial ELBO=" << model.logLikelihood() << endl;
  model.optimise(40);
  model.display(cout);

  vector<unsigned int> shared, p0, p1;
  model.factorizeSpace(shared, p0, p1, 0.05);
  cout << "factorizeSpace: shared";
  for(size_t i = 0; i < shared.size(); i++)
    cout << " " << shared[i];
  cout << " | private0";
  for(size_t i = 0; i < p0.size(); i++)
    cout << " " << p0[i];
  cout << " | private1";
  for(size_t i = 0; i < p1.size(); i++)
    cout << " " << p1[i];
  cout << endl;
  return 0;
}
