/* CIP-0015 Stage 1: thin API demo parallel to docs/tutorials/sparse_regression_1d.md */
#include <iostream>
#include "CClctrl.h"
#include "CGp.h"
#include "CKern.h"
#include "CNoise.h"

using namespace std;

namespace {
class DemoCtrl : public CClctrl {
 public:
  DemoCtrl(int c, char** v) : CClctrl(c, v) {}
  void helpInfo() {}
  void helpHeader() {}
};
const unsigned int M_ACTIVE = 20;
}  // namespace

int main(int argc, char** argv)
{
  DemoCtrl ctrl(argc, argv);
  ctrl.setVerbosity(1);
  ctrl.setSeed(1);

  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/spgp1d.csv");

  // CIP-0027: RBF only; sparse observation noise in β (no auto Bias/White).
  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.setByPath("rbf.lengthscale", 1.0);

  CGaussianNoise noise(&y);
  noise.setBias(0.0);

  CGp model(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, ctrl.getVerbosity());
  model.setInducingFixed(true);
  model.setBetaVal(1.0);
  model.optimise(25);

  cout << "demo_sparse_dtcvar: N=" << X.getRows()
       << " M=" << M_ACTIVE
       << " objective=" << model.computeObjectiveVal()
       << " rbf.variance=" << kern.getByPath("rbf.variance")
       << endl;
  return 0;
}
