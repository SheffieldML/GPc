/* CIP-0015 Stage 1: thin API demo parallel to docs/tutorials/regression_1d.md */
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
}  // namespace

int main(int argc, char** argv)
{
  DemoCtrl ctrl(argc, argv);
  ctrl.setVerbosity(1);
  ctrl.setSeed(1);

  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/sinc.csv");

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CBiasKern(X));
  kern.addKern(new CWhiteKern(X));
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);

  CGaussianNoise noise(&y);
  noise.setBias(0.0);

  CGp model(&kern, &noise, &X, CGp::FTC, 0, ctrl.getVerbosity());
  model.setBetaVal(1.0);
  model.optimise(25);

  cout << "demo_regression_1d: N=" << X.getRows()
       << " objective=" << model.computeObjectiveVal()
       << " rbf.variance=" << kern.getByPath("rbf.variance")
       << endl;
  return 0;
}
