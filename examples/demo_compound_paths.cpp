/* CIP-0015 Phase B: compound kernels + CIP-0014 paths (basic_kernels / models_basic). */
#include <iostream>
#include <vector>
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

  // Compound spatial prior: RBF + linear (Bias/White stay opt-in; noise on likelihood).
  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CLinKern(X));

  cout << "demo_compound_paths: paramPaths" << endl;
  vector<string> paths = kern.paramPaths();
  for(size_t i = 0; i < paths.size(); i++)
    cout << "  " << paths[i] << endl;

  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
  kern.setByPath("linear.variance", 0.1);
  kern.constrainPositive("rbf.variance");
  kern.constrainPositive("linear.variance");

  CGaussianNoise noise(&y);
  noise.setBias(0.0);

  CGp model(&kern, &noise, &X, CGp::FTC, 0, ctrl.getVerbosity());
  model.setLikelihoodVariance(0.1);
  model.optimise(25);

  cout << "demo_compound_paths: N=" << X.getRows()
       << " objective=" << model.computeObjectiveVal()
       << " rbf.lengthscale=" << kern.getByPath("rbf.lengthscale")
       << " rbf.variance=" << kern.getByPath("rbf.variance")
       << " linear.variance=" << kern.getByPath("linear.variance")
       << endl;
  return 0;
}
