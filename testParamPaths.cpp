/* CIP-0014 Phase A: GPy-like parameter path façade tests (no matio). */
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "CKern.h"
#include "ndlexceptions.h"

using namespace std;

namespace {

int failures = 0;

void expect(bool cond, const string& label)
{
  if(cond)
    cout << "PASS: " << label << endl;
  else
  {
    cout << "FAIL: " << label << endl;
    failures++;
  }
}

void expectNear(double got, double want, double tol, const string& label)
{
  if(std::fabs(got-want)<=tol)
    cout << "PASS: " << label << endl;
  else
  {
    cout << "FAIL: " << label << " (got " << got << ", want " << want << ")" << endl;
    failures++;
  }
}

} // namespace

int main()
{
  const double tol=1e-12;

  // --- Atomic RBF: lengthscale conversion ---
  {
    CRbfKern rbf(2);
    rbf.setByPath("lengthscale", 2.0);
    expectNear(rbf.getParam(0), 0.25, tol, "RBF set lengthscale=2 stores inverseWidth=1/4");
    expectNear(rbf.getByPath("lengthscale"), 2.0, tol, "RBF get lengthscale recovers 2");
    rbf.setByPath("inv_lengthscale", 4.0);
    expectNear(rbf.getParam(0), 4.0, tol, "RBF inv_lengthscale sets inverseWidth directly");
    expectNear(rbf.getByPath("lengthscale"), 0.5, tol, "RBF lengthscale after inv_lengthscale=4");
    rbf.setByPath("inverseWidth", 9.0);
    expectNear(rbf.getParam(0), 9.0, tol, "RBF legacy inverseWidth alias");
    rbf.setByPath("variance", 1.5);
    expectNear(rbf.getByPath("variance"), 1.5, tol, "RBF variance path");

    vector<string> paths=rbf.paramPaths();
    expect(paths.size()==2 && paths[0]=="lengthscale" && paths[1]=="variance",
           "RBF paramPaths are GPy names");
  }

  // --- Exp lengthscale ---
  {
    CExpKern expk(1);
    expk.setByPath("lengthscale", 5.0);
    expectNear(expk.getParam(0), 0.2, tol, "Exp lengthscale=5 stores inverseWidth=1/5");
    expectNear(expk.getByPath("lengthscale"), 5.0, tol, "Exp get lengthscale");
  }

  // --- Matern spelling ---
  {
    CMatern32Kern m32(1);
    m32.setByPath("lengthscale", 1.25);
    expectNear(m32.getParam(0), 1.25, tol, "Mat32 lengthscale is identity on lengthScale");
    expect(m32.gpyComponentName()=="Mat32", "Mat32 gpyComponentName");
  }

  // --- Compound rbf+white + duplicate rbf ---
  {
    CCmpndKern cmpnd(1);
    CRbfKern rbf(1);
    CWhiteKern white(1);
    CRbfKern rbf2(1);
    cmpnd.addKern(&rbf);
    cmpnd.addKern(&white);
    cmpnd.addKern(&rbf2);

    vector<string> paths=cmpnd.paramPaths();
    expect(paths.size()==5, "cmpnd rbf+white+rbf has 5 paths");
    expect(paths[0]=="rbf.lengthscale" && paths[1]=="rbf.variance", "first rbf paths");
    expect(paths[2]=="white.variance", "white path");
    expect(paths[3]=="rbf_1.lengthscale" && paths[4]=="rbf_1.variance", "second rbf is rbf_1");

    cmpnd.setByPath("rbf.lengthscale", 2.0);
    cmpnd.setByPath("rbf.variance", 3.0);
    cmpnd.setByPath("white.variance", 1e-4);
    cmpnd.setByPath("rbf_1.lengthscale", 4.0);
    expectNear(cmpnd.getByPath("rbf.lengthscale"), 2.0, tol, "cmpnd get rbf.lengthscale");
    expectNear(cmpnd.getByPath("rbf_1.lengthscale"), 4.0, tol, "cmpnd get rbf_1.lengthscale");
    expectNear(cmpnd.getParam(0), 0.25, tol, "cmpnd flat index 0 is first inverseWidth");
    expect(cmpnd.getParamName(0)=="rbf.lengthscale", "dotted getParamName");

    bool threw=false;
    try { cmpnd.paramIndex("nope.variance"); }
    catch(ndlexceptions::Error&) { threw=true; }
    expect(threw, "unknown path throws");
  }

  // --- constrainPositive / None ---
  {
    CRbfKern rbf(1);
    const unsigned int nTransBefore=rbf.getNumTransforms();
    expect(nTransBefore>=1, "RBF starts with transforms");
    rbf.constrainNone("variance");
    // variance index 1 should have no transform; lengthscale/inverseWidth still does
    bool varHas=false;
    for(unsigned int i=0; i<rbf.getNumTransforms(); i++)
      if(rbf.getTransformIndex(i)==1) varHas=true;
    expect(!varHas, "constrainNone removes variance transform");
    rbf.constrainPositive("variance");
    varHas=false;
    for(unsigned int i=0; i<rbf.getNumTransforms(); i++)
      if(rbf.getTransformIndex(i)==1) varHas=true;
    expect(varHas, "constrainPositive restores variance transform");
  }

  // --- MLP name aliases ---
  {
    CMlpKern mlp(1);
    expect(mlp.paramPaths()[0]=="weight_variance", "mlp weight_variance path");
    mlp.setByPath("weight_variance", 2.0);
    expectNear(mlp.getByPath("weightVariance"), 2.0, tol, "mlp legacy alias get");
  }

  if(failures)
  {
    cout << failures << " failure(s)." << endl;
    return 1;
  }
  cout << "All param path tests passed." << endl;
  return 0;
}
