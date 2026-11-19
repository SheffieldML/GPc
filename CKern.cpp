#include "CKern.h"
#include <cmath>
#include <vector>
#ifdef GPC_OPENMP
#include <omp.h>
#endif

using namespace std;

// CIP-0025: minimum input dimension to prefer the GEMM distance path over
// the scalar dist2Active loop in fillSymWorkspace / fillRectWorkspace.
// Measured knee: for q < 8 the dgemm overhead exceeds the per-element gain on
// the standard bench (n=100, q=2 protocol from REQ-0007 / CIP-0013).
static const unsigned int kGemmDistMinQ = 8;

namespace {

enum LeafConv {
  LEAF_NATIVE = 0,
  LEAF_RBF_LENGTHSCALE = 1,
  LEAF_EXP_LENGTHSCALE = 2
};

double toGpyValue(double native, LeafConv conv)
{
  switch(conv)
  {
  case LEAF_RBF_LENGTHSCALE:
    if(!(native>0.0))
      throw ndlexceptions::Error("inverseWidth must be positive to convert to lengthscale.");
    return 1.0/std::sqrt(native);
  case LEAF_EXP_LENGTHSCALE:
    if(!(native>0.0))
      throw ndlexceptions::Error("inverseWidth must be positive to convert to lengthscale.");
    return 1.0/native;
  case LEAF_NATIVE:
  default:
    return native;
  }
}

double fromGpyValue(double gpy, LeafConv conv)
{
  switch(conv)
  {
  case LEAF_RBF_LENGTHSCALE:
    if(!(gpy>0.0))
      throw ndlexceptions::Error("lengthscale must be positive.");
    return 1.0/(gpy*gpy);
  case LEAF_EXP_LENGTHSCALE:
    if(!(gpy>0.0))
      throw ndlexceptions::Error("lengthscale must be positive.");
    return 1.0/gpy;
  case LEAF_NATIVE:
  default:
    return gpy;
  }
}

bool typesCompatible(const string& type, const string& name)
{
  if(name==type)
    return true;
  if(type=="matern32" && (name=="Mat32" || name=="matern32"))
    return true;
  if(type=="matern52" && (name=="Mat52" || name=="matern52"))
    return true;
  if(type=="exp" && (name=="Exponential" || name=="exp"))
    return true;
  if((type=="ratquad" || type=="ratquadard") && (name=="RatQuad" || name=="ratquad"))
    return true;
  if((type=="lin" || type=="linard") && (name=="linear" || name=="lin"))
    return true;
  if(type=="stdperiodic" && (name=="std_periodic" || name=="stdperiodic"))
    return true;
  if(type=="cosine" && (name=="Cosine" || name=="cosine"))
    return true;
  if(type=="brownian" && (name=="Brownian" || name=="brownian"))
    return true;
  if((type=="ibrownian" || type=="integratedbrownian")
     && (name=="IntegratedBrownian" || name=="ibrownian" || name=="integratedbrownian"))
    return true;
  if((type=="rbf" || type=="rbfard") && name=="rbf")
    return true;
  if(type=="eq" && (name=="ExpQuad" || name=="eq" || name=="expquad"))
    return true;
  if((type=="mlp" || type=="mlpard") && name=="mlp")
    return true;
  if((type=="arccos" || type=="arccosine") && (name=="ArcCosine" || name=="arccos" || name=="arccosine"))
    return true;
  if(type=="poly" && name=="poly")
    return true;
  if(type=="white" && name=="white")
    return true;
  if(type=="bias" && name=="bias")
    return true;
  if(type=="coregionalize" && name=="coregionalize")
    return true;
  return false;
}

} // namespace

string CKern::gpyComponentName() const
{
  const string t = getType();
  if(t=="matern32") return "Mat32";
  if(t=="matern52") return "Mat52";
  if(t=="exp") return "Exponential";
  if(t=="ratquad" || t=="ratquadard") return "RatQuad";
  if(t=="lin" || t=="linard") return "linear";
  if(t=="stdperiodic") return "std_periodic";
  if(t=="cosine") return "Cosine";
  if(t=="brownian") return "Brownian";
  if(t=="ibrownian" || t=="integratedbrownian") return "IntegratedBrownian";
  if(t=="rbfard") return "rbf";
  if(t=="eq") return "ExpQuad";
  if(t=="mlpard") return "mlp";
  if(t=="arccos" || t=="arccosine") return "ArcCosine";
  return t;
}

namespace {

string canonicalLeafName(const CKern& kern, unsigned int index)
{
  BOUNDCHECK(index<kern.getNumParams());
  const string native = kern.getParamName(index);
  const string t = kern.getType();
  if(native=="inverseWidth")
    return "lengthscale";
  if(native=="lengthScale")
    return "lengthscale";
  if(native=="weightVariance")
    return "weight_variance";
  if(native=="biasVariance")
    return "bias_variance";
  if(native=="alpha" && (t=="ratquad" || t=="ratquadard"))
    return "power";
  if(native=="inputScale")
  {
    unsigned int k=0;
    for(unsigned int j=0; j<index; j++)
    {
      if(kern.getParamName(j)=="inputScale")
        k++;
    }
    return string("inputScale_")+ndlstrutil::itoa(k);
  }
  // Disambiguate duplicate identical native names (besides inputScale).
  unsigned int prior=0;
  for(unsigned int j=0; j<index; j++)
  {
    if(kern.getParamName(j)==native)
      prior++;
  }
  if(prior>0)
    return native+string("_")+ndlstrutil::itoa(prior);
  return native;
}

bool matchAtomicLeaf(const CKern& kern, unsigned int index, const string& leaf, LeafConv& conv)
{
  const string native = kern.getParamName(index);
  const string canon = canonicalLeafName(kern, index);
  const string t = kern.getType();

  if(leaf=="inv_lengthscale" && native=="inverseWidth")
  {
    conv = LEAF_NATIVE;
    return true;
  }
  if(leaf=="lengthscale" && native=="inverseWidth")
  {
    if(t=="exp")
      conv = LEAF_EXP_LENGTHSCALE;
    else
      conv = LEAF_RBF_LENGTHSCALE; // rbf / rbfard shared width
    return true;
  }
  if(leaf==canon || leaf==native)
  {
    conv = LEAF_NATIVE;
    return true;
  }
  if(native=="lengthScale" && leaf=="lengthscale")
  {
    conv = LEAF_NATIVE;
    return true;
  }
  if(native=="weightVariance" && leaf=="weight_variance")
  {
    conv = LEAF_NATIVE;
    return true;
  }
  if(native=="biasVariance" && leaf=="bias_variance")
  {
    conv = LEAF_NATIVE;
    return true;
  }
  if(native=="alpha" && leaf=="power" && (t=="ratquad" || t=="ratquadard"))
  {
    conv = LEAF_NATIVE;
    return true;
  }
  return false;
}

void resolveAtomicPath(const CKern& kern, const string& path, unsigned int& index, LeafConv& conv)
{
  if(path.find('.')!=string::npos)
    throw ndlexceptions::Error("Path '"+path+"' has a component prefix but kernel '"+kern.getType()+"' is not compound.");
  unsigned int matches=0;
  unsigned int found=0;
  LeafConv foundConv=LEAF_NATIVE;
  for(unsigned int i=0; i<kern.getNumParams(); i++)
  {
    LeafConv c=LEAF_NATIVE;
    if(matchAtomicLeaf(kern, i, path, c))
    {
      matches++;
      found=i;
      foundConv=c;
    }
  }
  if(matches==0)
    throw ndlexceptions::Error("Unknown parameter path '"+path+"' for kernel '"+kern.getType()+"'.");
  if(matches>1)
    throw ndlexceptions::Error("Ambiguous parameter path '"+path+"' for kernel '"+kern.getType()+"'.");
  index=found;
  conv=foundConv;
}

} // namespace

vector<string> CKern::paramPaths() const
{
  vector<string> paths;
  paths.reserve(getNumParams());
  for(unsigned int i=0; i<getNumParams(); i++)
    paths.push_back(canonicalLeafName(*this, i));
  return paths;
}

unsigned int CKern::paramIndex(const string& path) const
{
  unsigned int index=0;
  LeafConv conv=LEAF_NATIVE;
  resolveAtomicPath(*this, path, index, conv);
  return index;
}

double CKern::getByPath(const string& path) const
{
  unsigned int index=0;
  LeafConv conv=LEAF_NATIVE;
  resolveAtomicPath(*this, path, index, conv);
  return toGpyValue(getParam(index), conv);
}

void CKern::setByPath(const string& path, double val)
{
  unsigned int index=0;
  LeafConv conv=LEAF_NATIVE;
  resolveAtomicPath(*this, path, index, conv);
  setParam(fromGpyValue(val, conv), index);
}

void CKern::constrainPositive(const string& path)
{
  replaceTransform(CTransform::defaultPositive(), paramIndex(path));
}

void CKern::constrainZeroOne(const string& path)
{
  replaceTransform(CTransform::defaultZeroOne(), paramIndex(path));
}

void CKern::constrainNone(const string& path)
{
  replaceTransform(0, paramIndex(path));
}

string CComponentKern::componentPathLabel(unsigned int componentIndex) const
{
  BOUNDCHECK(componentIndex<components.size());
  const string base = components[componentIndex]->gpyComponentName();
  unsigned int prior=0;
  for(unsigned int j=0; j<componentIndex; j++)
  {
    if(components[j]->gpyComponentName()==base)
      prior++;
  }
  if(prior==0)
    return base;
  return base+string("_")+ndlstrutil::itoa(prior);
}

void CComponentKern::syncTransformsFromComponents()
{
  unlinkTransforms();
  unsigned int offset=0;
  for(size_t i=0; i<components.size(); i++)
  {
    for(unsigned int j=0; j<components[i]->getNumTransforms(); j++)
      addTransform(components[i]->getTransform(j), components[i]->getTransformIndex(j)+offset);
    offset+=components[i]->getNumParams();
  }
}

string CComponentKern::getParamName(unsigned int paramNo) const
{
  unsigned int start=0;
  unsigned int end=0;
  for(size_t i=0; i<components.size(); i++)
  {
    end=start+components[i]->getNumParams()-1;
    if(paramNo<=end)
    {
      const unsigned int local=paramNo-start;
      return componentPathLabel(static_cast<unsigned int>(i))+string(".")+canonicalLeafName(*components[i], local);
    }
    start=end+1;
  }
  return "";
}

vector<string> CComponentKern::paramPaths() const
{
  vector<string> paths;
  for(size_t i=0; i<components.size(); i++)
  {
    const string label=componentPathLabel(static_cast<unsigned int>(i));
    vector<string> childPaths=components[i]->paramPaths();
    for(size_t j=0; j<childPaths.size(); j++)
      paths.push_back(label+string(".")+childPaths[j]);
  }
  return paths;
}

namespace {

void stripCompoundPrefix(string& path)
{
  if(path.compare(0, 4, "sum.")==0)
    path=path.substr(4);
  else if(path.compare(0, 6, "cmpnd.")==0)
    path=path.substr(6);
  else if(path.compare(0, 7, "tensor.")==0)
    path=path.substr(7);
}

} // namespace

unsigned int CComponentKern::findComponent(const string& comp) const
{
  unsigned int matches=0;
  unsigned int found=0;
  for(size_t i=0; i<components.size(); i++)
  {
    const string label=componentPathLabel(static_cast<unsigned int>(i));
    bool hit=(comp==label);
    if(!hit && typesCompatible(components[i]->getType(), comp))
    {
      unsigned int prior=0;
      for(size_t j=0; j<i; j++)
        if(components[j]->gpyComponentName()==components[i]->gpyComponentName())
          prior++;
      // Bare type/GPy name matches only the first instance; use rbf_1 for later ones.
      if(prior==0)
        hit=true;
    }
    if(hit)
    {
      matches++;
      found=static_cast<unsigned int>(i);
    }
  }
  if(matches==0)
    throw ndlexceptions::Error("Unknown component '"+comp+"'.");
  if(matches>1)
    throw ndlexceptions::Error("Ambiguous component '"+comp+"'.");
  return found;
}

void CComponentKern::splitComponentPath(const string& path, unsigned int& componentIndex, string& leaf) const
{
  string p=path;
  stripCompoundPrefix(p);
  const size_t dot=p.find('.');
  if(dot==string::npos || dot+1>=p.size())
    throw ndlexceptions::Error("Compound kernel path '"+path+"' must be component.leaf.");
  componentIndex=findComponent(p.substr(0, dot));
  leaf=p.substr(dot+1);
}

unsigned int CComponentKern::paramIndex(const string& path) const
{
  unsigned int c=0;
  string leaf;
  splitComponentPath(path, c, leaf);
  unsigned int offset=0;
  for(unsigned int i=0; i<c; i++)
    offset+=components[i]->getNumParams();
  return offset+components[c]->paramIndex(leaf);
}

double CComponentKern::getByPath(const string& path) const
{
  unsigned int c=0;
  string leaf;
  splitComponentPath(path, c, leaf);
  return components[c]->getByPath(leaf);
}

void CComponentKern::setByPath(const string& path, double val)
{
  unsigned int c=0;
  string leaf;
  splitComponentPath(path, c, leaf);
  components[c]->setByPath(leaf, val);
}

void CComponentKern::constrainPositive(const string& path)
{
  unsigned int c=0;
  string leaf;
  splitComponentPath(path, c, leaf);
  unlinkTransforms();
  components[c]->constrainPositive(leaf);
  syncTransformsFromComponents();
}

void CComponentKern::constrainZeroOne(const string& path)
{
  unsigned int c=0;
  string leaf;
  splitComponentPath(path, c, leaf);
  unlinkTransforms();
  components[c]->constrainZeroOne(leaf);
  syncTransformsFromComponents();
}

void CComponentKern::constrainNone(const string& path)
{
  unsigned int c=0;
  string leaf;
  splitComponentPath(path, c, leaf);
  unlinkTransforms();
  components[c]->constrainNone(leaf);
  syncTransformsFromComponents();
}

ostream& CKern::display(ostream& os) const
{
  os << getName() << " kernel:" << endl;
  for(unsigned int i=0; i<nParams; i++)
  {
    os << getParamName(i) << ": " << getParam(i) << endl;
  }
  return os;
}

void CKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "numParams", getNumParams());
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
  writeToStream(out, "numPriors", getNumPriors());
  writePriorsToStream(out);
}

void CKern::getGradPrior(CMatrix& G) const
{
}

void CKern::getPriorLogProb(CMatrix& G) const
{
}

void CKern::getGradTransParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getNumParams());
  getGradParams(g, X, X2, cvGrd, regularise);
  double val;
  double param;
  for(unsigned int i=0; i<getNumTransforms(); i++)
  {
    val=g.getVal(getTransformIndex(i));
    param=getParam(getTransformIndex(i));
    g.setVal(val*getTransformGradFact(param, i), getTransformIndex(i));
  }
}
void CKern::getGradTransParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getNumParams());
  getGradParams(g, X, cvGrd, regularise);
  double val;
  double param;
  for(unsigned int i=0; i<getNumTransforms(); i++)
  {
    val=g.getVal(getTransformIndex(i));
    param=getParam(getTransformIndex(i));
    g.setVal(val*getTransformGradFact(param, i), getTransformIndex(i));
  }  
}
void CKern::getDiagGradTransParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getNumParams());
  getDiagGradParams(g, X, cvGrd, regularise);
  double val;
  double param;
  for(unsigned int i=0; i<getNumTransforms(); i++)
  {
    val=g.getVal(getTransformIndex(i));
    param=getParam(getTransformIndex(i));
    g.setVal(val*getTransformGradFact(param, i), getTransformIndex(i));
  }  
}
bool CKern::equals(const CKern& kern, double tol) const
{
  if(getType()!=kern.getType())
    return false;
  if(getNumParams()!=kern.getNumParams())
    return false;
  CMatrix params(1, getNumParams());
  getParams(params);
  CMatrix kernParams(1, getNumParams());
  kern.getParams(kernParams);
  if(!params.equals(kernParams, tol))
    return false;
  return true;
}
void CKern::initDefaultActiveDims()
{
  activeDims.resize(inputDim);
  for(unsigned int i=0; i<inputDim; i++)
    activeDims[i] = i;
}
bool CKern::isDefaultActiveDims() const
{
  if(activeDims.size()!=inputDim)
    return false;
  for(unsigned int i=0; i<inputDim; i++)
  {
    if(activeDims[i]!=i)
      return false;
  }
  return true;
}
void CKern::setActiveDims(const vector<unsigned int>& dims)
{
  DIMENSIONMATCH(dims.size()==inputDim);
  for(unsigned int i=0; i<dims.size(); i++)
  {
    for(unsigned int j=i+1; j<dims.size(); j++)
      BOUNDCHECK(dims[i]!=dims[j]);
  }
  activeDims = dims;
}
double CKern::dist2Active(const CMatrix& X1, unsigned int i1,
                          const CMatrix& X2, unsigned int i2) const
{
  BOUNDCHECK(i1<X1.getRows());
  BOUNDCHECK(i2<X2.getRows());
  if(isDefaultActiveDims()
     && X1.getCols()==inputDim
     && X2.getCols()==inputDim)
    return X1.dist2Row(i1, X2, i2);
  double val = 0.0;
  for(unsigned int d=0; d<activeDims.size(); d++)
  {
    unsigned int col = activeDims[d];
    BOUNDCHECK(col<X1.getCols());
    BOUNDCHECK(col<X2.getCols());
    double diff = X1.getVal(i1, col) - X2.getVal(i2, col);
    val += diff*diff;
  }
  return val;
}
unsigned int CKern::indexFromDouble(double x, unsigned int outputDim)
{
  long long rounded = llround(x);
  BOUNDCHECK(fabs(x - static_cast<double>(rounded)) < ndlutil::MATCHTOL);
  BOUNDCHECK(rounded >= 0);
  unsigned int i = static_cast<unsigned int>(rounded);
  BOUNDCHECK(i < outputDim);
  return i;
}
// The Component kernel
void CComponentKern::readParamsFromStream(istream& in) 
{
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setInputDim(readIntFromStream(in, "inputDim"));
  unsigned int nPar = readIntFromStream(in, "numParams");
  unsigned int numKerns = readIntFromStream(in, "numKerns");
  for(unsigned int i=0; i<numKerns; i++)
  {
    if(i<components.size())
      components[i]->fromStream(in);
    else
      addKern(readKernFromStream(in));
  }
  
}
void CComponentKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "numParams", getNumParams());
  writeToStream(out, "numKerns", (unsigned int)components.size());
  for(unsigned int i=0; i<components.size(); i++)
  {
    components[i]->toStream(out);
  }
}


// The Compound kernel.
CCmpndKern::CCmpndKern() : CComponentKern()
{
  _init();
}
CCmpndKern::CCmpndKern(unsigned int inDim) : CComponentKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CCmpndKern::CCmpndKern(const CMatrix& X) : CComponentKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CCmpndKern::CCmpndKern(const CCmpndKern& kern) : CComponentKern()
{
  _init();
  setInputDim(kern.getInputDim());
  // Start empty: CComponentKern(kern) would shallow-copy pointers, then
  // addKern would duplicate (and grow size mid-loop).
  for(size_t i=0; i<kern.components.size(); i++)
    addKern(kern.components[i]);
}
// Class destructor
CCmpndKern::~CCmpndKern()
{
  for(size_t i=0; i<components.size(); i++)
    delete components[i];
}
void CCmpndKern::_init()
{
  nParams=0;
  setType("cmpnd");
  setName("compound");
  setStationary(true);
}
void CCmpndKern::setInitParam()
{
}
double CCmpndKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  double y=0.0;
  for(size_t i=0; i<components.size(); i++)
    y+=components[i]->diagComputeElement(X, index);
  return y;
}
void CCmpndKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.zeros();
  CMatrix dStore(d.getRows(), d.getCols(), 0.0);
  for(size_t i=0; i < components.size(); i++)
  {
    components[i]->diagCompute(dStore, X);
    d.axpy(dStore, 1.0);
  }
}
void CCmpndKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  if(!addG)
    gX.zeros();
  for(size_t i=0; i<components.size(); i++)
    components[i]->getGradX(gX, X, row, X2, true);
}
void CCmpndKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
  for(size_t i=0; i<components.size(); i++)
  {
    components[i]->getDiagGradX(gX, X, true);
  }
}
double CCmpndKern::getWhite() const
{
  double white = 0.0;
  for(size_t i=0; i<components.size(); i++)
    white += components[i]->getWhite();
  return white;
}
double CCmpndKern::getVariance() const
{
  double totVariance = 0.0;
  for(size_t i=0; i<components.size(); i++)
    totVariance += components[i]->getVariance();
  return totVariance;
}

double CCmpndKern::computeElement(const CMatrix& X1, unsigned int index1, 
				  const CMatrix& X2, unsigned int index2) const
{
  double y=0.0;
  for(size_t i=0; i<components.size(); i++)
    y+=components[i]->computeElement(X1, index1, X2, index2);
  return y;
}

// CIP-0013 P4: dispatch to each child's compute and accumulate.
void CCmpndKern::compute(CMatrix& K, const CMatrix& X) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  MATRIXPROPERTIES(K.isSquare());
  K.zeros();
  CMatrix K2(K.getRows(), K.getCols());
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->compute(K2, X);
    K.axpy(K2, 1.0);
  }
  K.setSymmetric(true);
}
void CCmpndKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols() == X2.getRows());
  K.zeros();
  CMatrix K2(K.getRows(), K.getCols());
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->compute(K2, X, X2);
    K.axpy(K2, 1.0);
  }
}
void CCmpndKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getNumParams());
  unsigned int start = 0;
  unsigned int end = 0;
  for(size_t i=0; i<components.size(); i++)
  {
    end = start+components[i]->getNumParams()-1;
    CMatrix subg(1, components[i]->getNumParams());
    components[i]->getGradParams(subg, X, X2, covGrad, regularise);
    g.setMatrix(0, start, subg);
    start = end+1;
  }
}
void CCmpndKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getNumParams());
  unsigned int start = 0;
  unsigned int end = 0;
  for(size_t i=0; i<components.size(); i++)
  {
    end = start+components[i]->getNumParams()-1;
    CMatrix subg(1, components[i]->getNumParams());
    components[i]->getGradParams(subg, X, covGrad, regularise);
    g.setMatrix(0, start, subg);
    start = end+1;
  }
}
double CCmpndKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  unsigned int start=0;
  unsigned int end=0;
  for(size_t i=0; i<components.size(); i++)
  {
    end = start+components[i]->getNumParams()-1;
    if(index<end)
      return components[i]->getGradParam(index-start, X, X2, covGrad);
    start = end + 1;
  }
  return -1;
}
double CCmpndKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  unsigned int start=0;
  unsigned int end=0;
  for(size_t i=0; i<components.size(); i++)
  {
    end = start+components[i]->getNumParams()-1;
    if(index<end)
      return components[i]->getGradParam(index-start, X, covGrad);
    start = end + 1;
  }
  return -1;
}

// The Tensor kernel.
CTensorKern::CTensorKern() : CComponentKern()
{
  _init();
}
CTensorKern::CTensorKern(unsigned int inDim) : CComponentKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CTensorKern::CTensorKern(const CMatrix& X) : CComponentKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CTensorKern::CTensorKern(const CTensorKern& kern) : CComponentKern()
{
  _init();
  setInputDim(kern.getInputDim());
  for(size_t i=0; i<kern.components.size(); i++)
    addKern(kern.components[i]);
}
CTensorKern::CTensorKern(const CTensorKern& kern, unsigned int comp) : CComponentKern()
{
  // Leave-one-out (slash) tensor for product-rule gradients. Must not
  // inherit shallow-copied component pointers from kern.
  _init();
  BOUNDCHECK(comp<kern.components.size());
  setInputDim(kern.getInputDim());
  for(size_t i=0; i<comp; i++)
    addKern(kern.components[i]);
  for(size_t i=comp+1; i<kern.components.size(); i++)
    addKern(kern.components[i]);
}
// Class destructor
CTensorKern::~CTensorKern()
{
  for(size_t i=0; i<components.size(); i++)
    delete components[i];

}
void CTensorKern::_init()
{
  nParams=0;
  setType("tensor");
  setName("tensor");
  setStationary(true);
}
void CTensorKern::setInitParam()
{
}
unsigned int CTensorKern::addKern(const CKern* kern)
{
  if(kern->getWhite())
  {
    throw ndlexceptions::Error("Components of a tensor kernel should not contain white noise");

  }
  if(kern->getType()=="indexwhite")
  {
    throw ndlexceptions::Error("indexwhite must be summed outside a tensor (CIP-000D), not multiplied inside it");
  }
  return CComponentKern::addKern(kern);
}
double CTensorKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  double y=1.0;
  for(size_t i=0; i<components.size(); i++)
    y*=components[i]->diagComputeElement(X, index);
  return y;
}
void CTensorKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.ones();
  CMatrix dStore(d.getRows(), d.getCols(), 0.0);
  for(size_t i=0; i < components.size(); i++)
  {
    components[i]->diagCompute(dStore, X);
    d.dotMultiply(dStore);
  }
}
double CTensorKern::getWhite() const
{
  return 0.0;
}
double CTensorKern::getVariance() const
{
  double totVariance = 1.0;
  for(size_t i=0; i<components.size(); i++)
    totVariance *= components[i]->getVariance();
  return totVariance;
}

double CTensorKern::computeElement(const CMatrix& X1, unsigned int index1, 
				   const CMatrix& X2, unsigned int index2) const
{
  double y=1.0;
  for(size_t i=0; i<components.size(); i++)
    y*=components[i]->computeElement(X1, index1, X2, index2);
  return y;
}

// void CTensorKern::compute(CMatrix& K, const CMatrix& X) const
// {
//   DIMENSIONMATCH(K.rowsMatch(X));
//   DIMENSIONMATCH(K.isSquare());
//   K.ones();
//   CMatrix K2(K.getRows(), K.getCols());
//   for(size_t i=0; i<components.size(); i++)
//   {
//     components[i]->compute(K2, X);
//     K.dotMultiply(K2);
//   }
// }

// void CTensorKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
// {
//   DIMENSIONMATCH(K.rowsMatch(X));
//   DIMENSIONMATCH(K.getCols()==X2.getRows());
//   //CMatrix K2(K.getRows(), K.getCols());
//   for(unsigned int i=0; i<X.getRows(); i++)
//   {
//     for(unsigned int j=0; j<X2.getRows(); j++)
//     {
//       double kval=1.0;
//       for(size_t k=0; k<components.size(); k++)
// 	kval*=components[k]->computeElement(X, i, X2, j);
//       K.setVal(kval, i, j);
//     }
//   }
// }
// void CTensorKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const
// {
//   DIMENSIONMATCH(K.rowsMatch(X));
//   DIMENSIONMATCH(K.getCols()==1);
//   //CMatrix K2(K.getRows(), K.getCols());
//   for(unsigned int i=0; i<X.getRows(); i++)
//   {
//     double kval=1.0;
//     for(size_t k=0; k<components.size(); k++)
//       kval*=components[k]->computeElement(X, i, X2, row);
//     K.setVal(kval, i, row);
//   }
// }
void CTensorKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  if(!addG)
    gX.zeros();
  CMatrix gXTemp(gX.getRows(), gX.getCols());
  CMatrix Kslash(X2.getRows(), 1);
  for(size_t i=0; i<components.size(); i++)
  {
    CTensorKern slashKern(*this, (unsigned int)i);
    components[i]->getGradX(gXTemp, X, row, X2);
    slashKern.compute(Kslash, X2, X, row);
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      gXTemp.dotMultiplyColCol(j, Kslash, 0);
    }
    gX.add(gXTemp);
  }
}
void CTensorKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
  CMatrix gXTemp(gX.getRows(), gX.getCols());
  CMatrix diagKslash(X.getRows(), 1);
  for(size_t i=0; i<components.size(); i++)
  {
    CTensorKern slashKern(*this, (unsigned int)i);
    components[i]->getDiagGradX(gXTemp, X);
    slashKern.diagCompute(diagKslash, X);
    for(unsigned int k=0; k<getInputDim(); k++)
    {
      gXTemp.dotMultiplyColCol(k, diagKslash, 0);
    }
    gX.add(gXTemp);
  }
}
void CTensorKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getNumParams());
  unsigned int start = 0;
  unsigned int end = 0;
  CMatrix tempCovGrad(covGrad.getRows(), covGrad.getCols());
  for(size_t i=0; i<components.size(); i++)
  {
    CTensorKern slashKern(*this, (unsigned int)i);
    slashKern.compute(tempCovGrad, X, X2);
    tempCovGrad.dotMultiply(covGrad);
    end = start+components[i]->getNumParams()-1;
    CMatrix subg(1, components[i]->getNumParams());
    components[i]->getGradParams(subg, X, X2, tempCovGrad, regularise);
    g.setMatrix(0, start, subg);
    start = end+1;
  }
}
void CTensorKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getNumParams());
  unsigned int start = 0;
  unsigned int end = 0;
  CMatrix tempCovGrad(covGrad.getRows(), covGrad.getCols());
  for(size_t i=0; i<components.size(); i++)
  {
    CTensorKern slashKern(*this, (unsigned int)i);
    slashKern.compute(tempCovGrad, X);
    tempCovGrad.dotMultiply(covGrad);
    end = start+components[i]->getNumParams()-1;
    CMatrix subg(1, components[i]->getNumParams());
    tempCovGrad.setSymmetric(true);
    components[i]->getGradParams(subg, X, tempCovGrad, regularise);
    g.setMatrix(0, start, subg);
    start = end+1;
  }
}
double CTensorKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  unsigned int start=0;
  unsigned int end=0;
  CMatrix tempCovGrad(covGrad.getRows(), covGrad.getCols());
  for(size_t i=0; i<components.size(); i++)
  {
    end = start+components[i]->getNumParams()-1;
    if(index<=end)
    {
      CTensorKern slashKern(*this, (unsigned int)i);
      slashKern.compute(tempCovGrad, X, X2);
      tempCovGrad.dotMultiply(covGrad);
      return components[i]->getGradParam(index-start, X, X2, tempCovGrad);
    }
    start = end + 1;
  }
  return -1;
}
double CTensorKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  unsigned int start=0;
  unsigned int end=0;
  CMatrix tempCovGrad(covGrad.getRows(), covGrad.getCols());
  for(unsigned int i=0; i<components.size(); i++)
  {
    end = start+components[i]->getNumParams()-1;
    if(index<=end)
    {
      CTensorKern slashKern(*this, i);
      slashKern.compute(tempCovGrad, X);
      tempCovGrad.dotMultiply(covGrad);
      tempCovGrad.setSymmetric(true);

      return components[i]->getGradParam(index-start, X, tempCovGrad);
    }
    start = end + 1;
  }
  return -1;
}
// the white noise kernel.
CWhiteKern::CWhiteKern() : CKern()
{
  _init();
}
CWhiteKern::CWhiteKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CWhiteKern::CWhiteKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CWhiteKern::CWhiteKern(const CWhiteKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
}
// Class destructor
CWhiteKern::~CWhiteKern()
{
}
double CWhiteKern::getVariance() const
{
  return variance;
}
void CWhiteKern::_init()
{
  nParams = 1;
  setType("white");
  setName("white noise");
  setParamName("variance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setStationary(true);
}
void CWhiteKern::setInitParam()
{
  variance = exp(-2.0);  
}

inline double CWhiteKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CWhiteKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
void CWhiteKern::setParam(double val, unsigned int paramNo)
{
  DIMENSIONMATCH(paramNo==0);
  switch(paramNo)
  {
  case 0:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
// Parameters are kernel parameters
double CWhiteKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo==0);
  switch(paramNo)
  {
  case 0:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
void CWhiteKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2,  bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  if(!addG)
  {
    gX.zeros();
  }
}
void CWhiteKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CWhiteKern::getWhite() const
{
  return variance;
}

inline double CWhiteKern::computeElement(const CMatrix& X1, unsigned int index1,
					 const CMatrix& X2, unsigned int index2) const
{
  return 0.0;
}

void CWhiteKern::compute(CMatrix& K, const CMatrix& X) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  MATRIXPROPERTIES(K.isSquare());
  K.zeros();
  for(unsigned int i=0; i<K.getRows(); i++)
    K.setVal(variance, i, i);
  K.setSymmetric(true);
}

void CWhiteKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==X2.getRows());
  K.zeros();
}
void CWhiteKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==1);
  K.zeros();
}
double CWhiteKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index==0);
  return 0.0;
}
double CWhiteKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index==0);
  return trace(covGrad);
}

// the white noise kernel.
CWhitefixedKern::CWhitefixedKern() : CKern()
{
  _init();
}
CWhitefixedKern::CWhitefixedKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CWhitefixedKern::CWhitefixedKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CWhitefixedKern::CWhitefixedKern(const CWhitefixedKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
}
// Class destructor
CWhitefixedKern::~CWhitefixedKern()
{
}
ostream& CWhitefixedKern::display(ostream& os) const
{
  os << getName() << " kernel:" << endl;
  os << "variance: " << variance << endl;
  return os;
}

void CWhitefixedKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "numParams", getNumParams());
  writeToStream(out, "variance", variance);
}
void CWhitefixedKern::readParamsFromStream(istream& in)
{
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setInputDim(readIntFromStream(in, "inputDim"));
  unsigned int numParams=readIntFromStream(in, "numParams");
  setVariance(readDoubleFromStream(in, "variance"));
}

double CWhitefixedKern::getVariance() const
{
  return variance;
}
void CWhitefixedKern::_init()
{
  nParams = 0;
  setType("whitefixed");
  setName("fixed white noise");
  setStationary(true);
}
void CWhitefixedKern::setInitParam()
{
  variance = exp(-2.0);  
}

inline double CWhitefixedKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CWhitefixedKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
void CWhitefixedKern::setParam(double val, unsigned int paramNo)
{
  throw ndlexceptions::Error("Requested parameter doesn't exist.");
}
// Parameters are kernel parameters
double CWhitefixedKern::getParam(unsigned int paramNo) const
{
  throw ndlexceptions::Error("Requested parameter doesn't exist.");
}
void CWhitefixedKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2,  bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  if(!addG)
  {
    gX.zeros();
  }
}
void CWhitefixedKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CWhitefixedKern::getWhite() const
{
  return variance;
}

inline double CWhitefixedKern::computeElement(const CMatrix& X1, unsigned int index1,
					 const CMatrix& X2, unsigned int index2) const
{
  return 0.0;
}

void CWhitefixedKern::compute(CMatrix& K, const CMatrix& X) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  MATRIXPROPERTIES(K.isSquare());
  K.zeros();
  for(unsigned int i=0; i<K.getRows(); i++)
    K.setVal(variance, i, i);
  K.setSymmetric(true);
}

void CWhitefixedKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==X2.getRows());
  K.zeros();
}
void CWhitefixedKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==1);
  K.zeros();
}
double CWhitefixedKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  throw ndlexceptions::Error("Requested parameter doesn't exist.");

}
double CWhitefixedKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  throw ndlexceptions::Error("Requested parameter doesn't exist.");
}

// the bias kernel.
CBiasKern::CBiasKern() : CKern()
{
  _init();
}
CBiasKern::CBiasKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CBiasKern::CBiasKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CBiasKern::CBiasKern(const CBiasKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  
}
// Class destructor
CBiasKern::~CBiasKern()
{
}
double CBiasKern::getVariance() const
{
  return variance;
}
void CBiasKern::_init()
{
  nParams = 1;
  setType("bias");
  setName("bias");
  setParamName("variance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setStationary(true);
}

void CBiasKern::setInitParam()
{
  variance = exp(-2.0);  
}

double CBiasKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CBiasKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CBiasKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo==0);
  switch(paramNo)
  {
  case 0:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CBiasKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo==0);
  switch(paramNo)
  {
  case 0:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
void CBiasKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  if(!addG)
  {
    gX.zeros();
  }
}
void CBiasKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CBiasKern::getWhite() const
{
  return 0.0;
}

inline double CBiasKern::computeElement(const CMatrix& X1, unsigned int index1, 
					const CMatrix& X2, unsigned int index2) const
{
  return variance;
}

void CBiasKern::compute(CMatrix& K, const CMatrix& X) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  MATRIXPROPERTIES(K.isSquare());
  K.setVals(variance);
  K.setSymmetric(true);
}

void CBiasKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==X2.getRows());
  K.setVals(variance);
}
void CBiasKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==1);
  K.setVals(variance);
}
double CBiasKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const 
{
  BOUNDCHECK(index==0);
  return covGrad.sum();
}
double CBiasKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const 
{
  BOUNDCHECK(index==0);
  return covGrad.sum();
}


// CIP-0013 P4/P5: Stationary base — bulk compute + per-eval workspace.
void CStationaryKern::invalidateWorkspace()
{
  ++m_wsEpoch;
  m_symValid = false;
  m_rectValid = false;
  m_symX = 0;
  m_rectX = 0;
  m_rectX2 = 0;
  setUpdateXused(false);
}
bool CStationaryKern::hasSymWorkspace(const CMatrix& X) const
{
  const unsigned int n = X.getRows();
  return m_symValid && m_symEpoch == m_wsEpoch && m_symX == &X
    && m_workR2Sym.getRows() == n && m_workR2Sym.getCols() == n
    && m_workKSym.getRows() == n && m_workKSym.getCols() == n;
}
bool CStationaryKern::hasRectWorkspace(const CMatrix& X, const CMatrix& X2) const
{
  const unsigned int n = X.getRows();
  const unsigned int m = X2.getRows();
  return m_rectValid && m_rectEpoch == m_wsEpoch && m_rectX == &X && m_rectX2 == &X2
    && m_workR2Rect.getRows() == n && m_workR2Rect.getCols() == m
    && m_workKRect.getRows() == n && m_workKRect.getCols() == m;
}
void CStationaryKern::fillSymWorkspace(const CMatrix& X) const
{
  const unsigned int n = X.getRows();
  m_workR2Sym.resize(n, n);
  m_workKSym.resize(n, n);
  const double s = scaleFactor();
  const unsigned int q = X.getCols();

  if(isDefaultActiveDims() && q == getInputDim() && q >= kGemmDistMinQ)
  {
    // CIP-0025 Track 1 — GEMM distance path.
    // R^2_ij = ||xi||^2 + ||xj||^2 - 2 xi·xj
    // Step 1: m_workR2Sym = -2 * X * X^T  (column-major dgemm, beta=0)
    m_workR2Sym.gemm(X, X, -2.0, 0.0, "n", "t");
    // Step 2: Row squared norms
    std::vector<double> rowNorm2(n, 0.0);
    for(unsigned int i = 0; i < n; i++)
    {
      double nrm = 0.0;
      for(unsigned int c = 0; c < q; c++)
      {
        const double v = X.getVal(i, c);
        nrm += v * v;
      }
      rowNorm2[i] = nrm;
    }
    // Step 3: Add row norms, clamp r2 >= 0, set diagonal exactly 0
    for(unsigned int i = 0; i < n; i++)
    {
      m_workR2Sym.setVal(0.0, i, i);
      for(unsigned int j = 0; j < i; j++)
      {
        double r2 = m_workR2Sym.getVal(i, j) + rowNorm2[i] + rowNorm2[j];
        if(r2 < 0.0) r2 = 0.0;
        m_workR2Sym.setVal(r2, i, j);
        m_workR2Sym.setVal(r2, j, i);
      }
    }
    // Step 4: Apply K_of_r — diagonal via diagComputeElement; off-diag rows
    // in parallel (each thread i writes row i and symmetric col i — disjoint).
    for(unsigned int i = 0; i < n; i++)
      m_workKSym.setVal(diagComputeElement(X, i), i, i);
#ifdef GPC_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(unsigned int i = 0; i < n; i++)
    {
      for(unsigned int j = 0; j < i; j++)
      {
        const double k = K_of_r(s * sqrt(m_workR2Sym.getVal(i, j)));
        m_workKSym.setVal(k, i, j);
        m_workKSym.setVal(k, j, i);
      }
    }
  }
  else
  {
    // Scalar path: small q (< kGemmDistMinQ) or non-default activeDims.
    // OpenMP over disjoint-write symmetric rows: thread i writes
    // row i (cols 0..i-1) and col i (rows 0..i-1) — no overlap between threads.
#ifdef GPC_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(unsigned int i = 0; i < n; i++)
    {
      m_workR2Sym.setVal(0.0, i, i);
      m_workKSym.setVal(diagComputeElement(X, i), i, i);
      for(unsigned int j = 0; j < i; j++)
      {
        const double r2 = dist2Active(X, i, X, j);
        const double k = K_of_r(s * sqrt(r2));
        m_workR2Sym.setVal(r2, i, j);
        m_workR2Sym.setVal(r2, j, i);
        m_workKSym.setVal(k, i, j);
        m_workKSym.setVal(k, j, i);
      }
    }
  }
  m_symX = &X;
  m_symEpoch = m_wsEpoch;
  m_symValid = true;
}
void CStationaryKern::fillRectWorkspace(const CMatrix& X, const CMatrix& X2) const
{
  const unsigned int n = X.getRows();
  const unsigned int m = X2.getRows();
  m_workR2Rect.resize(n, m);
  m_workKRect.resize(n, m);
  const double s = scaleFactor();
  const unsigned int q = X.getCols();

  if(isDefaultActiveDims() && q == getInputDim()
     && X2.getCols() == getInputDim() && q >= kGemmDistMinQ)
  {
    // CIP-0025 Track 1 — GEMM distance path for rectangular blocks.
    // R^2_ij = ||xi||^2 + ||x2j||^2 - 2 xi·x2j
    m_workR2Rect.gemm(X, X2, -2.0, 0.0, "n", "t");
    // Row squared norms for X and X2
    std::vector<double> rowNorm2X(n, 0.0);
    for(unsigned int i = 0; i < n; i++)
    {
      double nrm = 0.0;
      for(unsigned int c = 0; c < q; c++) { const double v = X.getVal(i, c); nrm += v * v; }
      rowNorm2X[i] = nrm;
    }
    std::vector<double> rowNorm2X2(m, 0.0);
    for(unsigned int j = 0; j < m; j++)
    {
      double nrm = 0.0;
      for(unsigned int c = 0; c < q; c++) { const double v = X2.getVal(j, c); nrm += v * v; }
      rowNorm2X2[j] = nrm;
    }
    // Add norms, clamp r2 >= 0, apply K_of_r.
    // Rows are fully disjoint: OpenMP parallel over outer index i is safe.
#ifdef GPC_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(unsigned int i = 0; i < n; i++)
    {
      for(unsigned int j = 0; j < m; j++)
      {
        double r2 = m_workR2Rect.getVal(i, j) + rowNorm2X[i] + rowNorm2X2[j];
        if(r2 < 0.0) r2 = 0.0;
        m_workR2Rect.setVal(r2, i, j);
        m_workKRect.setVal(K_of_r(s * sqrt(r2)), i, j);
      }
    }
  }
  else
  {
    // Scalar path: small q or non-default activeDims. Each row i is independent.
#ifdef GPC_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(unsigned int i = 0; i < n; i++)
    {
      for(unsigned int j = 0; j < m; j++)
      {
        const double r2 = dist2Active(X, i, X2, j);
        const double k = K_of_r(s * sqrt(r2));
        m_workR2Rect.setVal(r2, i, j);
        m_workKRect.setVal(k, i, j);
      }
    }
  }
  m_rectX = &X;
  m_rectX2 = &X2;
  m_rectEpoch = m_wsEpoch;
  m_rectValid = true;
}
void CStationaryKern::updateX(const CMatrix& X)
{
  // Cache only for the default full-width path; activeDims ICM layouts skip it.
  if(!isDefaultActiveDims() || X.getCols() != getInputDim())
  {
    setUpdateXused(false);
    m_symValid = false;
    m_symX = 0;
    return;
  }
  fillSymWorkspace(X);
  setUpdateXused(true);
}
double CStationaryKern::computeElement(const CMatrix& X1, unsigned int index1,
                                       const CMatrix& X2, unsigned int index2) const
{
  const double r2 = dist2Active(X1, index1, X2, index2);
  const double r = scaleFactor() * sqrt(r2);
  return K_of_r(r);
}
void CStationaryKern::compute(CMatrix& K, const CMatrix& X) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  MATRIXPROPERTIES(K.isSquare());
  fillSymWorkspace(X);
  const unsigned int n = K.getRows();
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = 0; j < i; j++)
    {
      const double k = symK(i, j);
      K.setVal(k, i, j);
      K.setVal(k, j, i);
    }
    K.setVal(symK(i, i), i, i);
  }
  K.setSymmetric(true);
}
void CStationaryKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols() == X2.getRows());
  fillRectWorkspace(X, X2);
  const unsigned int n = K.getRows();
  const unsigned int m = K.getCols();
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = 0; j < m; j++)
      K.setVal(rectK(i, j), i, j);
  }
}

// the RBF kernel.
CRbfKern::CRbfKern() : CStationaryKern()
{
  _init();
}
CRbfKern::CRbfKern(unsigned int inDim) : CStationaryKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CRbfKern::CRbfKern(const CMatrix& X) : CStationaryKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CRbfKern::CRbfKern(const CRbfKern& kern) : CStationaryKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  activeDims = kern.activeDims;
  variance = kern.variance;
  inverseWidth = kern.inverseWidth;
}
// Class destructor
CRbfKern::~CRbfKern()
{
}
double CRbfKern::getVariance() const
{
  return variance;
}
void CRbfKern::_init()
{
  nParams = 2;
  setType("rbf");
  setName("RBF");
  setParamName("inverseWidth", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}

// CIP-0024: GPy ExpQuad / EQ — thin RBF subclass, lengthscale-first surface.
void CEqKern::_markEq()
{
  setType("eq");
  setName("ExpQuad");
}
CEqKern::CEqKern() : CRbfKern()
{
  _markEq();
}
CEqKern::CEqKern(unsigned int inDim) : CRbfKern(inDim)
{
  _markEq();
}
CEqKern::CEqKern(const CMatrix& X) : CRbfKern(X)
{
  _markEq();
}
CEqKern::CEqKern(const CEqKern& kern) : CRbfKern(kern)
{
  _markEq();
}
CEqKern::~CEqKern()
{
}
string CEqKern::gpyComponentName() const
{
  return "ExpQuad";
}
void CRbfKern::setInitParam()
{
  inverseWidth = 1.0;
  variance = 1.0;
}

inline double CRbfKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CRbfKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CRbfKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    inverseWidth = val;
    break;
  case 1:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
  invalidateWorkspace();
}
double CRbfKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return inverseWidth;
    break;
  case 1:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
void CRbfKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2 = 0.5*inverseWidth;
  double pf = variance*inverseWidth;
  DIMENSIONMATCH(gX.getCols()==X2.getCols());
  if(!addG)
    gX.zeros();
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double n2 = dist2Active(X, row, X2, k);
    for(unsigned int d=0; d<activeDims.size(); d++)
    {
      unsigned int j = activeDims[d];
      BOUNDCHECK(j<X2.getCols());
      double val = pf*(X2.getVal(k, j)-X.getVal(row, j))*exp(-n2*wi2);
      gX.addVal(val, k, j);
    }
  }
}
void CRbfKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CRbfKern::getWhite() const
{
  return 0.0;
}

void CRbfKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  double halfInverseWidth=0.5*inverseWidth;
  double halfVariance=0.5*variance;
  const unsigned int nrows = X.getRows();
  const unsigned int ncols = X2.getRows();
  const bool useWork = hasRectWorkspace(X, X2);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;

  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<ncols; i++)
    {
      double dist2 = 0.0;
      double k = 0.0;
      if(useWork)
      {
        dist2 = rectR2(j, i);
        k = rectK(j, i) * invVar;
      }
      else
      {
        dist2 = dist2Active(X, j, X2, i);
        k = exp(-dist2*halfInverseWidth);
      }
      double kcg_ij = k*covGrad.getVal(j,i);
      g1 -= halfVariance*dist2*kcg_ij; // dk()/dgamma in SBIK paper
      g2 += kcg_ij;                    // dk()/dalpha in SBIK paper
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  if(regularise)
    addPriorGrad(g);
}

void CRbfKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  MATRIXPROPERTIES(covGrad.isSymmetric());
  double g1=0.0;
  double g2=0.0;
  double halfInverseWidth=0.5*inverseWidth;
  double halfVariance=0.5*variance;
  const unsigned int nrows = X.getRows();
  const bool useWork = hasSymWorkspace(X);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;

  for(unsigned int j=0; j<nrows; j++)
  {
    g2 += covGrad.getVal(j,j);
    for(unsigned int i=0; i<j; i++)
    {
      double k = 0.0;
      double dist2 = 0.0;
      if(useWork)
      {
        dist2 = symR2(i, j);
        k = symK(i, j) * invVar;
      }
      else
      {
        dist2 = dist2Active(X, i, X, j);
        k = exp(-dist2*halfInverseWidth);
      }
      double kcg_ij = k*covGrad.getVal(i,j);
      g1 -= 2.0*halfVariance*dist2*kcg_ij; // dk()/dgamma in SBIK paper
      g2 += 2.0*kcg_ij;                    // dk()/dalpha in SBIK paper
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  if(regularise)
    addPriorGrad(g);
}

double CRbfKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfKern");
  
}
double CRbfKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfKern");
  
}


//////////////////////////////////////////////////////////////////////////////////////////////
// Exponential kernel, written by Karel Lebeda with no guarantees whatsoever...
//////////////////////////////////////////////////////////////////////////////////////////////
CExpKern::CExpKern() : CStationaryKern()
{
  _init();
}
CExpKern::CExpKern(unsigned int inDim) : CStationaryKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CExpKern::CExpKern(const CMatrix& X) : CStationaryKern(X)
{
  _init();
  setInputDim(X.getCols());
}
CExpKern::CExpKern(const CExpKern& kern) : CStationaryKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  inverseWidth = kern.inverseWidth;
}
// Class destructor
CExpKern::~CExpKern()
{
}
double CExpKern::getVariance() const
{
  return variance;
}
void CExpKern::_init()
{
  nParams = 2;
  setType("exp");
  setName("EXP");
  setParamName("inverseWidth", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}
void CExpKern::setInitParam()
{
  inverseWidth = 1.0;
  variance = 1.0;
}

inline double CExpKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CExpKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CExpKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    inverseWidth = val;
    break;
  case 1:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
  invalidateWorkspace();
}
double CExpKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return inverseWidth;
    break;
  case 1:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
// gradient of the kernel function w.r.t. the data
void CExpKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
	DIMENSIONMATCH(gX.getRows() == X2.getRows());
	BOUNDCHECK(row < X.getRows());
	DIMENSIONMATCH(X.getCols()==X2.getCols());
	DIMENSIONMATCH(gX.getCols()==X2.getCols());
	for(unsigned int k=0; k<X2.getRows(); k++)
	{
		double dist = sqrt(dist2Active(X, row, X2, k))*inverseWidth;
		for(unsigned int j=0; j<X2.getCols(); j++)
			{
			// ddist_dX = (X[:, None, :] - X2[None, :, :]) / self.lengthscale ** 2 / np.where(dist != 0., dist, np.inf)
			double ddist_dX = (X2.getVal(k, j) - X.getVal(row, j)) * inverseWidth * inverseWidth;
			if (dist != 0) {
				ddist_dX /= dist;
			} else {
				ddist_dX = 0;
			}
			// dK_dX = -np.transpose(self.variance * np.exp(-dist) * ddist_dX, (1, 0, 2))
			double val = variance * exp(-dist) * ddist_dX;
			if(addG)
				gX.addVal(val, k, j);
			else
				gX.setVal(val, k, j);
		}
	}
}
// gradient of the kernel function w.r.t. the data - some special diagonal version?
void CExpKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CExpKern::getWhite() const
{
  return 0.0;
}

// gradient of the kernel function w.r.t. the parameters (0:inverseWidth=1/gamma, 1:variance=alpha, with two data matrices given)
void CExpKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
	DIMENSIONMATCH(g.getRows()==1);
	DIMENSIONMATCH(g.getCols()==nParams);
	DIMENSIONMATCH(X.getRows()==covGrad.getRows());
	DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
	double g1=0.0;
	double g2=0.0;
	const unsigned int nrows = X.getRows();
	const unsigned int ncols = X2.getRows();
	const bool useWork = hasRectWorkspace(X, X2);
	const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;

	for(unsigned int j=0; j<nrows; j++)
	{
		for(unsigned int i=0; i<ncols; i++)
		{
			double dist2 = 0.0;
			double k = 0.0;
			if(useWork)
			{
				dist2 = rectR2(j, i);
				k = rectK(j, i) * invVar;
			}
			else
			{
				dist2 = dist2Active(X, j, X2, i);
				k = exp(-sqrt(dist2) * inverseWidth);
			}
			double dist = sqrt(dist2) * inverseWidth;
			double invdist = (dist != 0.0) ? (1.0 / dist) : 0.0;
			g2 += k * covGrad.getVal(j,i);
			double dl = - dist2 * k * inverseWidth * variance * invdist;
			g1 += dl * covGrad.getVal(j,i);
		}
	}
	g.setVal(g1, 0);
	g.setVal(g2, 1);
	if(regularise)
		addPriorGrad(g);
}

// gradient of the kernel function w.r.t. the parameters (0:inverseWidth, 1:variance, with one data matrix given)
void CExpKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
	getGradParams(g, X, X, covGrad, regularise); // just call it like the non-sym. case
	return; // it could be sped up by exploiting the fact it's symmetric
}

double CExpKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CExpKern");
}
double CExpKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CExpKern");
}
//////////////////////////////////////////////////////////////////////////////////////////////


// the Rational Quadratic kernel.
CRatQuadKern::CRatQuadKern() : CStationaryKern()
{
  _init();
}
CRatQuadKern::CRatQuadKern(unsigned int inDim) : CStationaryKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CRatQuadKern::CRatQuadKern(const CMatrix& X) : CStationaryKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CRatQuadKern::CRatQuadKern(const CRatQuadKern& kern) : CStationaryKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  alpha = kern.alpha;
  lengthScale = kern.lengthScale;
}
// Class destructor
CRatQuadKern::~CRatQuadKern()
{
}
double CRatQuadKern::getVariance() const
{
  return variance;
}
void CRatQuadKern::_init()
{
  nParams = 3;
  setType("ratquad");
  setName("Rational Quadratic");
  setParamName("alpha", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("lengthScale", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setParamName("variance", 2);
  addTransform(CTransform::defaultPositive(), 2);
  setStationary(true);
}
void CRatQuadKern::setInitParam()
{
  alpha = 1.0;
  lengthScale = 1.0;
  variance = 1.0;
}

inline double CRatQuadKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CRatQuadKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CRatQuadKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    alpha = val;
    break;
  case 1:
    lengthScale = val;
    break;
  case 2:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
  invalidateWorkspace();
}
double CRatQuadKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return alpha;
    break;
  case 1:
    return lengthScale;
    break;
  case 2:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}

void CRatQuadKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2 = 0.5/(lengthScale*lengthScale*alpha);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double n2 = X.dist2Row(row, X2, k);
    double ratquadPart = variance*pow((1+n2*wi2),-(alpha+1))/(lengthScale*lengthScale);
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = ratquadPart*(X2.getVal(k, j)-X.getVal(row, j));
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  }
}
void CRatQuadKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CRatQuadKern::getWhite() const
{
  return 0.0;
}

void CRatQuadKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  double wi2=0.5/(lengthScale*lengthScale*alpha);
  const unsigned int nrows = X.getRows();
  const unsigned int ncols = X2.getRows();
  const bool useWork = hasRectWorkspace(X, X2);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;

  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<ncols; i++)
    {
      double dist2 = 0.0;
      double kbase = 0.0;
      if(useWork)
      {
        dist2 = rectR2(j, i);
        kbase = rectK(j, i) * invVar;
      }
      else
      {
        dist2 = dist2Active(X, j, X2, i);
        kbase = pow(1.0 + dist2*wi2, -alpha);
      }
      double baseVal = (1+dist2*wi2);
      double kbase2 = -alpha*kbase/baseVal;
      double cgVal = covGrad.getVal(j,i);
      double kcg_ij = kbase*cgVal;
      g1 -= cgVal*(dist2*wi2/alpha*kbase2 + log(baseVal)*kbase); 
      g2 -= cgVal*(dist2*kbase2);
      g3 += kcg_ij;                   
    }
  }
  g.setVal(g1*variance, 0);
  g.setVal(g2*variance*2*wi2/lengthScale, 1);
  g.setVal(g3, 2);
  if(regularise)
    addPriorGrad(g);
}

void CRatQuadKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  MATRIXPROPERTIES(covGrad.isSymmetric());
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  double wi2=0.5/(lengthScale*lengthScale*alpha);
  const unsigned int nrows = X.getRows();
  const bool useWork = hasSymWorkspace(X);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;

  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<j; i++)
    {
      double dist2 = 0.0;
      double kbase = 0.0;
      if(useWork)
      {
        dist2 = symR2(i, j);
        kbase = symK(i, j) * invVar;
      }
      else
      {
        dist2 = dist2Active(X, i, X, j);
        kbase = pow(1.0 + dist2*wi2, -alpha);
      }
      double baseVal = (1+dist2*wi2);
      double kbase2 = -alpha*kbase/baseVal;
      double cgVal = covGrad.getVal(j,i);
      double kcg_ij = kbase*cgVal;
      g1 -= cgVal*(dist2*wi2/alpha*kbase2 + log(baseVal)*kbase); 
      g2 -= cgVal*(dist2*kbase2);
      g3 += kcg_ij;                   
    }
    g3+=0.5*covGrad.getVal(j, j);
  }
  g.setVal(2.0*g1*variance, 0);
  g.setVal(4.0*g2*variance*wi2/lengthScale, 1);
  g.setVal(2.0*g3, 2);
  if(regularise)
    addPriorGrad(g);
}

double CRatQuadKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("getGradParam is not currently implemented for CRatQuadKern.");
}
double CRatQuadKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("getGradParam is not currently implemented for CRatQuadKern.");
}


// the Matern 3/2 kernel.
CMatern32Kern::CMatern32Kern() : CStationaryKern()
{
  _init();
}
CMatern32Kern::CMatern32Kern(unsigned int inDim) : CStationaryKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CMatern32Kern::CMatern32Kern(const CMatrix& X) : CStationaryKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CMatern32Kern::CMatern32Kern(const CMatern32Kern& kern) : CStationaryKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  lengthScale = kern.lengthScale;
}
// Class destructor
CMatern32Kern::~CMatern32Kern()
{
}
double CMatern32Kern::getVariance() const
{
  return variance;
}
void CMatern32Kern::_init()
{
  nParams = 2;
  setType("matern32");
  setName("Matern dof=3/2");
  setParamName("lengthScale", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}
void CMatern32Kern::setInitParam()
{
  lengthScale = 1.0;
  variance = 1.0;
}

inline double CMatern32Kern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CMatern32Kern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CMatern32Kern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    lengthScale = val;
    break;
  case 1:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
  invalidateWorkspace();
}
double CMatern32Kern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return lengthScale;
    break;
  case 1:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}

void CMatern32Kern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2= 3.0/(lengthScale*lengthScale);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double n2 = dist2Active(X, row, X2, k);
    double sqrtn2wi2 = sqrt(n2*wi2);
    double expmSqrtn2wi2 = exp(-sqrtn2wi2);
    double K = variance*(1+sqrtn2wi2)*expmSqrtn2wi2;
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double ratio;
      if(sqrtn2wi2 != 0)
	ratio = (X2.getVal(k, j)-X.getVal(row, j))/sqrtn2wi2;
      else
	ratio = 1.0;
      double val = wi2*ratio*(K-variance*expmSqrtn2wi2);
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  }
}
void CMatern32Kern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CMatern32Kern::getWhite() const
{
  return 0.0;
}

void CMatern32Kern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  double wi2=3.0/(lengthScale*lengthScale);
  const unsigned int nrows = X.getRows();
  const unsigned int ncols = X2.getRows();
  const bool useWork = hasRectWorkspace(X, X2);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<ncols; i++)
    {
      double cgVal = covGrad.getVal(j,i);
      double n2 = 0.0;
      double sqrtn2wi2 = 0.0;
      double k = 0.0;
      if(useWork)
      {
        n2 = rectR2(j, i);
        sqrtn2wi2 = scaleFactor() * sqrt(n2);
        k = rectK(j, i) * invVar;
      }
      else
      {
        n2 = dist2Active(X, j, X2, i);
        sqrtn2wi2 = sqrt(n2*wi2);
        k = (1+sqrtn2wi2)*exp(-sqrtn2wi2);
      }
      double expmsq = (sqrtn2wi2 == 0.0) ? 1.0 : (k / (1.0 + sqrtn2wi2));
      double ratio;
      if(sqrtn2wi2==0.0)
	ratio = 1.0;
      else
	ratio = n2/sqrtn2wi2;
      g1 += cgVal*wi2/lengthScale*ratio*(k-expmsq);
      g2 += cgVal*k;
    }
  }
  g.setVal(g1*variance, 0);
  g.setVal(g2, 1);
  if(regularise)
    addPriorGrad(g);
}

void CMatern32Kern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  MATRIXPROPERTIES(covGrad.isSymmetric());
  double g1=0.0;
  double g2=0.0;
  double wi2=3.0/(lengthScale*lengthScale);
  const unsigned int nrows = X.getRows();
  const bool useWork = hasSymWorkspace(X);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;

  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<j; i++)
    {
      double n2 = 0.0;
      double sqrtwi2n2 = 0.0;
      double k = 0.0;
      if(useWork)
      {
	n2 = symR2(i, j);
	sqrtwi2n2 = scaleFactor() * sqrt(n2);
	k = symK(i, j) * invVar;
      }
      else
      {
	n2 = dist2Active(X, i, X, j);
	sqrtwi2n2 = sqrt(n2*wi2);
	k = (1+sqrtwi2n2)*exp(-sqrtwi2n2);
      }
      double ratio;
      if(sqrtwi2n2==0.0)
	ratio = 1.0;
      else
	ratio = n2/sqrtwi2n2;
      double expmsqrtwi2n2 = (sqrtwi2n2 == 0.0) ? 1.0 : (k / (1.0 + sqrtwi2n2));
      double cgVal = covGrad.getVal(j,i);
      g1 += cgVal*wi2/lengthScale*ratio*(k-expmsqrtwi2n2);
      g2 += cgVal*k;
    }
    g2 += 0.5*covGrad.getVal(j, j);
  }
  g.setVal(2.0*g1*variance, 0);
  g.setVal(2.0*g2, 1);
  if(regularise)
    addPriorGrad(g);
}

double CMatern32Kern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern32Kern");
  
}
double CMatern32Kern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern32Kern"); 
}

// the Matern 5/2 kernel.
CMatern52Kern::CMatern52Kern() : CStationaryKern()
{
  _init();
}
CMatern52Kern::CMatern52Kern(unsigned int inDim) : CStationaryKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CMatern52Kern::CMatern52Kern(const CMatrix& X) : CStationaryKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CMatern52Kern::CMatern52Kern(const CMatern52Kern& kern) : CStationaryKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  lengthScale = kern.lengthScale;
}
// Class destructor
CMatern52Kern::~CMatern52Kern()
{
}
double CMatern52Kern::getVariance() const
{
  return variance;
}
void CMatern52Kern::_init()
{
  nParams = 2;
  setType("matern52");
  setName("Matern dof=5/2");
  setParamName("lengthScale", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}
void CMatern52Kern::setInitParam()
{
  lengthScale = 1.0;
  variance = 1.0;
}

inline double CMatern52Kern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CMatern52Kern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CMatern52Kern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    lengthScale = val;
    break;
  case 1:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
  invalidateWorkspace();
}
double CMatern52Kern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return lengthScale;
    break;
  case 1:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
void CMatern52Kern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2= 5.0/(lengthScale*lengthScale);
  double n2 = 0.0;
  double n2wi2 = 0.0;
  double sqrtn2wi2 = 0.0;
  double K = 0.0;
  double ratio = 0.0;
  double val = 0.0;
  double expmSqrtn2wi2 = 0.0;
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    n2 = dist2Active(X, row, X2, k);
    n2wi2 = n2*wi2;
    sqrtn2wi2 = sqrt(n2wi2);
    expmSqrtn2wi2 = exp(-sqrtn2wi2);
    K = variance*(1+sqrtn2wi2+n2wi2/3.0)*expmSqrtn2wi2;
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      if(sqrtn2wi2 != 0.0)
	ratio = (X2.getVal(k, j)-X.getVal(row, j))/sqrtn2wi2;
      else
	ratio = 1.0;
      val = wi2*ratio*(K-variance*expmSqrtn2wi2)-variance*2.0*wi2/3.0*(X2.getVal(k, j)-X.getVal(row, j))*expmSqrtn2wi2;
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  } 
}
void CMatern52Kern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CMatern52Kern::getWhite() const
{
  return 0.0;
}

void CMatern52Kern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  double wi2=5.0/(lengthScale*lengthScale);
  const unsigned int nrows = X.getRows();
  const unsigned int ncols = X2.getRows();
  const bool useWork = hasRectWorkspace(X, X2);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<ncols; i++)
    {
      double cgVal = covGrad.getVal(j,i);
      double n2 = 0.0;
      double k = 0.0;
      if(useWork)
      {
        n2 = rectR2(j, i);
        k = rectK(j, i) * invVar;
      }
      else
      {
        n2 = dist2Active(X, j, X2, i);
      }
      double n2wi2 = n2*wi2;
      double sqrtn2wi2 = sqrt(n2wi2);
      double expmsqrtn2wi2 = exp(-sqrtn2wi2);
      if(!useWork)
        k = (1+sqrtn2wi2+n2wi2/3.0)*expmsqrtn2wi2;
      double ratio;
      if(sqrtn2wi2==0.0)
	ratio = 1.0;
      else
	ratio = n2/sqrtn2wi2;
      g1 += cgVal*(wi2/lengthScale*ratio*(k-expmsqrtn2wi2)-2.0*wi2/(3.0*lengthScale)*n2*expmsqrtn2wi2);
      g2 += cgVal*k;
    }
  }
  g.setVal(g1*variance, 0);
  g.setVal(g2, 1);
  if(regularise)
    addPriorGrad(g);
}

void CMatern52Kern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  MATRIXPROPERTIES(covGrad.isSymmetric());
  double g1=0.0;
  double g2=0.0;
  double wi2=5.0/(lengthScale*lengthScale);
  const unsigned int nrows = X.getRows();
  const bool useWork = hasSymWorkspace(X);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;

  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<j; i++)
    {
      double n2 = 0.0;
      double k = 0.0;
      if(useWork)
      {
	n2 = symR2(i, j);
	k = symK(i, j) * invVar;
      }
      else
      {
	n2 = dist2Active(X, i, X, j);
      }
      double n2wi2 = n2*wi2;
      double sqrtwi2n2 = sqrt(n2wi2);
      double ratio;
      if(sqrtwi2n2==0.0)
	ratio = 1.0;
      else
	ratio = n2/sqrtwi2n2;
      double expmsqrtwi2n2 = exp(-sqrtwi2n2);
      double cgVal = covGrad.getVal(j,i);
      if(!useWork)
        k = (1+sqrtwi2n2+n2wi2/3.0)*expmsqrtwi2n2;
      g1 += cgVal*(wi2/lengthScale*ratio*(k-expmsqrtwi2n2)-2.0*wi2/(3.0*lengthScale)*n2*expmsqrtwi2n2);
      g2 += cgVal*k;
    }
    g2 += 0.5*covGrad.getVal(j, j);
  }
  g.setVal(2.0*g1*variance, 0);
  g.setVal(2.0*g2, 1);
  if(regularise)
    addPriorGrad(g);
}

double CMatern52Kern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern52Kern");
  
}
double CMatern52Kern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern52Kern"); 
}


// the Linear kernel.
CLinKern::CLinKern() : CKern()
{
  _init();
}
CLinKern::CLinKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CLinKern::CLinKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CLinKern::CLinKern(const CLinKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
}
// Class destructor
CLinKern::~CLinKern()
{
}
double CLinKern::getVariance() const
{
  return variance;
}
void CLinKern::_init()
{
  nParams = 1;
  setType("lin");
  setName("linear");
  setParamName("variance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setStationary(false);
}
void CLinKern::setInitParam()
{
  variance = 1.0;
}

double CLinKern::diagComputeElement(const CMatrix& X, unsigned int index1) const
{
  return variance*X.norm2Row(index1);  
}
// Parameters are kernel parameters
void CLinKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo==0);
  switch(paramNo)
  {
  case 0:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CLinKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo==0);
  switch(paramNo)
  {
  case 0:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
void CLinKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());  

  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(k, j);
      val += variance*X2.getVal(k, j);
      gX.setVal(val, k, j);
    }
  }
}

void CLinKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
  {
    gX.deepCopy(X);
    gX.scale(2.0*variance);
  }
  else
  {
    gX.axpy(X, 2.0*variance);
  }
}
double CLinKern::getWhite() const
{
  return 0.0;
}

double CLinKern::computeElement(const CMatrix& X1, unsigned int index1, 
				const CMatrix& X2, unsigned int index2) const
{
  return variance*X1.dotRowRow(index1, X2, index2);
}

void CLinKern::compute(CMatrix& K, const CMatrix& X) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  MATRIXPROPERTIES(K.isSquare());
  K.setSymmetric(true);
  K.syrk(X, variance, 0.0, "u", "n");
}

void CLinKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==X2.getRows());
  K.gemm(X, X2, variance, 0.0, "n", "t");
}
void CLinKern::compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const
{
  DIMENSIONMATCH(K.rowsMatch(X));
  DIMENSIONMATCH(K.getCols()==1);
  K.gemvRowRow(0, X, X2, row, variance, 0.0, "n");
}
double CLinKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index==0);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double dot;
  double g1=0.0;
  for(unsigned int i=0; i<X.getRows(); i++)
    for(unsigned int j=0; j<X2.getRows(); j++)
    {
      dot = X.dotRowRow(i, X2, j);
      g1 += dot*covGrad.getVal(i, j);
    }
  return g1;
}
double CLinKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index==0);
  DIMENSIONMATCH(X.rowsMatch(covGrad));
  MATRIXPROPERTIES(covGrad.isSquare());
  double dot;
  double g1=0.0;
  for(unsigned int i=0; i<X.getRows(); i++)
    for(unsigned int j=0; j<X.getRows(); j++)
    {
      dot = X.dotRowRow(i, X, j);
      g1 += dot*covGrad.getVal(i, j);
    }
  return g1;
}
  

CMlpKern::CMlpKern() : CKern()
{
  _init();
}
CMlpKern::CMlpKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CMlpKern::CMlpKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CMlpKern::CMlpKern(const CMlpKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  weightVariance = kern.weightVariance;
  biasVariance = kern.biasVariance;
}
// Class destructor
CMlpKern::~CMlpKern()
{
}
double CMlpKern::getVariance() const
{
  return variance;
}
void CMlpKern::_init()
{
  nParams = 3;
  setType("mlp");
  setName("MLP");
  setParamName("weightVariance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("biasVariance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setParamName("variance", 2);
  addTransform(CTransform::defaultPositive(), 2);
  setStationary(false);
}
void CMlpKern::setInitParam()
{
  weightVariance = 10.0;
  biasVariance = 10.0;
  variance = 1.0;  
}

inline double CMlpKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  double numer=weightVariance*X.norm2Row(index)+biasVariance;  
  double denom = numer+1.0;
  return variance*asin(numer/denom);
}
// Parameters are kernel parameters
void CMlpKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    weightVariance = val;
    break;
  case 1:
    biasVariance = val;
    break;
  case 2:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CMlpKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return weightVariance;
    break;
  case 1:
    return biasVariance;
    break;
  case 2:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}

void CMlpKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double denomPart1 = weightVariance*X.norm2Row(row) + biasVariance + 1.0;
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double numer=weightVariance*X.dotRowRow(row, X2, k) + biasVariance;
    double denomPart2=weightVariance*X2.norm2Row(k) + biasVariance + 1.0;
    double denom=sqrt(denomPart1*denomPart2);
    double arg = numer/denom;
    double kval=variance*weightVariance/sqrt(1-arg*arg);
    
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(k, j);
      val += (X2.getVal(k, j)/denom-X.getVal(row, j)*denomPart2*numer/(denom*denom*denom))*kval;
      gX.setVal(val, k, j);
    }
  } 
}
void CMlpKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double numer=weightVariance*X.norm2Row(i) + biasVariance;
    double denom=numer+1.0;
    double arg=numer/denom;
    double kval=variance*weightVariance/sqrt(1-arg*arg);
    for(unsigned int j=0; j<X.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(i, j);
      val += 2*(1/denom - numer/(denom*denom))*kval*X.getVal(i, j);
      gX.setVal(val, i, j);
    }
  }
}
double CMlpKern::getWhite() const
{
  return 0.0;
}

double CMlpKern::computeElement(const CMatrix& X1, unsigned int index1, 
				const CMatrix& X2, unsigned int index2) const
{
  double numer= weightVariance*X1.dotRowRow(index1, X2, index2) + biasVariance;
  double denom1=weightVariance*X1.norm2Row(index1)+biasVariance+1.0;  
  double denom2=weightVariance*X2.norm2Row(index2)+biasVariance+1.0;  
  return variance*asin(numer/sqrt(denom1*denom2));
}
void CMlpKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());
  innerProdj.resize(1, X2.getRows());
  for(unsigned int j=0; j<X2.getRows(); j++)
    innerProdj.setVal(X2.norm2Row(j), j);
  // do off diagonal gradients first.
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    innerProdi.setVal(X.norm2Row(i), i);
    for(unsigned int j=0; j<X2.getRows(); j++)
    {	  
      double crossProd=X.dotRowRow(i, X2, j);
      double numer=weightVariance*crossProd + biasVariance;
      double denomi=weightVariance*innerProdi.getVal(i)+biasVariance+1.0;  
      double denomj=weightVariance*innerProdj.getVal(j)+biasVariance+1.0;  
      double denom = sqrt(denomi*denomj);
      double arg = numer/denom;
      double baseCovGrad = variance/sqrt(1-arg*arg)*covGrad.getVal(i, j);
      double denom3=denom*denom*denom;
      g1+=baseCovGrad*(crossProd/denom-0.5*numer/denom3*(denomi*innerProdj.getVal(j) + innerProdi.getVal(i)*denomj));
      g2+=baseCovGrad*(1.0/denom-0.5*numer/denom3*((innerProdi.getVal(i)+innerProdj.getVal(j))*weightVariance +2.0*biasVariance+2.0));
      g3+=asin(arg)*covGrad.getVal(i, j);
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  if(regularise)
    addPriorGrad(g);
  
}
void CMlpKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());

  // do off diagonal gradients first.
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    innerProdi.setVal(X.norm2Row(i), i);
    for(unsigned int j=0; j<i; j++)
    {	  
      double crossProd=X.dotRowRow(i, X, j);
      double numer=weightVariance*crossProd + biasVariance;
      double denomi=weightVariance*innerProdi.getVal(i)+biasVariance+1.0;  
      double denomj=weightVariance*innerProdi.getVal(j)+biasVariance+1.0;  
      double denom = sqrt(denomi*denomj);
      double arg = numer/denom;
      double baseCovGrad = variance/sqrt(1-arg*arg)*covGrad.getVal(i, j);
      double denom3=denom*denom*denom;
      g1+=baseCovGrad*(crossProd/denom-0.5*numer/denom3*(denomi*innerProdi.getVal(j) + innerProdi.getVal(i)*denomj));
      g2+=baseCovGrad*(1.0/denom-0.5*numer/denom3*((innerProdi.getVal(i)+innerProdi.getVal(j))*weightVariance +2.0*biasVariance+2.0));
      g3+=asin(arg)*covGrad.getVal(i, j);
    }
  }
  // double the result due to symmetry.
  g1*=2;
  g2*=2;
  g3*=2;
  // add effect of diagonals.
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double numer = weightVariance*innerProdi.getVal(i)+biasVariance;
    double denom = numer+1.0;
    double denom3=denom*denom*denom;
    double arg = numer/denom;
    double baseCovGrad = variance/sqrt(1-arg*arg)*covGrad.getVal(i, i);
    g1+=baseCovGrad*(innerProdi.getVal(i)/denom-0.5*numer/denom3*(2*denom*innerProdi.getVal(i)));
      
    g2+=baseCovGrad*(1.0/denom-0.5*numer/denom3*(2.0*weightVariance*innerProdi.getVal(i)+2.0*biasVariance+2.0));
    g3+=asin(arg)*covGrad.getVal(i, i);

  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  if(regularise)
    addPriorGrad(g);

}
double CMlpKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CMlpKern");
  
}
double CMlpKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CMlpKern");
  
}

// ---- CIP-0028 Phase 1: MLP weight-space inducing (√p Gaussian product) ----
namespace {

const double WS_FD_EPS = 1e-6;

// φ = sqrt(π variance / 2) erf(wᵀx̃ / √2) recovers MLP asin (no 2/π).
// K_fu = ∫ φ √p ψ with ψ L²-normalised Gaussian windows on weight space.
// sigmaDiag length D=d+1: isotropic uses (α,...,α,β); ARD uses (α s_0,...,α s_{d-1},β).
inline double mlpWsKfuElementDiag(double variance, const vector<double>& sigmaDiag,
                                  const CMatrix& X, unsigned int n,
                                  const CMatrix& W, unsigned int m, double s)
{
  const unsigned int d = X.getCols();
  const unsigned int D = d + 1;
  BOUNDCHECK(W.getCols() == D);
  BOUNDCHECK(sigmaDiag.size() == D);
  BOUNDCHECK(s > 0.0 && variance >= 0.0);

  const double s2 = s * s;
  const double invS2 = 1.0 / s2;
  double logDetSeff = 0.0;
  double quadW = 0.0;
  double quadMuLamW = 0.0;
  double logAbsSigma = 0.0;
  vector<double> mu(D);
  vector<double> Sdiag(D);
  for(unsigned int j = 0; j < D; j++)
  {
    const double sig2 = sigmaDiag[j];
    BOUNDCHECK(sig2 > 0.0);
    const double lamP = 1.0 / (2.0 * sig2);
    const double lamEff = invS2 + lamP;
    Sdiag[j] = 1.0 / lamEff;
    const double wh = W.getVal(m, j);
    mu[j] = Sdiag[j] * invS2 * wh;
    logDetSeff += log(Sdiag[j]);
    logAbsSigma += log(sig2);
    quadW += wh * wh * invS2;
    quadMuLamW += mu[j] * wh * invS2;
  }

  const double logCsqrtP = -0.25 * (double)D * log(2.0 * M_PI) - 0.25 * logAbsSigma;
  const double logCpsi = 0.25 * (double)D * log(4.0 * M_PI * s2)
                         - 0.5 * (double)D * log(2.0 * M_PI * s2);
  const double logInt = logCsqrtP + logCpsi
                        + 0.5 * (double)D * log(2.0 * M_PI)
                        + 0.5 * logDetSeff
                        + (-0.5 * quadW + 0.5 * quadMuLamW);
  const double intSqrtPPsi = exp(logInt);

  double muDot = mu[d];
  double xSx = Sdiag[d];
  for(unsigned int j = 0; j < d; j++)
  {
    const double xj = X.getVal(n, j);
    muDot += mu[j] * xj;
    xSx += Sdiag[j] * xj * xj;
  }
  const double erfArg = (muDot / sqrt(2.0)) / sqrt(1.0 + xSx);
  const double erfMean = erf(erfArg);
  const double B = sqrt(variance * M_PI / 2.0);
  return B * intSqrtPPsi * erfMean;
}

inline double mlpWsKfuElement(double variance, double weightVar, double biasVar,
                              const CMatrix& X, unsigned int n,
                              const CMatrix& W, unsigned int m, double s)
{
  const unsigned int d = X.getCols();
  vector<double> sigmaDiag(d + 1, weightVar);
  sigmaDiag[d] = biasVar;
  return mlpWsKfuElementDiag(variance, sigmaDiag, X, n, W, m, s);
}

void wsFdGradParams(CKern& kern, CMatrix& g, const CMatrix& X, const CMatrix& W,
                    double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = W.getRows();
  const unsigned int nParams = kern.getNumParams();
  DIMENSIONMATCH(g.getCols() == nParams);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  g.zeros();
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  for(unsigned int p = 0; p < nParams; p++)
  {
    const double p0 = kern.getParam(p);
    const double h = WS_FD_EPS * max(1.0, fabs(p0));
    // Keep positive / interior for MLP weight/bias/scales (and ARD inputScale in (0,1)).
    const double pPlus = min(p0 + h, (p0 > 0.0 && p0 < 1.0) ? 0.999999 : p0 + h);
    const double pMinus = max(p0 - h, 1e-12);
    kern.setParam(pPlus, p);
    kern.wsCrossCompute(Kp, X, W, s);
    kern.setParam(pMinus, p);
    kern.wsCrossCompute(Km, X, W, s);
    kern.setParam(p0, p);
    const double denom = pPlus - pMinus;
    double gp = 0.0;
    if(denom > 0.0)
    {
      for(unsigned int n = 0; n < N; n++)
        for(unsigned int m = 0; m < M; m++)
          gp += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / denom;
    }
    g.setVal(gp, p);
  }
}

void wsFdGradW(const CKern& kern, CMatrix& gW, const CMatrix& X, CMatrix& W,
               double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = W.getRows();
  const unsigned int D = W.getCols();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gW.resize(M, D);
  gW.zeros();
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  for(unsigned int m = 0; m < M; m++)
  {
    for(unsigned int j = 0; j < D; j++)
    {
      const double w0 = W.getVal(m, j);
      const double h = WS_FD_EPS * max(1.0, fabs(w0));
      W.setVal(w0 + h, m, j);
      kern.wsCrossCompute(Kp, X, W, s);
      W.setVal(w0 - h, m, j);
      kern.wsCrossCompute(Km, X, W, s);
      W.setVal(w0, m, j);
      double acc = 0.0;
      for(unsigned int n = 0; n < N; n++)
        acc += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
      gW.setVal(acc, m, j);
    }
  }
}

void wsFdGradX(const CKern& kern, CMatrix& gX, CMatrix& X, const CMatrix& W,
               double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = W.getRows();
  const unsigned int d = X.getCols();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gX.resize(N, d);
  gX.zeros();
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int j = 0; j < d; j++)
    {
      const double x0 = X.getVal(n, j);
      const double h = WS_FD_EPS * max(1.0, fabs(x0));
      X.setVal(x0 + h, n, j);
      kern.wsCrossCompute(Kp, X, W, s);
      X.setVal(x0 - h, n, j);
      kern.wsCrossCompute(Km, X, W, s);
      X.setVal(x0, n, j);
      double acc = 0.0;
      for(unsigned int m = 0; m < M; m++)
        acc += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
      gX.setVal(acc, n, j);
    }
  }
}

void wsFdGradS(const CKern& kern, CMatrix& gS, const CMatrix& X, const CMatrix& W,
               double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = W.getRows();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gS.resize(1, 1);
  const double h = WS_FD_EPS * max(1.0, fabs(s));
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  kern.wsCrossCompute(Kp, X, W, s + h);
  kern.wsCrossCompute(Km, X, W, max(s - h, 1e-12));
  double acc = 0.0;
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      acc += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
  gS.setVal(acc, 0);
}

} // namespace

void CMlpKern::wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = W.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(X.getCols() == d);
  DIMENSIONMATCH(W.getCols() == d + 1);
  BOUNDCHECK(s > 0.0);
  Kfu.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      Kfu.setVal(mlpWsKfuElement(variance, weightVariance, biasVariance, X, n, W, m, s), n, m);
}

void CMlpardKern::wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = W.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(X.getCols() == d);
  DIMENSIONMATCH(W.getCols() == d + 1);
  BOUNDCHECK(s > 0.0);
  vector<double> sigmaDiag(d + 1);
  for(unsigned int j = 0; j < d; j++)
    sigmaDiag[j] = weightVariance * scales.getVal(j);
  sigmaDiag[d] = biasVariance;
  Kfu.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      Kfu.setVal(mlpWsKfuElementDiag(variance, sigmaDiag, X, n, W, m, s), n, m);
}

void CMlpardKern::wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                               const CMatrix& gKfu) const
{
  wsFdGradParams(const_cast<CMlpardKern&>(*this), g, X, W, s, gKfu);
}

void CMlpardKern::wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
                          const CMatrix& gKfu) const
{
  CMatrix Wmut(W);
  wsFdGradW(*this, gW, X, Wmut, s, gKfu);
}

void CMlpardKern::wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
                          const CMatrix& gKfu) const
{
  CMatrix Xmut(X);
  wsFdGradX(*this, gX, Xmut, W, s, gKfu);
}

void CMlpardKern::wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
                          const CMatrix& gKfu) const
{
  wsFdGradS(*this, gS, X, W, s, gKfu);
}

void CMlpKern::wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                            const CMatrix& gKfu) const
{
  // const_cast: FD temporarily mutates params then restores (same pattern as greens FD).
  wsFdGradParams(const_cast<CMlpKern&>(*this), g, X, W, s, gKfu);
}

void CMlpKern::wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
                       const CMatrix& gKfu) const
{
  CMatrix Wmut(W);
  wsFdGradW(*this, gW, X, Wmut, s, gKfu);
}

void CMlpKern::wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
                       const CMatrix& gKfu) const
{
  CMatrix Xmut(X);
  wsFdGradX(*this, gX, Xmut, W, s, gKfu);
}

void CMlpKern::wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
                       const CMatrix& gKfu) const
{
  wsFdGradS(*this, gS, X, W, s, gKfu);
}

// ---- CIP-0028 Phase 2: arc-cosine kernel ----
namespace {

inline void arccosAugmentStats(const CMatrix& X1, unsigned int i1,
                               const CMatrix& X2, unsigned int i2,
                               double& dot, double& n1, double& n2)
{
  const unsigned int d = X1.getCols();
  dot = 1.0; // bias components
  double n1sq = 1.0;
  double n2sq = 1.0;
  for(unsigned int j = 0; j < d; j++)
  {
    const double a = X1.getVal(i1, j);
    const double b = X2.getVal(i2, j);
    dot += a * b;
    n1sq += a * a;
    n2sq += b * b;
  }
  n1 = sqrt(n1sq);
  n2 = sqrt(n2sq);
}

inline double arccosClampCos(double c)
{
  if(c > 1.0) return 1.0;
  if(c < -1.0) return -1.0;
  return c;
}

inline double arccosNormPdf(double z)
{
  return exp(-0.5 * z * z) / sqrt(2.0 * M_PI);
}

inline double arccosNormCdf(double z)
{
  return 0.5 * (1.0 + erf(z / sqrt(2.0)));
}

inline double arccos0WsElement(double variance, const CMatrix& X, unsigned int n,
                               const CMatrix& W, unsigned int m, double s)
{
  const unsigned int d = X.getCols();
  const unsigned int D = d + 1;
  double n2 = 1.0;
  double muDot = W.getVal(m, d); // bias weight × 1
  for(unsigned int j = 0; j < d; j++)
  {
    const double xj = X.getVal(n, j);
    n2 += xj * xj;
    muDot += W.getVal(m, j) * xj;
  }
  const double psiScale = pow(4.0 * M_PI * s * s, 0.25 * (double)D);
  const double B = sqrt(variance * M_PI / 2.0);
  const double erfArg = (muDot / sqrt(2.0)) / sqrt(1.0 + s * s * n2);
  return B * psiScale * erf(erfArg);
}

inline double arccos1WsElement(double variance, const CMatrix& X, unsigned int n,
                               const CMatrix& W, unsigned int m, double s)
{
  const unsigned int d = X.getCols();
  const unsigned int D = d + 1;
  double n2 = 1.0;
  double mu = W.getVal(m, d);
  for(unsigned int j = 0; j < d; j++)
  {
    const double xj = X.getVal(n, j);
    n2 += xj * xj;
    mu += W.getVal(m, j) * xj;
  }
  const double nx = sqrt(n2);
  const double psiScale = pow(4.0 * M_PI * s * s, 0.25 * (double)D);
  const double B = sqrt(variance);
  const double sig = s * nx;
  double soft = 0.0;
  if(sig < 1e-14)
    soft = (mu > 0.0) ? mu : 0.0;
  else
  {
    const double z = mu / sig;
    soft = mu * arccosNormCdf(z) + sig * arccosNormPdf(z);
  }
  return B * psiScale * soft;
}

} // namespace

CArcCosineKern::CArcCosineKern() : CKern(), order(0)
{
  _init();
}
CArcCosineKern::CArcCosineKern(unsigned int inDim) : CKern(inDim), order(0)
{
  _init();
  setInputDim(inDim);
}
CArcCosineKern::CArcCosineKern(const CMatrix& X) : CKern(X), order(0)
{
  _init();
  setInputDim(X.getCols());
}
CArcCosineKern::CArcCosineKern(const CArcCosineKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  order = kern.order;
}
CArcCosineKern::~CArcCosineKern()
{
}
double CArcCosineKern::getVariance() const
{
  return variance;
}
void CArcCosineKern::setOrder(unsigned int ord)
{
  if(ord > 1)
    throw ndlexceptions::Error("CArcCosineKern order must be 0 or 1.");
  order = ord;
}
void CArcCosineKern::_init()
{
  nParams = 1;
  setType("arccos");
  setName("ArcCosine");
  setParamName("variance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setStationary(false);
  order = 0;
}
void CArcCosineKern::setInitParam()
{
  variance = 1.0;
}
void CArcCosineKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  variance = val;
}
double CArcCosineKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  return variance;
}
double CArcCosineKern::getWhite() const
{
  return 0.0;
}

double CArcCosineKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  if(order == 0)
    return variance * (M_PI / 2.0); // asin(1)
  // order 1: σ² ||x̃||²
  double n2 = 1.0;
  for(unsigned int j = 0; j < X.getCols(); j++)
  {
    const double x = X.getVal(index, j);
    n2 += x * x;
  }
  return variance * n2;
}

double CArcCosineKern::computeElement(const CMatrix& X1, unsigned int index1,
                                      const CMatrix& X2, unsigned int index2) const
{
  double dot = 0.0, n1 = 0.0, n2 = 0.0;
  arccosAugmentStats(X1, index1, X2, index2, dot, n1, n2);
  if(!(n1 > 0.0) || !(n2 > 0.0))
    return 0.0;
  const double c = arccosClampCos(dot / (n1 * n2));
  const double theta = acos(c);
  if(order == 0)
    return variance * asin(c); // Williams infinite-MLP angular form
  // Cho & Saul order 1: (σ²/π) ||x||||y|| (sinθ + (π-θ)cosθ)
  const double J1 = sin(theta) + (M_PI - theta) * c;
  return variance * (n1 * n2 / M_PI) * J1;
}

void CArcCosineKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2,
                                   const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getCols() == nParams);
  double g0 = 0.0;
  for(unsigned int i = 0; i < X.getRows(); i++)
    for(unsigned int j = 0; j < X2.getRows(); j++)
    {
      const double k = computeElement(X, i, X2, j);
      if(variance > 0.0)
        g0 += (k / variance) * covGrad.getVal(i, j);
    }
  g.setVal(g0, 0);
  if(regularise)
    addPriorGrad(g);
}

void CArcCosineKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad,
                                   bool regularise) const
{
  DIMENSIONMATCH(g.getCols() == nParams);
  double g0 = 0.0;
  for(unsigned int i = 0; i < X.getRows(); i++)
  {
    for(unsigned int j = 0; j < i; j++)
    {
      const double k = computeElement(X, i, X, j);
      if(variance > 0.0)
        g0 += 2.0 * (k / variance) * covGrad.getVal(i, j);
    }
    const double kd = diagComputeElement(X, i);
    if(variance > 0.0)
      g0 += (kd / variance) * covGrad.getVal(i, i);
  }
  g.setVal(g0, 0);
  if(regularise)
    addPriorGrad(g);
}

double CArcCosineKern::getGradParam(unsigned int /*index*/, const CMatrix& /*X*/,
                                    const CMatrix& /*X2*/, const CMatrix& /*cvGrd*/) const
{
  throw ndlexceptions::NotImplementedError("getGradParam not implemented for CArcCosineKern");
}
double CArcCosineKern::getGradParam(unsigned int /*index*/, const CMatrix& /*X*/,
                                    const CMatrix& /*cvGrd*/) const
{
  throw ndlexceptions::NotImplementedError("getGradParam not implemented for CArcCosineKern");
}

void CArcCosineKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row,
                              const CMatrix& X2, bool addG) const
{
  // FD w.r.t. X row for both orders (keeps order-1 tractable).
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols() == X2.getCols());
  CMatrix Xp(X);
  CMatrix Xm(X);
  const double eps = 1e-6;
  for(unsigned int j = 0; j < X.getCols(); j++)
  {
    const double x0 = X.getVal(row, j);
    const double h = eps * max(1.0, fabs(x0));
    Xp.setVal(x0 + h, row, j);
    Xm.setVal(x0 - h, row, j);
    for(unsigned int k = 0; k < X2.getRows(); k++)
    {
      const double gp = computeElement(Xp, row, X2, k);
      const double gm = computeElement(Xm, row, X2, k);
      double val = 0.0;
      if(addG)
        val = gX.getVal(k, j);
      val += (gp - gm) / (2.0 * h);
      gX.setVal(val, k, j);
    }
    Xp.setVal(x0, row, j);
    Xm.setVal(x0, row, j);
  }
}

void CArcCosineKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  CMatrix Xp(X);
  CMatrix Xm(X);
  const double eps = 1e-6;
  for(unsigned int i = 0; i < X.getRows(); i++)
  {
    for(unsigned int j = 0; j < X.getCols(); j++)
    {
      const double x0 = X.getVal(i, j);
      const double h = eps * max(1.0, fabs(x0));
      Xp.setVal(x0 + h, i, j);
      Xm.setVal(x0 - h, i, j);
      const double gp = diagComputeElement(Xp, i);
      const double gm = diagComputeElement(Xm, i);
      double val = 0.0;
      if(addG)
        val = gX.getVal(i, j);
      val += (gp - gm) / (2.0 * h);
      gX.setVal(val, i, j);
      Xp.setVal(x0, i, j);
      Xm.setVal(x0, i, j);
    }
  }
}

void CArcCosineKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "numParams", getNumParams());
  writeToStream(out, "order", (int)order);
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
  writeToStream(out, "numPriors", getNumPriors());
  writePriorsToStream(out);
}

void CArcCosineKern::readParamsFromStream(istream& in)
{
  setInputDim(readIntFromStream(in, "inputDim"));
  unsigned int numParams = readIntFromStream(in, "numParams");
  setOrder((unsigned int)readIntFromStream(in, "order"));
  CMatrix par(1, numParams);
  par.fromStream(in);
  if(numParams == getNumParams())
    setParams(par);
  else
    throw ndlexceptions::StreamFormatError("numParams",
        "Listed number of parameters does not match computed number of parameters.");
  unsigned int numPriors = readIntFromStream(in, "numPriors");
  readPriorsFromStream(in, numPriors);
}

void CArcCosineKern::wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = W.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(X.getCols() == d);
  DIMENSIONMATCH(W.getCols() == d + 1);
  BOUNDCHECK(s > 0.0);
  Kfu.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      const double val = (order == 0)
        ? arccos0WsElement(variance, X, n, W, m, s)
        : arccos1WsElement(variance, X, n, W, m, s);
      Kfu.setVal(val, n, m);
    }
  }
}

void CArcCosineKern::wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                                  const CMatrix& gKfu) const
{
  wsFdGradParams(const_cast<CArcCosineKern&>(*this), g, X, W, s, gKfu);
}
void CArcCosineKern::wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
                             const CMatrix& gKfu) const
{
  CMatrix Wmut(W);
  wsFdGradW(*this, gW, X, Wmut, s, gKfu);
}
void CArcCosineKern::wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
                             const CMatrix& gKfu) const
{
  CMatrix Xmut(X);
  wsFdGradX(*this, gX, Xmut, W, s, gKfu);
}
void CArcCosineKern::wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
                             const CMatrix& gKfu) const
{
  wsFdGradS(*this, gS, X, W, s, gKfu);
}

CPolyKern::CPolyKern() : CKern()
{
  _init();
}
CPolyKern::CPolyKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CPolyKern::CPolyKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CPolyKern::CPolyKern(const CPolyKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  weightVariance = kern.weightVariance;
  biasVariance = kern.biasVariance;
  degree = kern.degree;
}
// Class destructor
CPolyKern::~CPolyKern()
{
}
void CPolyKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "numParams", getNumParams());
  double deg = getDegree();
  if((deg - (int)deg)==0)
    writeToStream(out, "degree", (int)deg);
  else
    writeToStream(out, "degree", deg);
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
  writeToStream(out, "numPriors", getNumPriors());
  writePriorsToStream(out);
}
void CPolyKern::readParamsFromStream(istream& in)
{
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setInputDim(readIntFromStream(in, "inputDim"));
  unsigned int numParams=readIntFromStream(in, "numParams");
  setDegree(readDoubleFromStream(in, "degree"));

  CMatrix par(1, numParams);
  par.fromStream(in);
  if(numParams==getNumParams())
    setParams(par);
  else
    throw ndlexceptions::StreamFormatError("numParams", "Listed number of parameters does not match computed number of parameters.");
  unsigned int numPriors = readIntFromStream(in, "numPriors");
  readPriorsFromStream(in, numPriors);
}
double CPolyKern::getVariance() const
{
  return variance;
}
void CPolyKern::_init()
{
  nParams = 3;
  setType("poly");
  setName("Polynomial");
  setParamName("weightVariance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("biasVariance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setParamName("variance", 2);
  addTransform(CTransform::defaultPositive(), 2);
  setStationary(false);
}
void CPolyKern::setInitParam()
{
  weightVariance = 1.0;
  biasVariance = 1.0;
  variance = 1.0;
  degree = 2.0;
}

inline double CPolyKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  double arg=weightVariance*X.norm2Row(index)+biasVariance;  
  return variance*pow(arg, degree);
}
// Parameters are kernel parameters
void CPolyKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    weightVariance = val;
    break;
  case 1:
    biasVariance = val;
    break;
  case 2:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CPolyKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return weightVariance;
    break;
  case 1:
    return biasVariance;
    break;
  case 2:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}

void CPolyKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double arg=weightVariance*X.dotRowRow(row, X2, k) + biasVariance;
    double kval=degree*variance*weightVariance*pow(arg, degree-1);
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(k, j);
      val += kval*X2.getVal(k, j);
      gX.setVal(val, k, j);
    }
  }
}
void CPolyKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double arg=weightVariance*X.norm2Row(i) + biasVariance;
    double kval=degree*variance*weightVariance*pow(arg, degree-1);
    for(unsigned int j=0; j<X.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(i, j);
      val += 2*kval*X.getVal(i, j);
      gX.setVal(val, i, j);
    }
  }
}
double CPolyKern::getWhite() const
{
  return 0.0;
}

double CPolyKern::computeElement(const CMatrix& X1, unsigned int index1, 
				 const CMatrix& X2, unsigned int index2) const
{
  double arg=weightVariance*X1.dotRowRow(index1, X2, index2) + biasVariance;
  return variance*pow(arg, degree);
}
void CPolyKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());
  // do off diagonal gradients first.
  for(unsigned int i=0; i<X.getRows(); i++) {
    innerProdi.setVal(X.norm2Row(i),i);
    for(unsigned int j=0; j<X2.getRows(); j++) {
      double crossProd=X.dotRowRow(i, X2, j);
      double arg=weightVariance*crossProd + biasVariance;
      double base = variance*degree*pow(arg, degree-1)*covGrad.getVal(i, j);	  
      g1+=crossProd*base;
      g2+=base;
      g3+=pow(arg, degree)*covGrad.getVal(i, j);
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  if(regularise)
    addPriorGrad(g);

}
void CPolyKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());
  // do off diagonal gradients first.
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    innerProdi.setVal(X.norm2Row(i),i);
    for(unsigned int j=0; j<i; j++)
    {
	  
      double crossProd=X.dotRowRow(i, X, j);
      double arg=weightVariance*crossProd + biasVariance;
      double base = variance*degree*pow(arg, degree-1)*covGrad.getVal(i, j);	  
      g1+=crossProd*base;
      g2+=base;
      g3+=pow(arg, degree)*covGrad.getVal(i, j);
    }
  }
  // double the result due to symmetry.
  g1*=2;
  g2*=2;
  g3*=2;
  // add effect of diagonals.
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double arg = weightVariance*innerProdi.getVal(i)+biasVariance;
    double base = variance*degree*pow(arg, degree-1)*covGrad.getVal(i, i);	  
    g1+=innerProdi.getVal(i)*base;
    g2+=base;
    g3+=pow(arg,degree)*covGrad.getVal(i, i);

  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  if(regularise)
    addPriorGrad(g);

}
double CPolyKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CPolyKern");
  
}
double CPolyKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CPolyKern");
  
}
#ifdef _NDLMATLAB
void CPolyKern::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "degree");
  mxSetField(matlabArray, 0, "degree", convertMxArray(degree));
  CKern::addParamToMxArray(matlabArray);
}
void CPolyKern::extractParamFromMxArray(const mxArray* matlabArray)
{
  degree = mxArrayExtractIntField(matlabArray, "degree");
  CKern::extractParamFromMxArray(matlabArray);
}
#endif

// the Linear ARD kernel.
CLinardKern::CLinardKern() : CArdKern()
{
  _init();
}
CLinardKern::CLinardKern(unsigned int inDim) : CArdKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CLinardKern::CLinardKern(const CMatrix& X) : CArdKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CLinardKern::CLinardKern(const CLinardKern& kern) : CArdKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  scales = kern.scales;
}
// Class destructor
CLinardKern::~CLinardKern()
{
}
double CLinardKern::getVariance() const
{
  return variance;
}
void CLinardKern::_init()
{
  nParams = 1;
  setType("linard");
  setName("linear ARD");
  setParamName("variance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setStationary(false);
}
void CLinardKern::setInitParam()
{ 
  nParams = 1+getInputDim();
  variance = 1.0;
  scales.resize(1, getInputDim());
  for(unsigned int i=1; i<getInputDim()+1; i++)
  {
    string name = "inputScale";
    setParamName(name, i);
  }
  scales.setVals(0.5);
  for(unsigned int i=1; i<getInputDim()+1; i++)
  {
    addTransform(CTransform::defaultZeroOne(), i);
  }
}

double CLinardKern::diagComputeElement(const CMatrix& X, unsigned int index1) const
{
  double val = 0.0;
  for(unsigned int i=0; i<getInputDim(); i++)
  {
    double x=X.getVal(index1, i);
    val += x*x*scales.getVal(i);
  }
  return val*variance;
}
// Parameters are kernel parameters
void CLinardKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    variance = val;
    break;
  default:
    if(paramNo<getInputDim()+1)
      scales.setVal(val, paramNo-1);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
double CLinardKern::getParam(unsigned int paramNo) const
{
  
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    return variance;
    break;
  default:
    if(paramNo<getInputDim()+1)
      return scales.getVal(paramNo-1);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
void CLinardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(k, j);
      val += variance*X2.getVal(k, j)*scales.getVal(j);
      gX.setVal(val, k, j);
    }
  }
}
void CLinardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
  {
    gX.deepCopy(X);
    double baseScale = 2.0*variance;
    for(unsigned int j=0; j<gX.getCols(); j++)
      gX.scaleCol(j, baseScale*scales.getVal(j));
  }
  else
  {
    double baseScale = 2.0*variance;
    for(unsigned int j=0; j<gX.getCols(); j++)
      gX.axpyColCol(j, X, j, baseScale*scales.getVal(j));
  }
}
double CLinardKern::getWhite() const
{
  return 0.0;
}

double CLinardKern::computeElement(const CMatrix& X1, unsigned int index1, 
				   const CMatrix& X2, unsigned int index2) const
{
  double val = 0.0;
  for(unsigned int i=0; i<getInputDim(); i++)
  {
    val += X1.getVal(index1, i)*X2.getVal(index2, i)*scales.getVal(i);
  }
  return val*variance;
}
void CLinardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  for(unsigned int i=0; i<X.getRows(); i++) {
    for(unsigned int j=0; j<X2.getRows(); j++) {
      double val=0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
	val += X.getVal(i, k)*X2.getVal(j, k)*scales.getVal(k);
      g1+=covGrad.getVal(i, j)*val;
    }
  }
  g.setVal(g1, 0);
  for(unsigned int k=0; k<getInputDim(); k++) {
    double g2=0.0;
    for(unsigned int i=0; i<X.getRows(); i++) {
      for(unsigned int j=0; j<X2.getRows(); j++) {
	g2+=X.getVal(i, k)*X2.getVal(j, k)*covGrad.getVal(i, j);
      }
    }
    g2*=variance;
    g.setVal(g2, k+1);
  }
  if(regularise)
    addPriorGrad(g);

}
void CLinardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    for(unsigned int j=0; j<i; j++)
    {
      double val=0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
	val += X.getVal(i, k)*X.getVal(j, k)*scales.getVal(k);
      g1+=covGrad.getVal(i, j)*val;
    }
  }
  g1*=2.0;
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double val=0.0;
    for(unsigned int k=0; k<getInputDim(); k++)
      val += X.getVal(i, k)*X.getVal(i, k)*scales.getVal(k);
    g1+=covGrad.getVal(i, i)*val;
  }
  g.setVal(g1, 0);
  for(unsigned int k=0; k<getInputDim(); k++)
  {
    double g2=0.0;
    for(unsigned int i=0; i<X.getRows(); i++)
    {
      for(unsigned int j=0; j<i; j++)
      {
	g2+=X.getVal(i, k)*X.getVal(j, k)*covGrad.getVal(i, j);
      }
    }
    g2*=2;
    for(unsigned int i=0; i<X.getRows(); i++)
      g2+=X.getVal(i, k)*X.getVal(i, k)*covGrad.getVal(i, i);
    g2*=variance;
    g.setVal(g2, k+1);
  }
  if(regularise)
    addPriorGrad(g);

}
double CLinardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CLinardKern");
  
}
double CLinardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CLinardKern");
  
}
// the RBF ARD kernel.
CRbfardKern::CRbfardKern() : CArdKern()
{
  _init();
}
CRbfardKern::CRbfardKern(unsigned int inDim) : CArdKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CRbfardKern::CRbfardKern(const CMatrix& X) : CArdKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CRbfardKern::CRbfardKern(const CRbfardKern& kern) : CArdKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  inverseWidth = kern.inverseWidth;
  scales = kern.scales;
}
// Class destructor
CRbfardKern::~CRbfardKern()
{
}
double CRbfardKern::getVariance() const
{
  return variance;
}
void CRbfardKern::_init()
{
  nParams = 2;
  setType("rbfard");
  setName("RBF ARD");
  setParamName("inverseWidth", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}
void CRbfardKern::setInitParam()
{
  nParams = 2+getInputDim();
  inverseWidth=1.0;
  variance = 1.0;

  // input scales.
  scales.resize(1, getInputDim());
  gscales.resize(1, getInputDim());
  scales.setVals(0.5);
  for(unsigned int i=2; i<getInputDim()+2; i++)
  {
    string name = "inputScale";
    setParamName(name, i);
  }
  for(unsigned int i=2; i<getInputDim()+2; i++)
  {
    addTransform(CTransform::defaultZeroOne(), i);
  }

}

double CRbfardKern::diagComputeElement(const CMatrix& X, unsigned int index1) const
{
  return variance;
}
// Parameters are kernel parameters
void CRbfardKern::setParam(double val, unsigned int paramNo)
{
  
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    inverseWidth=val;
    break;
  case 1:
    variance=val;
    break;
  default:
    if(paramNo<nParams)
      scales.setVal(val, paramNo-2);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }    
  }
}
double CRbfardKern::getParam(unsigned int paramNo) const
{
  
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    return inverseWidth;
    break;
  case 1:
    return variance;
    break;
  default:
    if(paramNo<nParams)
      return scales.getVal(paramNo-2);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
void CRbfardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2 = 0.5*inverseWidth;
  double pf = variance*inverseWidth;
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double n2=0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      double x = X.getVal(row, j);
      x = x-X2.getVal(k, j);
      n2+=x*scales.getVal(j)*x;
    }
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(k, j);
      val += pf*(X2.getVal(k, j)-X.getVal(row, j))*exp(-n2*wi2)*scales.getVal(j);
      gX.setVal(val, k, j);	      
    }
  }
}
void CRbfardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CRbfardKern::getWhite() const
{
  return 0.0;
}

double CRbfardKern::computeElement(const CMatrix& X1, unsigned int index1, 
				   const CMatrix& X2, unsigned int index2) const
{
  double val = 0.0;
  for(unsigned int i=0; i<getInputDim(); i++)
  {
    double x = X1.getVal(index1, i);
    x = x-X2.getVal(index2, i);
    val+=x*scales.getVal(i)*x;
  }
  return variance*exp(-val*inverseWidth*0.5);
}

void CRbfardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  gscales.zeros();
  double halfInverseWidth = 0.5*inverseWidth;
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    for(unsigned int j=0; j<X2.getRows(); j++)
    {
      double val = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double x = X.getVal(i, k);
	x-=X2.getVal(j, k);
	val+=x*scales.getVal(k)*x;
      }
      double kCovGrad = exp(-halfInverseWidth*val)*covGrad.getVal(i, j);
      g1-=0.5*val*kCovGrad*variance;
      g2+=kCovGrad;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double g3=gscales.getVal(k);
	double xi=X.getVal(i, k);
	double xj=X2.getVal(j, k);
	g3+=inverseWidth*kCovGrad*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
	gscales.setVal(g3, k);
      }
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k), k+2);
  if(regularise)
    addPriorGrad(g);

}

void CRbfardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  gscales.zeros();
  double halfInverseWidth = 0.5*inverseWidth;
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    for(unsigned int j=0; j<i; j++)
    {
      double val = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double x = X.getVal(i, k);
	x-=X.getVal(j, k);
	val+=x*scales.getVal(k)*x;
      }
      double kCovGrad = exp(-halfInverseWidth*val)*covGrad.getVal(i, j);
      g1-=0.5*val*kCovGrad*variance;
      g2+=kCovGrad;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double g3=gscales.getVal(k);
	double xi=X.getVal(i, k);
	double xj=X.getVal(j, k);
	g3+=inverseWidth*kCovGrad*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
	gscales.setVal(g3, k);
      }
    }
  }
  g1*=2.0;
  g2*=2.0;
  for(unsigned int i=0; i<X.getRows(); i++)
    g2+=covGrad.getVal(i, i);
  gscales.scale(2.0);
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k), k+2);
  if(regularise)
    addPriorGrad(g);

}
double CRbfardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfardKern");
  
}
double CRbfardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfardKern");
  
}

// the MLP ARD kernel.
CMlpardKern::CMlpardKern() : CArdKern()
{
  _init();
}
CMlpardKern::CMlpardKern(unsigned int inDim) : CArdKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CMlpardKern::CMlpardKern(const CMatrix& X) : CArdKern(X) 
{
  _init();
  setInputDim(X.getCols());
}  
CMlpardKern::CMlpardKern(const CMlpardKern& kern) : CArdKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  weightVariance=kern.weightVariance;
  biasVariance=kern.biasVariance;
  scales = kern.scales;
}
// Class destructor
CMlpardKern::~CMlpardKern()
{
}
double CMlpardKern::getVariance() const
{
  return variance;
}
void CMlpardKern::_init()
{
  nParams = 3;
  setType("mlpard");
  setName("MLP ARD");
  setParamName("weightVariance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("biasVariance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setParamName("variance", 2);
  addTransform(CTransform::defaultPositive(), 2);
  setStationary(false);

}
void CMlpardKern::setInitParam()
{
  nParams = 3+getInputDim();
  weightVariance=10.0;
  biasVariance=10.0;
  variance = 1.0;
  scales.resize(1, getInputDim());
  gscales.resize(1, getInputDim());
  scales.setVals(0.5);
  for(unsigned int i=3; i<getInputDim()+3; i++)
  {
    string name = "inputScale";
    setParamName(name, i);
  }
  for(unsigned int i=3; i<getInputDim()+3; i++)
  {
    addTransform(CTransform::defaultZeroOne(), i);
  }
}

double CMlpardKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  double val=0.0;
  for(unsigned int i=0; i<getInputDim(); i++)
  {
    double x = X.getVal(index, i);
    val+=x*x*scales.getVal(i);
  }
  double numer=weightVariance*val+biasVariance;  
  double denom = numer+1.0;
  return variance*asin(numer/denom);
}
// Parameters are kernel parameters
void CMlpardKern::setParam(double val, unsigned int paramNo)
{
  
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    weightVariance=val;
    break;
  case 1:
    biasVariance=val;
    break;
  case 2:
    variance=val;
    break;
  default:
    if(paramNo<nParams)
      scales.setVal(val, paramNo-3);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
double CMlpardKern::getParam(unsigned int paramNo) const
{
  
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    return weightVariance;
    break;
  case 1:
    return biasVariance;
    break;
  case 2:
    return variance;
    break;
  default:
    if(paramNo<nParams)
      return scales.getVal(paramNo-3);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}

void CMlpardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double val=0.0;
  for(unsigned int j=0; j<getInputDim(); j++)
  {
    double x = X.getVal(row, j);
    val+=x*x*scales.getVal(j);
  }
  double denomPart1=weightVariance*val+biasVariance+1.0;  
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double val=0.0;
    double val2=0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      double xk = X2.getVal(k, j);
      double xi = X.getVal(row, j);
      double scalesX = xk*scales.getVal(j);
      val+=xk*scalesX;
      val2+=xi*scalesX;
    }
    double numer=weightVariance*val2 + biasVariance;
    double denomPart2=weightVariance*val+biasVariance+1.0;  
    double denom=sqrt(denomPart1*denomPart2);
    double arg = numer/denom;
    double kval=variance*weightVariance/sqrt(1-arg*arg);
    
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(k, j);
      val += (X2.getVal(k, j)/denom-X.getVal(row, j)*denomPart2*numer/(denom*denom*denom))*kval*scales.getVal(j);
      gX.setVal(val, k, j);
    }
  }
}
void CMlpardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double val=0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      double x = X.getVal(i, j);
      val+=x*x*scales.getVal(j);
    }
    double numer=weightVariance*val+biasVariance;  
    double denom=numer+1.0;
    double arg=numer/denom;
    double kval=variance*weightVariance/sqrt(1-arg*arg);
    for(unsigned int j=0; j<X.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(i, j);
      val += 2*(1/denom - numer/(denom*denom))*kval*X.getVal(i, j)*scales.getVal(j);
      gX.setVal(val, i, j);
    }
  }
}
double CMlpardKern::getWhite() const
{
  return 0.0;
}

double CMlpardKern::computeElement(const CMatrix& X1, unsigned int index1, 
				   const CMatrix& X2, unsigned int index2) const
{
  double valij=0.0;
  double valii=0.0;
  double valjj=0.0;
  for(unsigned int i=0; i<getInputDim(); i++)
  {
    double xi = X1.getVal(index1, i);
    double xj = X2.getVal(index2, i);
    valij+=xi*xj*scales.getVal(i);
    valii+=xi*xi*scales.getVal(i);
    valjj+=xj*xj*scales.getVal(i);
  }
  double numer=weightVariance*valij + biasVariance;
  double denom1=weightVariance*valii+biasVariance+1.0;  
  double denom2=weightVariance*valjj+biasVariance+1.0;  
  return variance*asin(numer/sqrt(denom1*denom2));
}

void CMlpardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());
  innerProdj.resize(1, X2.getRows());
  gscales.zeros();
  innerProdi.zeros();
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double val=0.0;
    for(unsigned int k=0; k<getInputDim(); k++)
    {
      double x = X.getVal(i, k);
      val+=x*x*scales.getVal(k);
    }
    innerProdi.setVal(val, i);
  }
  for(unsigned int j=0; j<X2.getRows(); j++)
  {
    double val=0.0;
    for(unsigned int k=0; k<getInputDim(); k++)
    {
      double x = X2.getVal(j, k);
      val+=x*x*scales.getVal(k);
    }
    innerProdj.setVal(val, j);
  }
  
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    for(unsigned int j=0; j<X2.getRows(); j++)
    {
      double val=0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double xi = X.getVal(i, k);
	double xj = X2.getVal(j, k);
	val+=xi*xj*scales.getVal(k);
      }
      double numer=weightVariance*val + biasVariance;
      double denomi=weightVariance*innerProdi.getVal(i)+biasVariance+1.0;  
      double denomj=weightVariance*innerProdj.getVal(j)+biasVariance+1.0;  
      double denom = sqrt(denomi*denomj);
      double arg = numer/denom;
      double baseCovGrad = variance/sqrt(1-arg*arg)*covGrad.getVal(i, j);
      double denom3=denom*denom*denom;
      g1+=baseCovGrad*(val/denom-0.5*numer/denom3*(denomi*innerProdj.getVal(j) + innerProdi.getVal(i)*denomj));
      g2+=baseCovGrad*(1.0/denom-0.5*numer/denom3*((innerProdi.getVal(i)+innerProdj.getVal(j))*weightVariance+2.0*biasVariance+2.0));
      g3+=asin(arg)*covGrad.getVal(i, j);


      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double g4=gscales.getVal(k);
	double xik=X.getVal(i, k);
	double xjk=X2.getVal(j, k);
	g4+=(xik*xjk/denom - 0.5*numer/denom3*(xik*xik*denomj + xjk*xjk*denomi))*baseCovGrad*weightVariance;
	gscales.setVal(g4, k);
      }
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k), k+3);
  if(regularise)
    addPriorGrad(g);

}
void CMlpardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());
  gscales.zeros();
  innerProdi.zeros();
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double val=0.0;
    for(unsigned int k=0; k<getInputDim(); k++)
    {
      double x = X.getVal(i, k);
      val+=x*x*scales.getVal(k);
    }
    innerProdi.setVal(val, i);
  }
  
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    for(unsigned int j=0; j<i; j++)
    {
      double val=0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double xi = X.getVal(i, k);
	double xj = X.getVal(j, k);
	val+=xi*xj*scales.getVal(k);
      }
      double numer=weightVariance*val + biasVariance;
      double denomi=weightVariance*innerProdi.getVal(i)+biasVariance+1.0;  
      double denomj=weightVariance*innerProdi.getVal(j)+biasVariance+1.0;  
      double denom = sqrt(denomi*denomj);
      double arg = numer/denom;
      double baseCovGrad = variance/sqrt(1-arg*arg)*covGrad.getVal(i, j);
      double denom3=denom*denom*denom;
      g1+=baseCovGrad*(val/denom-0.5*numer/denom3*(denomi*innerProdi.getVal(j) + innerProdi.getVal(i)*denomj));
      g2+=baseCovGrad*(1.0/denom-0.5*numer/denom3*((innerProdi.getVal(i)+innerProdi.getVal(j))*weightVariance+2.0*biasVariance+2.0));
      g3+=asin(arg)*covGrad.getVal(i, j);


      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double g4=gscales.getVal(k);
	double xik=X.getVal(i, k);
	double xjk=X.getVal(j, k);
	g4+=(xik*xjk/denom - 0.5*numer/denom3*(xik*xik*denomj + xjk*xjk*denomi))*baseCovGrad*weightVariance;
	gscales.setVal(g4, k);
      }
    }
  }
  // double due to symmetriy.
  g1*=2.0;
  g2*=2.0;
  g3*=2.0;
  gscales.scale(2.0);
  // add effect of diagonals.
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double numer = weightVariance*innerProdi.getVal(i)+biasVariance;
    double denom = numer+1.0;
    double denom3=denom*denom*denom;
    double arg = numer/denom;
    double baseCovGrad = variance/sqrt(1-arg*arg)*covGrad.getVal(i, i);
    g1+=baseCovGrad*(innerProdi.getVal(i)/denom-0.5*numer/denom3*(2*denom*innerProdi.getVal(i)));
      
    g2+=baseCovGrad*(1.0/denom-0.5*numer/denom3*(2.0*weightVariance*innerProdi.getVal(i)+2.0*biasVariance+2.0));
    g3+=asin(arg)*covGrad.getVal(i, i);

    for(unsigned int k=0; k<getInputDim(); k++)
    {
      double g4=gscales.getVal(k);
      double xik=X.getVal(i, k);
      g4+=(xik*xik/denom - 0.5*numer/denom3*(2*xik*xik*denom))*baseCovGrad*weightVariance;
      gscales.setVal(g4, k);
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k), k+3);
  if(regularise)
    addPriorGrad(g);

}
double CMlpardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CMlpardKern");
  
}
double CMlpardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CMlpardKern");
  
}

// the POLY ARD kernel.
CPolyardKern::CPolyardKern() : CArdKern()
{
  _init();
}
CPolyardKern::CPolyardKern(unsigned int inDim) : CArdKern(inDim)
{
  _init();
  setInputDim(inDim);
  
}
CPolyardKern::CPolyardKern(const CMatrix& X) : CArdKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CPolyardKern::CPolyardKern(const CPolyardKern& kern) : CArdKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  weightVariance=kern.weightVariance;
  biasVariance=kern.biasVariance;
  scales = kern.scales;
}
// Class destructor
CPolyardKern::~CPolyardKern()
{
}
void CPolyardKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "numParams", getNumParams());
  double deg = getDegree();
  if((deg - (int)deg)==0)
    writeToStream(out, "degree", (int)deg);
  else
    writeToStream(out, "degree", deg);
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
  writeToStream(out, "numPriors", getNumPriors());
  writePriorsToStream(out);



}
void CPolyardKern::readParamsFromStream(istream& in)
{
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setInputDim(readIntFromStream(in, "inputDim"));
  unsigned int numParams=readIntFromStream(in, "numParams");
  setDegree(readDoubleFromStream(in, "degree"));

  CMatrix par(1, numParams);
  par.fromStream(in);
  if(numParams==getNumParams())
    setParams(par);
  else
    throw ndlexceptions::StreamFormatError("numParams", "Listed number of parameters does not match computed number of parameters.");
  unsigned int numPriors = readIntFromStream(in, "numPriors");
  readPriorsFromStream(in, numPriors);

}
double CPolyardKern::getVariance() const
{
  return variance;
}
void CPolyardKern::_init()
{
  nParams = 3;
  setType("polyard");
  setName("Polynomial ARD");
  setParamName("weightVariance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("biasVariance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setParamName("variance", 2);
  addTransform(CTransform::defaultPositive(), 2);
  setStationary(false);
}
void CPolyardKern::setInitParam()
{
  nParams = 3+getInputDim();
  weightVariance=1.0;
  biasVariance=1.0;
  variance = 1.0;
  scales.resize(1, getInputDim());
  gscales.resize(1, getInputDim());
  scales.setVals(0.5);
  for(unsigned int i=3; i<getInputDim()+3; i++)
  {
    string name = "inputScale";
    setParamName(name, i);
  }
  for(unsigned int i=3; i<getInputDim()+3; i++)
  {
    addTransform(CTransform::defaultZeroOne(), i);
  }
  degree = 2.0;
}

double CPolyardKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  double val=0.0;
  for(unsigned int i=0; i<getInputDim(); i++)
  {
    double x = X.getVal(index, i);
    val+=x*x*scales.getVal(i);
  }
  double arg=weightVariance*val+biasVariance;  
  return variance*pow(arg, degree);
}
// Parameters are kernel parameters
void CPolyardKern::setParam(double val, unsigned int paramNo)
{
  
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    weightVariance=val;
    break;
  case 1:
    biasVariance=val;
    break;
  case 2:
    variance=val;
    break;
  default:
    if(paramNo<nParams)
      scales.setVal(val, paramNo-3);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }    
  }
}
double CPolyardKern::getParam(unsigned int paramNo) const
{
  
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    return weightVariance;
    break;
  case 1:
    return biasVariance;
    break;
  case 2:
    return variance;
    break;
  default:
    if(paramNo<nParams)
      return scales.getVal(paramNo-3);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
void CPolyardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double valik=0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      valik+=X.getVal(row, j)*X2.getVal(k, j)*scales.getVal(j);
    }
    double arg=weightVariance*valik + biasVariance;
    double kval=degree*variance*weightVariance*pow(arg, degree-1);
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(k, j);
      val += kval*X2.getVal(k, j)*scales.getVal(j);
      gX.setVal(val, k, j);
    }
  }
}
void CPolyardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double val=0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      double x = X.getVal(i, j);
      val+=x*x*scales.getVal(j);
    }
    double arg=weightVariance*val + biasVariance;
    double kval=degree*variance*weightVariance*pow(arg, degree-1);
    for(unsigned int j=0; j<X.getCols(); j++)
    {
      double val = 0.0;
      if(addG)
	val = gX.getVal(i, j);
      val += 2*kval*X.getVal(i, j)*scales.getVal(j);
      gX.setVal(val, i, j);
    }
  }
}
double CPolyardKern::getWhite() const
{
  return 0.0;
}

double CPolyardKern::computeElement(const CMatrix& X1, unsigned int index1, 
				    const CMatrix& X2, unsigned int index2) const
{
  double valij=0.0;
  for(unsigned int i=0; i<getInputDim(); i++)
  {
    valij+=X1.getVal(index1, i)*X2.getVal(index2, i)*scales.getVal(i);
  }
  double arg=weightVariance*valij + biasVariance;
  return variance*pow(arg, degree);
}

void CPolyardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());
  gscales.zeros();
  
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    for(unsigned int j=0; j<X2.getRows(); j++)
    {
      double val=0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	val+=X.getVal(i, k)*X2.getVal(j, k)*scales.getVal(k);
      }
      double arg=weightVariance*val + biasVariance;
      double baseCovGrad = variance*degree*pow(arg, degree-1)*covGrad.getVal(i, j);
      g1+=baseCovGrad*val;
      g2+=baseCovGrad;
      g3+=pow(arg, degree)*covGrad.getVal(i, j);


      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double g4=gscales.getVal(k);
	g4+=X.getVal(i, k)*X2.getVal(j, k)*baseCovGrad*weightVariance;
	gscales.setVal(g4, k);
      }
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k), k+3);
  if(regularise)
    addPriorGrad(g);

}
void CPolyardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  innerProdi.resize(1, X.getRows());
  gscales.zeros();
  innerProdi.zeros();
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double val=0.0;
    for(unsigned int k=0; k<getInputDim(); k++)
    {
      double x = X.getVal(i, k);
      val+=x*x*scales.getVal(k);
    }
    innerProdi.setVal(val, i);
  }
  
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    for(unsigned int j=0; j<i; j++)
    {
      double val=0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	val+=X.getVal(i, k)*X.getVal(j, k)*scales.getVal(k);
      }
      double arg=weightVariance*val + biasVariance;
      double baseCovGrad = variance*degree*pow(arg, degree-1)*covGrad.getVal(i, j);
      g1+=baseCovGrad*val;
      g2+=baseCovGrad;
      g3+=pow(arg, degree)*covGrad.getVal(i, j);


      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double g4=gscales.getVal(k);
	g4+=X.getVal(i, k)*X.getVal(j, k)*baseCovGrad*weightVariance;
	gscales.setVal(g4, k);
      }
    }
  }
  // double due to symmetry.
  g1*=2.0;
  g2*=2.0;
  g3*=2.0;
  gscales.scale(2.0);
  // add effect of diagonals.
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double arg = weightVariance*innerProdi.getVal(i)+biasVariance;
    double baseCovGrad = variance*degree*pow(arg, degree-1)*covGrad.getVal(i, i);
    g1+=baseCovGrad*innerProdi.getVal(i);
      
    g2+=baseCovGrad;
    g3+=pow(arg, degree)*covGrad.getVal(i, i);

    for(unsigned int k=0; k<getInputDim(); k++)
    {
      double g4=gscales.getVal(k);
      double xik=X.getVal(i, k);
      g4+=xik*xik*baseCovGrad*weightVariance;
      gscales.setVal(g4, k);
    }
  }
  g.setVal(g1, 0);
  g.setVal(g2, 1);
  g.setVal(g3, 2);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k), k+3);
  if(regularise)
    addPriorGrad(g);

}
double CPolyardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CPolyardKern");
  
}
double CPolyardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CPolyardKern");
  
}

// the Matern 3/2 ARD kernel.
CMatern32ardKern::CMatern32ardKern() : CArdKern()
{
  _init();
}
CMatern32ardKern::CMatern32ardKern(unsigned int inDim) : CArdKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CMatern32ardKern::CMatern32ardKern(const CMatrix& X) : CArdKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CMatern32ardKern::CMatern32ardKern(const CMatern32ardKern& kern) : CArdKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  lengthScale = kern.lengthScale;
  scales = kern.scales;
}
// Class destructor
CMatern32ardKern::~CMatern32ardKern()
{
}
double CMatern32ardKern::getVariance() const
{
  return variance;
}
void CMatern32ardKern::_init()
{
  nParams = 2;
  setType("matern32ard");
  setName("Matern dof=3/2 ARD");
  setParamName("lengthScale", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}
void CMatern32ardKern::setInitParam()
{
  nParams = 2+getInputDim();
  lengthScale = 1.0;
  variance = 1.0;

  // input scales.
  scales.resize(1, getInputDim());
  gscales.resize(1, getInputDim());
  scales.setVals(0.5);
  for(unsigned int i=2; i<getInputDim()+2; i++)
  {
    string name = "inputScale";
    setParamName(name, i);
  }
  for(unsigned int i=2; i<getInputDim()+2; i++)
  {
    addTransform(CTransform::defaultZeroOne(), i);
  }
}

inline double CMatern32ardKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
// Parameters are kernel parameters
void CMatern32ardKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    lengthScale=val;
    break;
  case 1:
    variance=val;
    break;
  default:
    if(paramNo<nParams)
      scales.setVal(val, paramNo-2);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }    
  }
}
double CMatern32ardKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    return lengthScale;
    break;
  case 1:
    return variance;
    break;
  default:
    if(paramNo<nParams)
      return scales.getVal(paramNo-2);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
void CMatern32ardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2 = 3.0/(lengthScale*lengthScale);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double r2 = 0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      double dx = X.getVal(row, j)-X2.getVal(k, j);
      r2 += scales.getVal(j)*dx*dx;
    }
    double t = sqrt(r2*wi2);
    double expmt = exp(-t);
    double K = variance*(1+t)*expmt;
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(t != 0.0)
      {
	double dx = X2.getVal(k, j)-X.getVal(row, j);
	val = wi2*scales.getVal(j)*dx/t*(K-variance*expmt);
      }
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  }
}
void CMatern32ardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CMatern32ardKern::getWhite() const
{
  return 0.0;
}

double CMatern32ardKern::computeElement(const CMatrix& X1, unsigned int index1, 
				     const CMatrix& X2, unsigned int index2) const
{
  double r2 = 0.0;
  for(unsigned int j=0; j<getInputDim(); j++)
  {
    double dx = X1.getVal(index1, j)-X2.getVal(index2, j);
    r2 += scales.getVal(j)*dx*dx;
  }
  double wi2 = 3.0/(lengthScale*lengthScale);
  double t = sqrt(r2*wi2);
  return variance*(1+t)*exp(-t);
}

void CMatern32ardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  gscales.zeros();
  double wi2=3.0/(lengthScale*lengthScale);
  unsigned int nrows = X.getRows();
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<X2.getRows(); i++)
    {
      double cgVal = covGrad.getVal(j,i);
      double r2 = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X.getVal(j, k)-X2.getVal(i, k);
	r2 += scales.getVal(k)*dx*dx;
      }
      double t = sqrt(r2*wi2);
      double expmt = exp(-t);
      double k = (1+t)*expmt;
      double ratio;
      if(t==0.0)
	ratio = 1.0;
      else
	ratio = r2/t;
      g1 += cgVal*wi2/lengthScale*ratio*(k-expmt);
      g2 += cgVal*k;
      for(unsigned int kk=0; kk<getInputDim(); kk++)
      {
	double xi = X.getVal(j, kk);
	double xj = X2.getVal(i, kk);
	double g3 = gscales.getVal(kk);
	g3 += wi2*cgVal*expmt*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
	gscales.setVal(g3, kk);
      }
    }
  }
  g.setVal(g1*variance, 0);
  g.setVal(g2, 1);
  for(unsigned int kk=0; kk<getInputDim(); kk++)
    g.setVal(gscales.getVal(kk), kk+2);
  if(regularise)
    addPriorGrad(g);
}

void CMatern32ardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  MATRIXPROPERTIES(covGrad.isSymmetric());
  double g1=0.0;
  double g2=0.0;
  gscales.zeros();
  double wi2=3.0/(lengthScale*lengthScale);
  unsigned int nrows = X.getRows();
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<j; i++)
    {
      double cgVal = covGrad.getVal(j,i);
      double r2 = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X.getVal(j, k)-X.getVal(i, k);
	r2 += scales.getVal(k)*dx*dx;
      }
      double t = sqrt(r2*wi2);
      double expmt = exp(-t);
      double k = (1+t)*expmt;
      double ratio;
      if(t==0.0)
	ratio = 1.0;
      else
	ratio = r2/t;
      g1 += cgVal*wi2/lengthScale*ratio*(k-expmt);
      g2 += cgVal*k;
      for(unsigned int kk=0; kk<getInputDim(); kk++)
      {
	double xi = X.getVal(j, kk);
	double xj = X.getVal(i, kk);
	double g3 = gscales.getVal(kk);
	g3 += wi2*cgVal*expmt*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
	gscales.setVal(g3, kk);
      }
    }
    g2 += 0.5*covGrad.getVal(j, j);
  }
  gscales.scale(2.0);
  g.setVal(2.0*g1*variance, 0);
  g.setVal(2.0*g2, 1);
  for(unsigned int kk=0; kk<getInputDim(); kk++)
    g.setVal(gscales.getVal(kk), kk+2);
  if(regularise)
    addPriorGrad(g);
}

double CMatern32ardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern32ardKern");
}
double CMatern32ardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern32ardKern");
}

// the Matern 5/2 ARD kernel.
CMatern52ardKern::CMatern52ardKern() : CArdKern()
{
  _init();
}
CMatern52ardKern::CMatern52ardKern(unsigned int inDim) : CArdKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CMatern52ardKern::CMatern52ardKern(const CMatrix& X) : CArdKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CMatern52ardKern::CMatern52ardKern(const CMatern52ardKern& kern) : CArdKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  lengthScale = kern.lengthScale;
  scales = kern.scales;
}
// Class destructor
CMatern52ardKern::~CMatern52ardKern()
{
}
double CMatern52ardKern::getVariance() const
{
  return variance;
}
void CMatern52ardKern::_init()
{
  nParams = 2;
  setType("matern52ard");
  setName("Matern dof=5/2 ARD");
  setParamName("lengthScale", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}
void CMatern52ardKern::setInitParam()
{
  nParams = 2+getInputDim();
  lengthScale = 1.0;
  variance = 1.0;

  // input scales.
  scales.resize(1, getInputDim());
  gscales.resize(1, getInputDim());
  scales.setVals(0.5);
  for(unsigned int i=2; i<getInputDim()+2; i++)
  {
    string name = "inputScale";
    setParamName(name, i);
  }
  for(unsigned int i=2; i<getInputDim()+2; i++)
  {
    addTransform(CTransform::defaultZeroOne(), i);
  }
}

inline double CMatern52ardKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
// Parameters are kernel parameters
void CMatern52ardKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    lengthScale=val;
    break;
  case 1:
    variance=val;
    break;
  default:
    if(paramNo<nParams)
      scales.setVal(val, paramNo-2);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }    
  }
}
double CMatern52ardKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    return lengthScale;
    break;
  case 1:
    return variance;
    break;
  default:
    if(paramNo<nParams)
      return scales.getVal(paramNo-2);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
void CMatern52ardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2 = 5.0/(lengthScale*lengthScale);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double r2 = 0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      double dx = X.getVal(row, j)-X2.getVal(k, j);
      r2 += scales.getVal(j)*dx*dx;
    }
    double n2wi2 = r2*wi2;
    double t = sqrt(n2wi2);
    double expmt = exp(-t);
    double K = variance*(1+t+n2wi2/3.0)*expmt;
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double dx = X2.getVal(k, j)-X.getVal(row, j);
      double ratio = (t != 0.0) ? dx/t : 0.0;
      double val = scales.getVal(j)*(wi2*ratio*(K-variance*expmt) - variance*2.0*wi2/3.0*dx*expmt);
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  }
}
void CMatern52ardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CMatern52ardKern::getWhite() const
{
  return 0.0;
}

double CMatern52ardKern::computeElement(const CMatrix& X1, unsigned int index1, 
				     const CMatrix& X2, unsigned int index2) const
{
  double r2 = 0.0;
  for(unsigned int j=0; j<getInputDim(); j++)
  {
    double dx = X1.getVal(index1, j)-X2.getVal(index2, j);
    r2 += scales.getVal(j)*dx*dx;
  }
  double wi2 = 5.0/(lengthScale*lengthScale);
  double n2wi2 = r2*wi2;
  double t = sqrt(n2wi2);
  return variance*(1+t+n2wi2/3.0)*exp(-t);
}

void CMatern52ardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  gscales.zeros();
  double wi2=5.0/(lengthScale*lengthScale);
  unsigned int nrows = X.getRows();
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<X2.getRows(); i++)
    {
      double cgVal = covGrad.getVal(j,i);
      double r2 = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X.getVal(j, k)-X2.getVal(i, k);
	r2 += scales.getVal(k)*dx*dx;
      }
      double n2wi2 = r2*wi2;
      double t = sqrt(n2wi2);
      double expmt = exp(-t);
      double k = (1+t+n2wi2/3.0)*expmt;
      double ratio;
      if(t==0.0)
	ratio = 1.0;
      else
	ratio = r2/t;
      g1 += cgVal*(wi2/lengthScale*ratio*(k-expmt)-2.0*wi2/(3.0*lengthScale)*r2*expmt);
      g2 += cgVal*k;
      double factor = wi2*(1+t)*expmt/3.0;
      for(unsigned int kk=0; kk<getInputDim(); kk++)
      {
	double xi = X.getVal(j, kk);
	double xj = X2.getVal(i, kk);
	double g3 = gscales.getVal(kk);
	g3 += factor*cgVal*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
	gscales.setVal(g3, kk);
      }
    }
  }
  g.setVal(g1*variance, 0);
  g.setVal(g2, 1);
  for(unsigned int kk=0; kk<getInputDim(); kk++)
    g.setVal(gscales.getVal(kk), kk+2);
  if(regularise)
    addPriorGrad(g);
}

void CMatern52ardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  MATRIXPROPERTIES(covGrad.isSymmetric());
  double g1=0.0;
  double g2=0.0;
  gscales.zeros();
  double wi2=5.0/(lengthScale*lengthScale);
  unsigned int nrows = X.getRows();
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<j; i++)
    {
      double cgVal = covGrad.getVal(j,i);
      double r2 = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X.getVal(j, k)-X.getVal(i, k);
	r2 += scales.getVal(k)*dx*dx;
      }
      double n2wi2 = r2*wi2;
      double t = sqrt(n2wi2);
      double expmt = exp(-t);
      double k = (1+t+n2wi2/3.0)*expmt;
      double ratio;
      if(t==0.0)
	ratio = 1.0;
      else
	ratio = r2/t;
      g1 += cgVal*(wi2/lengthScale*ratio*(k-expmt)-2.0*wi2/(3.0*lengthScale)*r2*expmt);
      g2 += cgVal*k;
      double factor = wi2*(1+t)*expmt/3.0;
      for(unsigned int kk=0; kk<getInputDim(); kk++)
      {
	double xi = X.getVal(j, kk);
	double xj = X.getVal(i, kk);
	double g3 = gscales.getVal(kk);
	g3 += factor*cgVal*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
	gscales.setVal(g3, kk);
      }
    }
    g2 += 0.5*covGrad.getVal(j, j);
  }
  gscales.scale(2.0);
  g.setVal(2.0*g1*variance, 0);
  g.setVal(2.0*g2, 1);
  for(unsigned int kk=0; kk<getInputDim(); kk++)
    g.setVal(gscales.getVal(kk), kk+2);
  if(regularise)
    addPriorGrad(g);
}

double CMatern52ardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern52ardKern");
}
double CMatern52ardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CMatern52ardKern");
}

// the Rational Quadratic ARD kernel.
CRatQuadardKern::CRatQuadardKern() : CArdKern()
{
  _init();
}
CRatQuadardKern::CRatQuadardKern(unsigned int inDim) : CArdKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CRatQuadardKern::CRatQuadardKern(const CMatrix& X) : CArdKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CRatQuadardKern::CRatQuadardKern(const CRatQuadardKern& kern) : CArdKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  alpha = kern.alpha;
  lengthScale = kern.lengthScale;
  scales = kern.scales;
}
// Class destructor
CRatQuadardKern::~CRatQuadardKern()
{
}
double CRatQuadardKern::getVariance() const
{
  return variance;
}
void CRatQuadardKern::_init()
{
  nParams = 3;
  setType("ratquadard");
  setName("Rational Quadratic ARD");
  setParamName("alpha", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("lengthScale", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setParamName("variance", 2);
  addTransform(CTransform::defaultPositive(), 2);
  setStationary(true);
}
void CRatQuadardKern::setInitParam()
{
  nParams = 3+getInputDim();
  alpha = 1.0;
  lengthScale = 1.0;
  variance = 1.0;

  // input scales.
  scales.resize(1, getInputDim());
  gscales.resize(1, getInputDim());
  scales.setVals(0.5);
  for(unsigned int i=3; i<getInputDim()+3; i++)
  {
    string name = "inputScale";
    setParamName(name, i);
  }
  for(unsigned int i=3; i<getInputDim()+3; i++)
  {
    addTransform(CTransform::defaultZeroOne(), i);
  }
}

inline double CRatQuadardKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
// Parameters are kernel parameters
void CRatQuadardKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    alpha=val;
    break;
  case 1:
    lengthScale=val;
    break;
  case 2:
    variance=val;
    break;
  default:
    if(paramNo<nParams)
      scales.setVal(val, paramNo-3);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }    
  }
}
double CRatQuadardKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo<nParams);
  switch(paramNo)
  {
  case 0:
    return alpha;
    break;
  case 1:
    return lengthScale;
    break;
  case 2:
    return variance;
    break;
  default:
    if(paramNo<nParams)
      return scales.getVal(paramNo-3);
    else
    {
      throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }  
  }
}
void CRatQuadardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double wi2 = 0.5/(lengthScale*lengthScale*alpha);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double r2 = 0.0;
    for(unsigned int j=0; j<getInputDim(); j++)
    {
      double dx = X.getVal(row, j)-X2.getVal(k, j);
      r2 += scales.getVal(j)*dx*dx;
    }
    double ratquadPart = variance*pow((1+r2*wi2),-(alpha+1))/(lengthScale*lengthScale);
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = ratquadPart*scales.getVal(j)*(X2.getVal(k, j)-X.getVal(row, j));
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  }
}
void CRatQuadardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CRatQuadardKern::getWhite() const
{
  return 0.0;
}

double CRatQuadardKern::computeElement(const CMatrix& X1, unsigned int index1, 
				     const CMatrix& X2, unsigned int index2) const
{
  double r2 = 0.0;
  for(unsigned int j=0; j<getInputDim(); j++)
  {
    double dx = X1.getVal(index1, j)-X2.getVal(index2, j);
    r2 += scales.getVal(j)*dx*dx;
  }
  double wi2 = 0.5/(lengthScale*lengthScale*alpha);
  return variance*pow(1+r2*wi2, -alpha);
}

void CRatQuadardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  gscales.zeros();
  double wi2=0.5/(lengthScale*lengthScale*alpha);
  unsigned int nrows = X.getRows();
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<X2.getRows(); i++)
    {
      double r2 = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X2.getVal(i, k)-X.getVal(j, k);
	r2 += scales.getVal(k)*dx*dx;
      }
      double baseVal = (1+r2*wi2);
      double kbase = pow(baseVal, -alpha);
      double kbase2 = -alpha*kbase/baseVal;
      double cgVal = covGrad.getVal(j,i);
      g1 -= cgVal*(r2*wi2/alpha*kbase2 + log(baseVal)*kbase); 
      g2 -= cgVal*(r2*kbase2);
      g3 += kbase*cgVal;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X.getVal(j, k)-X2.getVal(i, k);
	double gs = gscales.getVal(k);
	gs += cgVal*kbase2*dx*dx;
	gscales.setVal(gs, k);
      }
    }
  }
  g.setVal(g1*variance, 0);
  g.setVal(g2*variance*2*wi2/lengthScale, 1);
  g.setVal(g3, 2);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k)*variance*wi2, k+3);
  if(regularise)
    addPriorGrad(g);
}

void CRatQuadardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  MATRIXPROPERTIES(covGrad.isSymmetric());
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  gscales.zeros();
  double wi2=0.5/(lengthScale*lengthScale*alpha);
  unsigned int nrows = X.getRows();
  for(unsigned int j=0; j<nrows; j++)
  {
    for(unsigned int i=0; i<j; i++)
    {
      double r2 = 0.0;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X.getVal(i, k)-X.getVal(j, k);
	r2 += scales.getVal(k)*dx*dx;
      }
      double baseVal = (1+r2*wi2);
      double kbase = pow(baseVal, -alpha);
      double kbase2 = -alpha*kbase/baseVal;
      double cgVal = covGrad.getVal(j,i);
      g1 -= cgVal*(r2*wi2/alpha*kbase2 + log(baseVal)*kbase); 
      g2 -= cgVal*(r2*kbase2);
      g3 += kbase*cgVal;
      for(unsigned int k=0; k<getInputDim(); k++)
      {
	double dx = X.getVal(j, k)-X.getVal(i, k);
	double gs = gscales.getVal(k);
	gs += cgVal*kbase2*dx*dx;
	gscales.setVal(gs, k);
      }
    }
    g3 += 0.5*covGrad.getVal(j, j);
  }
  gscales.scale(2.0*variance*wi2);
  g.setVal(2.0*g1*variance, 0);
  g.setVal(4.0*g2*variance*wi2/lengthScale, 1);
  g.setVal(2.0*g3, 2);
  for(unsigned int k=0; k<getInputDim(); k++)
    g.setVal(gscales.getVal(k), k+3);
  if(regularise)
    addPriorGrad(g);
}

double CRatQuadardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CRatQuadardKern");
}
double CRatQuadardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CRatQuadardKern");
}

// the Standard Periodic kernel (GPy-aligned, isotropic period/lengthScale: CIP-000C v1).
CStdPeriodicKern::CStdPeriodicKern() : CKern()
{
  _init();
}
CStdPeriodicKern::CStdPeriodicKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CStdPeriodicKern::CStdPeriodicKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CStdPeriodicKern::CStdPeriodicKern(const CStdPeriodicKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  lengthScale = kern.lengthScale;
  period = kern.period;
}
// Class destructor
CStdPeriodicKern::~CStdPeriodicKern()
{
}
double CStdPeriodicKern::getVariance() const
{
  return variance;
}
void CStdPeriodicKern::_init()
{
  nParams = 3;
  setType("stdperiodic");
  setName("Standard Periodic");
  setParamName("period", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("lengthScale", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setParamName("variance", 2);
  addTransform(CTransform::defaultPositive(), 2);
  setStationary(true);
}
void CStdPeriodicKern::setInitParam()
{
  period = 1.0;
  lengthScale = 1.0;
  variance = 1.0;
}

inline double CStdPeriodicKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CStdPeriodicKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CStdPeriodicKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    period = val;
    break;
  case 1:
    lengthScale = val;
    break;
  case 2:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CStdPeriodicKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return period;
    break;
  case 1:
    return lengthScale;
    break;
  case 2:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CStdPeriodicKern::computeElement(const CMatrix& X1, unsigned int index1, 
				      const CMatrix& X2, unsigned int index2) const
{
  double term = 0.0;
  for(unsigned int j=0; j<getInputDim(); j++)
  {
    double base = M_PI*(X1.getVal(index1, j)-X2.getVal(index2, j))/period;
    double s = sin(base)/lengthScale;
    term += s*s;
  }
  return variance*exp(-0.5*term);
}
// gradient of the kernel function w.r.t. the data (derivative with respect to X, the `row' point).
void CStdPeriodicKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double factor = M_PI/(period*lengthScale*lengthScale);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double kval = computeElement(X, row, X2, k);
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double base = M_PI*(X.getVal(row, j)-X2.getVal(k, j))/period;
      double val = -kval*factor*sin(base)*cos(base);
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  }
}
void CStdPeriodicKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CStdPeriodicKern::getWhite() const
{
  return 0.0;
}
// gradient of the kernel function w.r.t. the parameters (0:period, 1:lengthScale, 2:variance, two data matrices given)
void CStdPeriodicKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double gPeriod=0.0;
  double gLs=0.0;
  double gVar=0.0;
  unsigned int nrows = X.getRows();
  for(unsigned int i=0; i<nrows; i++)
  {
    for(unsigned int j=0; j<X2.getRows(); j++)
    {
      double term = 0.0;
      double sumBaseSin2Base = 0.0;
      for(unsigned int d=0; d<getInputDim(); d++)
      {
	double base = M_PI*(X.getVal(i, d)-X2.getVal(j, d))/period;
	double s = sin(base);
	term += (s*s)/(lengthScale*lengthScale);
	sumBaseSin2Base += base*sin(2.0*base);
      }
      double expTerm = exp(-0.5*term);
      double k = variance*expTerm;
      double cgVal = covGrad.getVal(i,j);
      gVar += cgVal*expTerm;
      gLs += cgVal*k*term/lengthScale;
      gPeriod += cgVal*k*0.5/(period*lengthScale*lengthScale)*sumBaseSin2Base;
    }
  }
  g.setVal(gPeriod, 0);
  g.setVal(gLs, 1);
  g.setVal(gVar, 2);
  if(regularise)
    addPriorGrad(g);
}
// gradient of the kernel function w.r.t. the parameters (one data matrix given)
void CStdPeriodicKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  getGradParams(g, X, X, covGrad, regularise); // just call it like the non-sym. case
  return; // it could be sped up by exploiting the fact it's symmetric
}

double CStdPeriodicKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CStdPeriodicKern");
}
double CStdPeriodicKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CStdPeriodicKern");
}

// the Cosine kernel (GPy-aligned, stationary: k(r)=variance*cos(r)).
CCosineKern::CCosineKern() : CStationaryKern()
{
  _init();
}
CCosineKern::CCosineKern(unsigned int inDim) : CStationaryKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CCosineKern::CCosineKern(const CMatrix& X) : CStationaryKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CCosineKern::CCosineKern(const CCosineKern& kern) : CStationaryKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
  lengthScale = kern.lengthScale;
}
// Class destructor
CCosineKern::~CCosineKern()
{
}
double CCosineKern::getVariance() const
{
  return variance;
}
void CCosineKern::_init()
{
  nParams = 2;
  setType("cosine");
  setName("Cosine");
  setParamName("lengthScale", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setParamName("variance", 1);
  addTransform(CTransform::defaultPositive(), 1);
  setStationary(true);
}
void CCosineKern::setInitParam()
{
  lengthScale = 1.0;
  variance = 1.0;
}

inline double CCosineKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance;
}
void CCosineKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
  DIMENSIONMATCH(d.getCols()==1);
  DIMENSIONMATCH(X.rowsMatch(d));
  d.setVals(variance);
}
// Parameters are kernel parameters
void CCosineKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    lengthScale = val;
    break;
  case 1:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
  invalidateWorkspace();
}
double CCosineKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo < nParams);
  switch(paramNo)
  {
  case 0:
    return lengthScale;
    break;
  case 1:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
// gradient of the kernel function w.r.t. the data (derivative with respect to X, the `row' point).
void CCosineKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double r = scaleFactor() * sqrt(dist2Active(X, row, X2, k));
    for(unsigned int j=0; j<X2.getCols(); j++)
    {
      double val = 0.0;
      if(r != 0.0)
      {
	double dx = X.getVal(row, j)-X2.getVal(k, j);
	val = -variance*sin(r)*dx/(lengthScale*lengthScale*r);
      }
      if(addG)
	gX.addVal(val, k, j);
      else 
	gX.setVal(val, k, j);
    }
  }
}
void CCosineKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CCosineKern::getWhite() const
{
  return 0.0;
}
// gradient of the kernel function w.r.t. the parameters (0:lengthScale, 1:variance, two data matrices given)
void CCosineKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1=0.0;
  double g2=0.0;
  const unsigned int nrows = X.getRows();
  const unsigned int ncols = X2.getRows();
  const bool useWork = hasRectWorkspace(X, X2);
  const double invVar = (variance == 0.0) ? 0.0 : 1.0 / variance;
  for(unsigned int i=0; i<nrows; i++)
  {
    for(unsigned int j=0; j<ncols; j++)
    {
      double d2 = 0.0;
      double cosr = 0.0;
      if(useWork)
      {
        d2 = rectR2(i, j);
        cosr = rectK(i, j) * invVar;
      }
      else
      {
        d2 = dist2Active(X, i, X2, j);
        cosr = cos(sqrt(d2)/lengthScale);
      }
      double r = sqrt(d2)/lengthScale;
      double cgVal = covGrad.getVal(i,j);
      g1 += cgVal*sin(r)*r;
      g2 += cgVal*cosr;
    }
  }
  g.setVal(g1*variance/lengthScale, 0);
  g.setVal(g2, 1);
  if(regularise)
    addPriorGrad(g);
}
// gradient of the kernel function w.r.t. the parameters (one data matrix given)
void CCosineKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  getGradParams(g, X, X, covGrad, regularise); // just call it like the non-sym. case
  return; // it could be sped up by exploiting the fact it's symmetric
}

double CCosineKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CCosineKern");
}
double CCosineKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CCosineKern");
}

// the Brownian motion kernel (GPy-aligned, non-stationary, 1-D index: k(x,x')=variance*min(x,x') for same-signed, non-zero x, x').
CBrownianKern::CBrownianKern() : CKern()
{
  _init();
}
CBrownianKern::CBrownianKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CBrownianKern::CBrownianKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}  
CBrownianKern::CBrownianKern(const CBrownianKern& kern) : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
}
// Class destructor
CBrownianKern::~CBrownianKern()
{
}
double CBrownianKern::getVariance() const
{
  return variance;
}
void CBrownianKern::_init()
{
  nParams = 1;
  setType("brownian");
  setName("Brownian motion");
  setParamName("variance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setStationary(false);
}
void CBrownianKern::setInitParam()
{
  if(getInputDim()!=1)
    throw ndlexceptions::Error("The Brownian kernel is only defined for a single (1-D) input dimension.");
  variance = 1.0;
}

double CBrownianKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variance*fabs(X.getVal(index, 0));
}
// Parameters are kernel parameters
void CBrownianKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo==0);
  switch(paramNo)
  {
  case 0:
    variance = val;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CBrownianKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo==0);
  switch(paramNo)
  {
  case 0:
    return variance;
    break;
  default:
    throw ndlexceptions::Error("Requested parameter doesn't exist.");
  }
}
double CBrownianKern::computeElement(const CMatrix& X1, unsigned int index1, 
				     const CMatrix& X2, unsigned int index2) const
{
  double x = X1.getVal(index1, 0);
  double y = X2.getVal(index2, 0);
  bool sameSign = (x>0.0 && y>0.0) || (x<0.0 && y<0.0);
  if(!sameSign)
    return 0.0;
  double ax = fabs(x);
  double ay = fabs(y);
  return variance*(ax<ay?ax:ay);
}
// gradient of the kernel function w.r.t. the data (derivative with respect to X, the `row' point).
void CBrownianKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  double x = X.getVal(row, 0);
  double sgnX = (x>0.0) - (x<0.0);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    double y = X2.getVal(k, 0);
    bool sameSign = (x>0.0 && y>0.0) || (x<0.0 && y<0.0);
    double val = 0.0;
    if(sameSign && fabs(x)<fabs(y))
      val = variance*sgnX;
    if(addG)
      gX.addVal(val, k, 0);
    else 
      gX.setVal(val, k, 0);
  }
}
// gradient of the diagonal of the kernel matrix w.r.t. the data: d(variance*|x|)/dx = variance*sign(x).
void CBrownianKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    double x = X.getVal(i, 0);
    double sgnX = (x>0.0) - (x<0.0);
    double val = variance*sgnX;
    if(addG)
      gX.addVal(val, i, 0);
    else
      gX.setVal(val, i, 0);
  }
}
double CBrownianKern::getWhite() const
{
  return 0.0;
}
// gradient of the kernel function w.r.t. the parameters (0:variance, two data matrices given)
void CBrownianKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1 = 0.0;
  unsigned int nrows = X.getRows();
  for(unsigned int i=0; i<nrows; i++)
  {
    double x = X.getVal(i, 0);
    for(unsigned int j=0; j<X2.getRows(); j++)
    {
      double y = X2.getVal(j, 0);
      bool sameSign = (x>0.0 && y>0.0) || (x<0.0 && y<0.0);
      double base = 0.0;
      if(sameSign)
      {
	double ax = fabs(x);
	double ay = fabs(y);
	base = (ax<ay?ax:ay);
      }
      g1 += covGrad.getVal(i,j)*base;
    }
  }
  g.setVal(g1, 0);
  if(regularise)
    addPriorGrad(g);
}
// gradient of the kernel function w.r.t. the parameters (one data matrix given)
void CBrownianKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  getGradParams(g, X, X, covGrad, regularise); // just call it like the non-sym. case
  return; // it could be sped up by exploiting the fact it's symmetric
}

double CBrownianKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CBrownianKern");
}
double CBrownianKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("Error getGradParam is not currently implemented for CBrownianKern");
}

// CIP-0029 Phase 1: Wiener causal root × Gaussian windows.
// f(t)=σ∫_0^t ξ, u_m=∫ψ_m ξ, ψ=(4πs²)^{1/4} N(·;z,s²)
// ⇒ K_fu = σ (4πs²)^{1/4} [Φ((t-z)/s) − Φ(−z/s)].
namespace {
inline double greensPsiNorm(double s)
{
  return pow(4.0 * M_PI * s * s, 0.25);
}
} // namespace

void CBrownianKern::greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
                                       double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  Kfu.resize(N, M);
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  const double amp = sigma * c;
  for(unsigned int n = 0; n < N; n++)
  {
    const double t = X.getVal(n, 0);
    for(unsigned int m = 0; m < M; m++)
    {
      const double z = Z.getVal(m, 0);
      const double dPhi = ndlutil::cumGaussian((t - z) / s) - ndlutil::cumGaussian(-z / s);
      Kfu.setVal(amp * dPhi, n, m);
    }
  }
}

void CBrownianKern::greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                                     const CMatrix& gKfu) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  // ∂K/∂variance = K/(2 variance)
  double gVar = 0.0;
  const double inv2var = (variance == 0.0) ? 0.0 : 0.5 / variance;
  CMatrix Kfu;
  greensCrossCompute(Kfu, X, Z, s);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      gVar += gKfu.getVal(n, m) * Kfu.getVal(n, m) * inv2var;
  g.setVal(gVar, 0);
}

void CBrownianKern::greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gZ.resize(M, 1);
  gZ.zeros();
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  const double amp_s = sigma * c / s;
  for(unsigned int m = 0; m < M; m++)
  {
    const double z = Z.getVal(m, 0);
    const double phiBeta = ndlutil::ngaussian(-z / s);
    double acc = 0.0;
    for(unsigned int n = 0; n < N; n++)
    {
      const double t = X.getVal(n, 0);
      const double phiAlpha = ndlutil::ngaussian((t - z) / s);
      // ∂K/∂z = σ c / s (−φ(α) + φ(β))
      acc += gKfu.getVal(n, m) * amp_s * (-phiAlpha + phiBeta);
    }
    gZ.setVal(acc, m, 0);
  }
}

void CBrownianKern::greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gX.resize(N, 1);
  gX.zeros();
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  const double amp_s = sigma * c / s;
  for(unsigned int n = 0; n < N; n++)
  {
    const double t = X.getVal(n, 0);
    double acc = 0.0;
    for(unsigned int m = 0; m < M; m++)
    {
      const double z = Z.getVal(m, 0);
      // ∂K/∂t = σ c / s φ((t-z)/s)
      acc += gKfu.getVal(n, m) * amp_s * ndlutil::ngaussian((t - z) / s);
    }
    gX.setVal(acc, n, 0);
  }
}

void CBrownianKern::greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gS.resize(1, 1);
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  double acc = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    const double t = X.getVal(n, 0);
    for(unsigned int m = 0; m < M; m++)
    {
      const double z = Z.getVal(m, 0);
      const double alpha = (t - z) / s;
      const double beta = -z / s;
      const double dPhi = ndlutil::cumGaussian(alpha) - ndlutil::cumGaussian(beta);
      const double dKdS = sigma * (
          (c / (2.0 * s)) * dPhi
          + c * (-ndlutil::ngaussian(alpha) * alpha / s
                 + ndlutil::ngaussian(beta) * beta / s));
      acc += gKfu.getVal(n, m) * dKdS;
    }
  }
  gS.setVal(acc, 0, 0);
}

// CIP-0029 Phase 3: integrated Brownian I(t)=∫_0^t W.
CIntegratedBrownianKern::CIntegratedBrownianKern() : CKern()
{
  _init();
}
CIntegratedBrownianKern::CIntegratedBrownianKern(unsigned int inDim) : CKern(inDim)
{
  _init();
  setInputDim(inDim);
}
CIntegratedBrownianKern::CIntegratedBrownianKern(const CMatrix& X) : CKern(X)
{
  _init();
  setInputDim(X.getCols());
}
CIntegratedBrownianKern::CIntegratedBrownianKern(const CIntegratedBrownianKern& kern)
  : CKern(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  variance = kern.variance;
}
CIntegratedBrownianKern::~CIntegratedBrownianKern()
{
}
double CIntegratedBrownianKern::getVariance() const
{
  return variance;
}
void CIntegratedBrownianKern::_init()
{
  nParams = 1;
  setType("ibrownian");
  setName("Integrated Brownian");
  setParamName("variance", 0);
  addTransform(CTransform::defaultPositive(), 0);
  setStationary(false);
}
void CIntegratedBrownianKern::setInitParam()
{
  if(getInputDim()!=1)
    throw ndlexceptions::Error("The integrated Brownian kernel is only defined for a single (1-D) input dimension.");
  variance = 1.0;
}
double CIntegratedBrownianKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  const double ax = fabs(X.getVal(index, 0));
  return variance * ax * ax * ax / 3.0;
}
void CIntegratedBrownianKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo==0);
  variance = val;
}
double CIntegratedBrownianKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo==0);
  return variance;
}
double CIntegratedBrownianKern::computeElement(const CMatrix& X1, unsigned int index1,
                                              const CMatrix& X2, unsigned int index2) const
{
  const double x = X1.getVal(index1, 0);
  const double y = X2.getVal(index2, 0);
  const bool sameSign = (x>0.0 && y>0.0) || (x<0.0 && y<0.0);
  if(!sameSign)
    return 0.0;
  const double a = fabs(x) < fabs(y) ? fabs(x) : fabs(y);
  const double b = fabs(x) < fabs(y) ? fabs(y) : fabs(x);
  // 0≤a≤b: k = variance (a² b / 2 − a³ / 6)
  return variance * (0.5 * a * a * b - a * a * a / 6.0);
}
void CIntegratedBrownianKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row,
                                      const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows() == X2.getRows());
  BOUNDCHECK(row < X.getRows());
  DIMENSIONMATCH(X.getCols()==X2.getCols());
  const double x = X.getVal(row, 0);
  const double sgnX = (x>0.0) - (x<0.0);
  const double ax = fabs(x);
  for(unsigned int k=0; k<X2.getRows(); k++)
  {
    const double y = X2.getVal(k, 0);
    const bool sameSign = (x>0.0 && y>0.0) || (x<0.0 && y<0.0);
    double val = 0.0;
    if(sameSign)
    {
      const double ay = fabs(y);
      // ∂/∂ax of k(ax,ay); chain with sgnX.
      double dAbs = 0.0;
      if(ax <= ay)
        dAbs = variance * (ax * ay - 0.5 * ax * ax);
      else
        dAbs = variance * (0.5 * ay * ay);
      val = dAbs * sgnX;
    }
    if(addG)
      gX.addVal(val, k, 0);
    else
      gX.setVal(val, k, 0);
  }
}
void CIntegratedBrownianKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    const double x = X.getVal(i, 0);
    const double sgnX = (x>0.0) - (x<0.0);
    const double ax = fabs(x);
    // d(v |x|³/3)/dx = v x² sgn(x)
    const double val = variance * ax * ax * sgnX;
    if(addG)
      gX.addVal(val, i, 0);
    else
      gX.setVal(val, i, 0);
  }
}
double CIntegratedBrownianKern::getWhite() const
{
  return 0.0;
}
void CIntegratedBrownianKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2,
                                           const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  DIMENSIONMATCH(X.getRows()==covGrad.getRows());
  DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
  double g1 = 0.0;
  for(unsigned int i=0; i<X.getRows(); i++)
  {
    const double x = X.getVal(i, 0);
    for(unsigned int j=0; j<X2.getRows(); j++)
    {
      const double y = X2.getVal(j, 0);
      const bool sameSign = (x>0.0 && y>0.0) || (x<0.0 && y<0.0);
      double base = 0.0;
      if(sameSign)
      {
        const double a = fabs(x) < fabs(y) ? fabs(x) : fabs(y);
        const double b = fabs(x) < fabs(y) ? fabs(y) : fabs(x);
        base = 0.5 * a * a * b - a * a * a / 6.0;
      }
      g1 += covGrad.getVal(i, j) * base;
    }
  }
  g.setVal(g1, 0);
  if(regularise)
    addPriorGrad(g);
}
void CIntegratedBrownianKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad,
                                           bool regularise) const
{
  getGradParams(g, X, X, covGrad, regularise);
}
double CIntegratedBrownianKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2,
                                             const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("getGradParam is not implemented for CIntegratedBrownianKern");
}
double CIntegratedBrownianKern::getGradParam(unsigned int index, const CMatrix& X,
                                             const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  throw ndlexceptions::NotImplementedError("getGradParam is not implemented for CIntegratedBrownianKern");
}

// Causal root g(u)=σ u 1_{u≥0} on [0,t]:
// K_fu = σ c [(t−z)(Φ(α)−Φ(β)) + s(φ(α)−φ(β))], α=(t−z)/s, β=−z/s.
void CIntegratedBrownianKern::greensCrossCompute(CMatrix& Kfu, const CMatrix& X,
                                                 const CMatrix& Z, double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  Kfu.resize(N, M);
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  const double amp = sigma * c;
  for(unsigned int n = 0; n < N; n++)
  {
    const double t = X.getVal(n, 0);
    for(unsigned int m = 0; m < M; m++)
    {
      const double z = Z.getVal(m, 0);
      const double alpha = (t - z) / s;
      const double beta = -z / s;
      const double dPhi = ndlutil::cumGaussian(alpha) - ndlutil::cumGaussian(beta);
      const double dPhiDens = ndlutil::ngaussian(alpha) - ndlutil::ngaussian(beta);
      Kfu.setVal(amp * ((t - z) * dPhi + s * dPhiDens), n, m);
    }
  }
}
void CIntegratedBrownianKern::greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z,
                                               double s, const CMatrix& gKfu) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  double gVar = 0.0;
  const double inv2var = (variance == 0.0) ? 0.0 : 0.5 / variance;
  CMatrix Kfu;
  greensCrossCompute(Kfu, X, Z, s);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      gVar += gKfu.getVal(n, m) * Kfu.getVal(n, m) * inv2var;
  g.setVal(gVar, 0);
}
void CIntegratedBrownianKern::greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z,
                                          double s, const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gZ.resize(M, 1);
  gZ.zeros();
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  const double amp = sigma * c;
  for(unsigned int m = 0; m < M; m++)
  {
    const double z = Z.getVal(m, 0);
    const double beta = -z / s;
    const double phiBeta = ndlutil::ngaussian(beta);
    double acc = 0.0;
    for(unsigned int n = 0; n < N; n++)
    {
      const double t = X.getVal(n, 0);
      const double alpha = (t - z) / s;
      const double phiAlpha = ndlutil::ngaussian(alpha);
      const double dPhi = ndlutil::cumGaussian(alpha) - ndlutil::cumGaussian(beta);
      // ∂B/∂z, B=(t−z)ΔΦ + s Δφ
      const double dBdz = -dPhi
                          + (t - z) / s * (-phiAlpha + phiBeta)
                          + alpha * phiAlpha - beta * phiBeta;
      acc += gKfu.getVal(n, m) * amp * dBdz;
    }
    gZ.setVal(acc, m, 0);
  }
}
void CIntegratedBrownianKern::greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z,
                                          double s, const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gX.resize(N, 1);
  gX.zeros();
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  const double amp = sigma * c;
  for(unsigned int n = 0; n < N; n++)
  {
    const double t = X.getVal(n, 0);
    double acc = 0.0;
    for(unsigned int m = 0; m < M; m++)
    {
      const double z = Z.getVal(m, 0);
      const double alpha = (t - z) / s;
      const double beta = -z / s;
      const double dPhi = ndlutil::cumGaussian(alpha) - ndlutil::cumGaussian(beta);
      // ∂B/∂t = ΔΦ
      acc += gKfu.getVal(n, m) * amp * dPhi;
    }
    gX.setVal(acc, n, 0);
  }
}
void CIntegratedBrownianKern::greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z,
                                          double s, const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gS.resize(1, 1);
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm(s);
  double acc = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    const double t = X.getVal(n, 0);
    for(unsigned int m = 0; m < M; m++)
    {
      const double z = Z.getVal(m, 0);
      const double alpha = (t - z) / s;
      const double beta = -z / s;
      const double phiA = ndlutil::ngaussian(alpha);
      const double phiB = ndlutil::ngaussian(beta);
      const double dPhi = ndlutil::cumGaussian(alpha) - ndlutil::cumGaussian(beta);
      const double dPhiDens = phiA - phiB;
      const double B = (t - z) * dPhi + s * dPhiDens;
      const double dBds = -(t - z) / s * (alpha * phiA - beta * phiB)
                          + dPhiDens
                          + alpha * alpha * phiA - beta * beta * phiB;
      // K = σ c B; dc/ds = c/(2s)
      const double dKdS = sigma * ((c / (2.0 * s)) * B + c * dBds);
      acc += gKfu.getVal(n, m) * dKdS;
    }
  }
  gS.setVal(acc, 0, 0);
}

// Coregionalize kernel (CIP-000D / GPy Coregionalize).
CCoregionalizeKern::CCoregionalizeKern() : CKern(), outputDim(1), rank(1)
{
  _init();
  setInputDim(1);
}
CCoregionalizeKern::CCoregionalizeKern(unsigned int outDim, unsigned int r)
  : CKern(), outputDim(outDim), rank(r)
{
  _init();
  setInputDim(1);
}
CCoregionalizeKern::CCoregionalizeKern(unsigned int inDim, unsigned int outDim, unsigned int r)
  : CKern(), outputDim(outDim), rank(r)
{
  _init();
  setInputDim(inDim);
}
CCoregionalizeKern::CCoregionalizeKern(const CMatrix& X, unsigned int outDim, unsigned int r)
  : CKern(X), outputDim(outDim), rank(r)
{
  _init();
  setInputDim(1);
}
CCoregionalizeKern::CCoregionalizeKern(const CCoregionalizeKern& kern)
  : CKern(kern), outputDim(kern.outputDim), rank(kern.rank),
    W(kern.W), kappa(kern.kappa), B(kern.B)
{
  _init();
  setInputDim(kern.getInputDim());
  activeDims = kern.activeDims;
  W = kern.W;
  kappa = kern.kappa;
  _rebuildB();
}
CCoregionalizeKern::~CCoregionalizeKern()
{
}
void CCoregionalizeKern::_init()
{
  BOUNDCHECK(outputDim>=1);
  BOUNDCHECK(rank>=1);
  nParams = outputDim*rank + outputDim;
  setType("coregionalize");
  setName("coregionalize");
  setStationary(false);
  W.resize(outputDim, rank);
  kappa.resize(1, outputDim);
  B.resize(outputDim, outputDim);
  paramNames.clear();
  clearTransforms();
  for(unsigned int d=0; d<outputDim; d++)
  {
    for(unsigned int r=0; r<rank; r++)
      setParamName("W_" + ndlstrutil::itoa(d) + "_" + ndlstrutil::itoa(r), d*rank + r);
  }
  for(unsigned int d=0; d<outputDim; d++)
  {
    unsigned int idx = outputDim*rank + d;
    setParamName("kappa_" + ndlstrutil::itoa(d), idx);
    addTransform(CTransform::defaultPositive(), idx);
  }
}
void CCoregionalizeKern::setInitParam()
{
  // GPy defaults: W ~ 0.5 N(0,1)/sqrt(rank), kappa = 0.5. Deterministic fill for tests.
  for(unsigned int d=0; d<outputDim; d++)
  {
    for(unsigned int r=0; r<rank; r++)
      W.setVal(0.5/sqrt(static_cast<double>(rank)), d, r);
    kappa.setVal(0.5, d);
  }
  _rebuildB();
}
void CCoregionalizeKern::_rebuildB()
{
  B.resize(outputDim, outputDim);
  B.gemm(W, W, 1.0, 0.0, "n", "t");
  for(unsigned int d=0; d<outputDim; d++)
    B.setVal(B.getVal(d, d) + kappa.getVal(d), d, d);
  B.setSymmetric(true);
}
void CCoregionalizeKern::setW(const CMatrix& Win)
{
  DIMENSIONMATCH(Win.getRows()==outputDim);
  DIMENSIONMATCH(Win.getCols()==rank);
  W.deepCopy(Win);
  _rebuildB();
}
void CCoregionalizeKern::setKappa(const CMatrix& kappaIn)
{
  DIMENSIONMATCH(kappaIn.getRows()==1 || kappaIn.getCols()==outputDim || kappaIn.getRows()==outputDim);
  if(kappaIn.getRows()==1 && kappaIn.getCols()==outputDim)
    kappa.deepCopy(kappaIn);
  else if(kappaIn.getCols()==1 && kappaIn.getRows()==outputDim)
  {
    for(unsigned int d=0; d<outputDim; d++)
      kappa.setVal(kappaIn.getVal(d, 0), d);
  }
  else
    throw ndlexceptions::Error("kappa must be length outputDim");
  _rebuildB();
}
double CCoregionalizeKern::getVariance() const
{
  double s = 0.0;
  for(unsigned int d=0; d<outputDim; d++)
    s += B.getVal(d, d);
  return s/static_cast<double>(outputDim);
}
void CCoregionalizeKern::setVariance(double val)
{
  double cur = getVariance();
  if(cur<=0.0)
    return;
  double scale = val/cur;
  for(unsigned int d=0; d<outputDim; d++)
  {
    for(unsigned int r=0; r<rank; r++)
      W.setVal(W.getVal(d, r)*sqrt(scale), d, r);
    kappa.setVal(kappa.getVal(d)*scale, d);
  }
  _rebuildB();
}
unsigned int CCoregionalizeKern::_indexOf(const CMatrix& X, unsigned int row) const
{
  BOUNDCHECK(activeDims.size()==1);
  unsigned int col = activeDims[0];
  BOUNDCHECK(col<X.getCols());
  return indexFromDouble(X.getVal(row, col), outputDim);
}
double CCoregionalizeKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  unsigned int i = _indexOf(X, index);
  return B.getVal(i, i);
}
double CCoregionalizeKern::computeElement(const CMatrix& X1, unsigned int index1,
                                         const CMatrix& X2, unsigned int index2) const
{
  unsigned int i = _indexOf(X1, index1);
  unsigned int j = _indexOf(X2, index2);
  return B.getVal(i, j);
}
void CCoregionalizeKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo<nParams);
  if(paramNo < outputDim*rank)
  {
    unsigned int d = paramNo / rank;
    unsigned int r = paramNo % rank;
    W.setVal(val, d, r);
  }
  else
    kappa.setVal(val, paramNo - outputDim*rank);
  _rebuildB();
}
double CCoregionalizeKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo<nParams);
  if(paramNo < outputDim*rank)
  {
    unsigned int d = paramNo / rank;
    unsigned int r = paramNo % rank;
    return W.getVal(d, r);
  }
  return kappa.getVal(paramNo - outputDim*rank);
}
void CCoregionalizeKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows()==X2.getRows());
  DIMENSIONMATCH(gX.getCols()==X2.getCols());
  if(!addG)
    gX.zeros();
}
void CCoregionalizeKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
void CCoregionalizeKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==nParams);
  // Reduce covGrad onto B indices (GPy _gradient_reduce), then
  // dkappa = diag(dL); dL = dL + dL'; dW[j,r] = sum_i W[i,r]*dL[i,j].
  CMatrix dL(outputDim, outputDim);
  dL.zeros();
  for(unsigned int n=0; n<X.getRows(); n++)
  {
    unsigned int i = _indexOf(X, n);
    for(unsigned int m=0; m<X2.getRows(); m++)
    {
      unsigned int j = _indexOf(X2, m);
      dL.addVal(covGrad.getVal(n, m), j, i);
    }
  }
  for(unsigned int d=0; d<outputDim; d++)
    g.setVal(dL.getVal(d, d), outputDim*rank + d);
  CMatrix dLsym(outputDim, outputDim);
  for(unsigned int i=0; i<outputDim; i++)
    for(unsigned int j=0; j<outputDim; j++)
      dLsym.setVal(dL.getVal(i, j) + dL.getVal(j, i), i, j);
  for(unsigned int j=0; j<outputDim; j++)
  {
    for(unsigned int r=0; r<rank; r++)
    {
      double acc = 0.0;
      for(unsigned int i=0; i<outputDim; i++)
        acc += W.getVal(i, r) * dLsym.getVal(i, j);
      g.setVal(acc, j*rank + r);
    }
  }
  if(regularise)
    addPriorGrad(g);
}
void CCoregionalizeKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
  getGradParams(g, X, X, covGrad, regularise);
}
double CCoregionalizeKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const
{
  CMatrix g(1, nParams);
  getGradParams(g, X, X2, cvGrd, false);
  return g.getVal(index);
}
double CCoregionalizeKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const
{
  return getGradParam(index, X, X, cvGrd);
}
void CCoregionalizeKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "outputDim", outputDim);
  writeToStream(out, "rank", rank);
  writeToStream(out, "numParams", getNumParams());
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
  writeToStream(out, "numPriors", getNumPriors());
  writePriorsToStream(out);
}
void CCoregionalizeKern::readParamsFromStream(istream& in)
{
  setInputDim(readIntFromStream(in, "inputDim"));
  outputDim = readIntFromStream(in, "outputDim");
  rank = readIntFromStream(in, "rank");
  _init();
  unsigned int nPars = readIntFromStream(in, "numParams");
  CMatrix par(1, nPars);
  par.fromStream(in);
  if(nPars!=getNumParams())
    throw ndlexceptions::StreamFormatError("numParams", "Listed number of parameters does not match computed number of parameters.");
  setParams(par);
  unsigned int numPriors = readIntFromStream(in, "numPriors");
  readPriorsFromStream(in, numPriors);
}

// Index-white kernel (CIP-000D).
CIndexWhiteKern::CIndexWhiteKern() : CKern(), outputDim(1)
{
  _init();
  setInputDim(1);
}
CIndexWhiteKern::CIndexWhiteKern(unsigned int outDim) : CKern(), outputDim(outDim)
{
  _init();
  setInputDim(1);
}
CIndexWhiteKern::CIndexWhiteKern(unsigned int inDim, unsigned int outDim)
  : CKern(), outputDim(outDim)
{
  _init();
  setInputDim(inDim);
}
CIndexWhiteKern::CIndexWhiteKern(const CMatrix& X, unsigned int outDim)
  : CKern(X), outputDim(outDim)
{
  _init();
  setInputDim(1);
}
CIndexWhiteKern::CIndexWhiteKern(const CIndexWhiteKern& kern)
  : CKern(kern), outputDim(kern.outputDim), variances(kern.variances)
{
  _init();
  setInputDim(kern.getInputDim());
  activeDims = kern.activeDims;
  variances = kern.variances;
}
CIndexWhiteKern::~CIndexWhiteKern()
{
}
void CIndexWhiteKern::_init()
{
  BOUNDCHECK(outputDim>=1);
  nParams = outputDim;
  setType("indexwhite");
  setName("index white");
  setStationary(false);
  variances.resize(1, outputDim);
  paramNames.clear();
  clearTransforms();
  for(unsigned int i=0; i<nParams; i++)
  {
    setParamName("variance_" + ndlstrutil::itoa(i), i);
    addTransform(CTransform::defaultPositive(), i);
  }
}
void CIndexWhiteKern::setInitParam()
{
  for(unsigned int d=0; d<outputDim; d++)
    variances.setVal(exp(-2.0), d);
}
unsigned int CIndexWhiteKern::_indexOf(const CMatrix& X, unsigned int row) const
{
  BOUNDCHECK(activeDims.size()==1);
  unsigned int col = activeDims[0];
  BOUNDCHECK(col<X.getCols());
  return indexFromDouble(X.getVal(row, col), outputDim);
}
double CIndexWhiteKern::getVariance() const
{
  double s = 0.0;
  for(unsigned int d=0; d<outputDim; d++)
    s += variances.getVal(d);
  return s/static_cast<double>(outputDim);
}
void CIndexWhiteKern::setVariance(double val)
{
  for(unsigned int d=0; d<outputDim; d++)
    variances.setVal(val, d);
}
double CIndexWhiteKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
  return variances.getVal(_indexOf(X, index));
}
double CIndexWhiteKern::computeElement(const CMatrix& X1, unsigned int index1,
                                       const CMatrix& X2, unsigned int index2) const
{
  return 0.0;
}
void CIndexWhiteKern::setParam(double val, unsigned int paramNo)
{
  BOUNDCHECK(paramNo<nParams);
  variances.setVal(val, paramNo);
}
double CIndexWhiteKern::getParam(unsigned int paramNo) const
{
  BOUNDCHECK(paramNo<nParams);
  return variances.getVal(paramNo);
}
void CIndexWhiteKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
  DIMENSIONMATCH(gX.getRows()==X2.getRows());
  DIMENSIONMATCH(gX.getCols()==X2.getCols());
  if(!addG)
    gX.zeros();
}
void CIndexWhiteKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
  DIMENSIONMATCH(gX.dimensionsMatch(X));
  if(!addG)
    gX.zeros();
}
double CIndexWhiteKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  return 0.0;
}
double CIndexWhiteKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{
  BOUNDCHECK(index<nParams);
  double g = 0.0;
  for(unsigned int n=0; n<X.getRows(); n++)
  {
    if(_indexOf(X, n)==index)
      g += covGrad.getVal(n, n);
  }
  return g;
}
void CIndexWhiteKern::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "outputDim", outputDim);
  writeToStream(out, "numParams", getNumParams());
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
  writeToStream(out, "numPriors", getNumPriors());
  writePriorsToStream(out);
}
void CIndexWhiteKern::readParamsFromStream(istream& in)
{
  setInputDim(readIntFromStream(in, "inputDim"));
  outputDim = readIntFromStream(in, "outputDim");
  _init();
  unsigned int nPars = readIntFromStream(in, "numParams");
  CMatrix par(1, nPars);
  par.fromStream(in);
  if(nPars!=getNumParams())
    throw ndlexceptions::StreamFormatError("numParams", "Listed number of parameters does not match computed number of parameters.");
  setParams(par);
  unsigned int numPriors = readIntFromStream(in, "numPriors");
  readPriorsFromStream(in, numPriors);
}

CTensorKern* createIcmKern(const CKern& spatial, unsigned int numOutputs, unsigned int rank)
{
  unsigned int q = spatial.getInputDim();
  if(!spatial.isDefaultActiveDims())
  {
    // Still OK if caller already set active dims on spatial; we clone and keep them.
  }
  CKern* space = spatial.clone();
  if(space->isDefaultActiveDims())
  {
    vector<unsigned int> spaceDims(q);
    for(unsigned int i=0; i<q; i++)
      spaceDims[i] = i;
    space->setActiveDims(spaceDims);
  }
  CCoregionalizeKern* coreg = new CCoregionalizeKern(numOutputs, rank);
  vector<unsigned int> idx(1, q);
  coreg->setActiveDims(idx);
  CTensorKern* icm = new CTensorKern(q+1);
  icm->addKern(space);
  icm->addKern(coreg);
  delete space;
  delete coreg;
  return icm;
}
CCmpndKern* createLcmKern(const vector<const CKern*>& spatials, unsigned int numOutputs, unsigned int rank)
{
  if(spatials.empty())
    throw ndlexceptions::Error("LCM requires at least one spatial kernel");
  unsigned int q = spatials[0]->getInputDim();
  CCmpndKern* lcm = new CCmpndKern(q+1);
  for(size_t i=0; i<spatials.size(); i++)
  {
    DIMENSIONMATCH(spatials[i]->getInputDim()==q);
    CTensorKern* icm = createIcmKern(*spatials[i], numOutputs, rank);
    lcm->addKern(icm);
    delete icm;
  }
  return lcm;
}

// Functions that operate on CKern.
ostream& operator<<(ostream& out, const CKern& kern)
{
  // display() already writes to the stream and returns it; do not stream the ostream itself.
  kern.display(out);
  return out;
}
void writeKernToStream(const CKern& kern, ostream& out)
{
  kern.toStream(out);
}
CKern* readKernFromStream(istream& in)
{
  double ver = CStreamInterface::readVersionFromStream(in); 
  string tbaseType = CStreamInterface::getBaseTypeStream(in);
  if(tbaseType != "kern")
    throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, kern.");
  CKern* pkern;

  string type = CStreamInterface::getTypeStream(in);
  if(type=="white")
    pkern = new CWhiteKern();
  else if(type=="whitefixed")
    pkern = new CWhitefixedKern();
  else if(type=="bias")
    pkern = new CBiasKern();

  else if(type=="rbf")
    pkern = new CRbfKern();
  else if(type=="eq" || type=="ExpQuad" || type=="expquad")
    pkern = new CEqKern();
  else if(type=="exp")
    pkern = new CExpKern();
  else if(type=="ratquad")
    pkern = new CRatQuadKern();
  else if(type=="matern32")
    pkern = new CMatern32Kern();
  else if(type=="matern52")
    pkern = new CMatern52Kern();
  else if(type=="lin")
    pkern = new CLinKern();
  else if(type=="poly")
    pkern = new CPolyKern();
  else if(type=="mlp")
    pkern = new CMlpKern();
  else if(type=="arccos" || type=="arccosine")
    pkern = new CArcCosineKern();
  else if(type=="rbfard")
    pkern = new CRbfardKern();
  else if(type=="linard")
    pkern = new CLinardKern();
  else if(type=="polyard")
    pkern = new CPolyardKern();
  else if(type=="mlpard")
    pkern = new CMlpardKern();
  else if(type=="matern32ard")
    pkern = new CMatern32ardKern();
  else if(type=="matern52ard")
    pkern = new CMatern52ardKern();
  else if(type=="ratquadard")
    pkern = new CRatQuadardKern();
  else if(type=="stdperiodic")
    pkern = new CStdPeriodicKern();
  else if(type=="cosine")
    pkern = new CCosineKern();
  else if(type=="brownian")
    pkern = new CBrownianKern();
  else if(type=="ibrownian" || type=="integratedbrownian")
    pkern = new CIntegratedBrownianKern();
  else if(type=="coregionalize")
    pkern = new CCoregionalizeKern();
  else if(type=="indexwhite")
    pkern = new CIndexWhiteKern();
  else if(type=="cmpnd")
    pkern = new CCmpndKern();
  else if(type=="tensor")
    pkern = new CTensorKern();
  else
    throw ndlexceptions::StreamFormatError("type", "Unknown kernel type " + type);
  pkern->readParamsFromStream(in);  
  return pkern;
}
void CKern::readParamsFromStream(istream& in)
{
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setInputDim(readIntFromStream(in, "inputDim"));
  unsigned int nPars = readIntFromStream(in, "numParams");
  CMatrix par(1, nPars);
  par.fromStream(in);
  if(nPars==getNumParams())
    setParams(par);
  else
    throw ndlexceptions::StreamFormatError("numParams", "Listed number of parameters does not match computed number of parameters.");
  unsigned int numPriors = readIntFromStream(in, "numPriors");
  readPriorsFromStream(in, numPriors);
}
#ifdef _NDLMATLAB
mxArray* CKern::toMxArray() const
{
  size_t dims[1];
  dims[0] = 1;
  const char *fieldNames[] = {"type", "transforms", "inputDimension"};
  mxArray* matlabArray = mxCreateStructArray(1, dims, 3, fieldNames);
    
  // type field.
  const char *typeName[1];
  string ty=getType();
  typeName[0] = ty.c_str();
  mxSetField(matlabArray, 0, "type", 
	     mxCreateCharMatrixFromStrings(1, typeName));
  
  // transforms field.
  mxSetField(matlabArray, 0, "transforms", transformsToMxArray());

  // inputDimension field.
  mxSetField(matlabArray, 0, "inputDimension", convertMxArray((double)inputDim));

  
  // whiteVariance field.
  double whiteVar = getWhite();
  if(whiteVar !=0.0)
  {
    mxAddField(matlabArray, "whiteVariance");      
    mxSetField(matlabArray, 0, "whiteVariance", convertMxArray(whiteVar)); 
  }
  
  // priors field.
  /// if priros exist need to add the field.

  // Kernel specific code.
  addParamToMxArray(matlabArray);
  return matlabArray;

}
void CKern::fromMxArray(const mxArray* matlabArray) 
{
  string mxType = mxArrayExtractStringField(matlabArray, "type");
  if(mxType!=getType())
  {
    throw ndlexceptions::FileReadError("Error mismatch between saved type, " + mxType + ", and Class type, " + type + ".");
  }
  mxArray* transformArray = mxArrayExtractMxArrayField(matlabArray, "transforms");
  // transforms field.
  transformsFromMxArray(transformArray);
  extractParamFromMxArray(matlabArray);
}
void CKern::extractParamFromMxArray(const mxArray* matlabArray)
{
  inputDim = mxArrayExtractIntField(matlabArray, "inputDimension");
  nParams = mxArrayExtractIntField(matlabArray, "nParams");
  string pName;
  for(unsigned int i=0; i<nParams; i++)
  {
    pName=getParamName(i);
    setParam(mxArrayExtractDoubleField(matlabArray, pName), i);  
  }
}
void CKern::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)nParams));
  string pName;
  for(unsigned int i=0; i<nParams; i++)
  {      
    pName = getParamName(i);
    mxAddField(matlabArray, pName.c_str());      
    mxSetField(matlabArray, 0, pName.c_str(), convertMxArray(getParam(i))); 
  }
}
void CArdKern::extractParamFromMxArray(const mxArray* matlabArray)
{
  setInputDim(mxArrayExtractIntField(matlabArray, "inputDimension"));
  DIMENSIONMATCH(nParams == mxArrayExtractIntField(matlabArray, "nParams"));
  string pName;
  for(unsigned int i=0; i<nParams; i++)
  {
    pName=getParamName(i);
    if(pName!="inputScale")
      setParam(mxArrayExtractDoubleField(matlabArray, pName), i);  
  }
  scales.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "inputScales"));
}
void CArdKern::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)nParams));
  string pName;
  for(unsigned int i=0; i<nParams; i++)
  {      
    pName = getParamName(i);
    if(pName!="inputScale")
    {
      mxAddField(matlabArray, pName.c_str());      
      mxSetField(matlabArray, 0, pName.c_str(), convertMxArray(getParam(i))); 
    }
  }
  mxAddField(matlabArray, "inputScales");
  mxSetField(matlabArray, 0, "inputScales", scales.toMxArray());
}
void CComponentKern::extractParamFromMxArray(const mxArray* matlabArray)
{
  nParams=0;
  mxArray* compArray=mxArrayExtractMxArrayField(matlabArray, "comp");
  unsigned int numComps = mxGetNumberOfElements(compArray);
  mxArray* compElement;
  string kernType;
  unsigned int kernNum = 0;
  for(unsigned int i=0; i<numComps; i++)
  {
    // TODO this is a pretty ugly way of doing things ... perhaps dynamic casting can be used?
    compElement = mxGetCell(compArray, i);
    kernType = mxArrayExtractStringField(compElement, "type");
    if(kernType == "lin")
      kernNum = addKern(new CLinKern(getInputDim()));
    else if(kernType == "rbf")  
      kernNum = addKern(new CRbfKern(getInputDim()));
    else if(kernType == "eq" || kernType == "ExpQuad" || kernType == "expquad")
      kernNum = addKern(new CEqKern(getInputDim()));
    else if(kernType == "ratquad")  
      kernNum = addKern(new CRatQuadKern(getInputDim()));
    else if(kernType == "matern32")  
      kernNum = addKern(new CMatern32Kern(getInputDim()));
    else if(kernType == "matern52")  
      kernNum = addKern(new CMatern52Kern(getInputDim()));
    else if(kernType == "mlp")  
      kernNum = addKern(new CMlpKern(getInputDim()));
    else if(kernType == "poly")  
      kernNum = addKern(new CPolyKern(getInputDim()));
    else if(kernType == "polyard")  
      kernNum = addKern(new CPolyardKern(getInputDim()));
    else if(kernType == "linard")  
      kernNum = addKern(new CLinardKern(getInputDim()));
    else if(kernType == "rbfard")  
      kernNum = addKern(new CRbfardKern(getInputDim()));
    else if(kernType == "mlpard")  
      kernNum = addKern(new CMlpardKern(getInputDim()));
    else if(kernType == "white")
      kernNum = addKern(new CWhiteKern(getInputDim()));
    else if(kernType == "bias")
      kernNum = addKern(new CBiasKern(getInputDim()));
    else if(kernType == "cmpnd")
      kernNum = addKern(new CCmpndKern(getInputDim()));
    else if(kernType == "tensor")
      kernNum = addKern(new CTensorKern(getInputDim()));
    else
      throw ndlexceptions::Error("Unknown kernel type.");
    components[kernNum]->fromMxArray(compElement);
  }
}
void CComponentKern::addParamToMxArray(mxArray* matlabArray) const
{
  // Add comp field to mxArray.
  size_t dims[1];
  mxAddField(matlabArray, "comp");
  dims[0] = components.size();
  mxArray* compArray = mxCreateCellArray(1, dims);
  mxSetField(matlabArray, 0, "comp", 
	     compArray); 
  mxArray* currentKern;
  for(size_t i=0; i<components.size(); i++)
  {
    currentKern = components[i]->toMxArray();
    mxAddField(currentKern, "index");
    mxSetCell(compArray, i, currentKern);
  }
  
}
#endif /* _NDLMATLAB */

/* Psi statistics under diagonal Gaussian q(X) — CIP-0017 Phase 1.
   Closed forms follow GPy kern.src.psi_comp (rbf / linear) and kern.src.add. */
namespace {

void checkPsiInputs(const CMatrix& mu, const CMatrix& S, unsigned int inputDim)
{
  DIMENSIONMATCH(mu.dimensionsMatch(S));
  DIMENSIONMATCH(mu.getCols() == inputDim);
  for(unsigned int n = 0; n < mu.getRows(); n++)
    for(unsigned int q = 0; q < inputDim; q++)
      BOUNDCHECK(S.getVal(n, q) > 0.0);
}

void checkPsiZ(const CMatrix& Z, unsigned int inputDim)
{
  DIMENSIONMATCH(Z.getCols() == inputDim);
}

void zeroGradOutputs(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                     unsigned int nParams, unsigned int N, unsigned int M, unsigned int Q,
                     bool add)
{
  if(!add)
  {
    gParams.resize(1, nParams);
    gParams.zeros();
    gZ.resize(M, Q);
    gZ.zeros();
    gMu.resize(N, Q);
    gMu.zeros();
    gS.resize(N, Q);
    gS.zeros();
  }
  else
  {
    DIMENSIONMATCH(gParams.getRows() == 1 && gParams.getCols() == nParams);
    DIMENSIONMATCH(gZ.getRows() == M && gZ.getCols() == Q);
    DIMENSIONMATCH(gMu.dimensionsMatch(gS));
    DIMENSIONMATCH(gMu.getRows() == N && gMu.getCols() == Q);
  }
}

// lengthscale2[q] = ℓ_q². Gradients returned w.r.t. variance and ℓ_q (GPy convention).
void rbfPsi1(CMatrix& Psi1, double variance, const vector<double>& lengthscale2,
             const CMatrix& mu, const CMatrix& S, const CMatrix& Z)
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = mu.getCols();
  Psi1.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
  {
    double logdenom = 0.0;
    for(unsigned int q = 0; q < Q; q++)
      logdenom += log(S.getVal(n, q) / lengthscale2[q] + 1.0);
    for(unsigned int m = 0; m < M; m++)
    {
      double quad = 0.0;
      for(unsigned int q = 0; q < Q; q++)
      {
        const double d = mu.getVal(n, q) - Z.getVal(m, q);
        quad += d * d / (S.getVal(n, q) + lengthscale2[q]);
      }
      Psi1.setVal(variance * exp(-0.5 * (logdenom + quad)), n, m);
    }
  }
}

void rbfPsi2(CMatrix& Psi2, double variance, const vector<double>& lengthscale2,
             const CMatrix& mu, const CMatrix& S, const CMatrix& Z)
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = mu.getCols();
  Psi2.resize(M, M);
  Psi2.zeros();
  const double var2 = variance * variance;
  for(unsigned int n = 0; n < N; n++)
  {
    double logdenom = 0.0;
    for(unsigned int q = 0; q < Q; q++)
      logdenom += log(2.0 * S.getVal(n, q) / lengthscale2[q] + 1.0);
    logdenom *= -0.5;
    for(unsigned int m1 = 0; m1 < M; m1++)
    {
      for(unsigned int m2 = 0; m2 <= m1; m2++)
      {
        double exp1 = 0.0;
        double exp2 = 0.0;
        for(unsigned int q = 0; q < Q; q++)
        {
          const double z1 = Z.getVal(m1, q);
          const double z2 = Z.getVal(m2, q);
          const double l2 = lengthscale2[q];
          const double Snq = S.getVal(n, q);
          const double dZ = z1 - z2;
          const double zhat = 0.5 * (z1 + z2);
          const double muq = mu.getVal(n, q);
          exp1 += (dZ * dZ) / l2;
          const double denom = 1.0 / (2.0 * Snq + l2);
          exp2 += -(muq - zhat) * (muq - zhat) * denom;
        }
        exp1 *= -0.25;
        const double val = var2 * exp(logdenom + exp1 + exp2);
        Psi2.addVal(val, m1, m2);
        if(m1 != m2)
          Psi2.addVal(val, m2, m1);
      }
    }
  }
  Psi2.setSymmetric(true);
}

// Per-n psi2 element (for gradients); returns var^2 * exp(...).
double rbfPsi2nElement(double variance, const vector<double>& lengthscale2,
                       const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                       unsigned int n, unsigned int m1, unsigned int m2)
{
  const unsigned int Q = mu.getCols();
  double logdenom = 0.0;
  double exp1 = 0.0;
  double exp2 = 0.0;
  for(unsigned int q = 0; q < Q; q++)
  {
    const double l2 = lengthscale2[q];
    const double Snq = S.getVal(n, q);
    logdenom += log(2.0 * Snq / l2 + 1.0);
    const double z1 = Z.getVal(m1, q);
    const double z2 = Z.getVal(m2, q);
    const double dZ = z1 - z2;
    const double zhat = 0.5 * (z1 + z2);
    const double muq = mu.getVal(n, q);
    exp1 += (dZ * dZ) / l2;
    const double denom = 1.0 / (2.0 * Snq + l2);
    exp2 += -(muq - zhat) * (muq - zhat) * denom;
  }
  logdenom *= -0.5;
  exp1 *= -0.25;
  return variance * variance * exp(logdenom + exp1 + exp2);
}

void rbfGradPsi(vector<double>& gLengthscale,
                CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                double& gVariance,
                const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                double variance, const vector<double>& lengthscale2,
                const CMatrix& mu, const CMatrix& S, const CMatrix& Z)
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = mu.getCols();
  gVariance = 0.0;
  gLengthscale.assign(Q, 0.0);
  gZ.zeros();
  gMu.zeros();
  gS.zeros();

  CMatrix Psi1(N, M);
  rbfPsi1(Psi1, variance, lengthscale2, mu, S, Z);

  // psi0: each diagonal is variance → dL/dvar += sum_n dL_dpsi0_n
  for(unsigned int n = 0; n < N; n++)
    gVariance += dL_dpsi0.getVal(n);

  // psi1 grads (GPy _psi1compDer)
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      const double Lpsi1 = dL_dpsi1.getVal(n, m) * Psi1.getVal(n, m);
      gVariance += Lpsi1 / variance;
      for(unsigned int q = 0; q < Q; q++)
      {
        const double l2 = lengthscale2[q];
        const double l = sqrt(l2);
        const double Snq = S.getVal(n, q);
        const double denom = 1.0 / (Snq + l2);
        const double Zmu = Z.getVal(m, q) - mu.getVal(n, q);
        const double Zmu2_denom = Zmu * Zmu * denom;
        gMu.addVal(Lpsi1 * Zmu * denom, n, q); // wait: GPy uses (Z-mu)? _dL_dmu = einsum Lpsi1, Zmu, denom with Zmu=Z-mu
        // GPy: Zmu = Z[None,:,:] - mu[:,None,:] so Z - mu
        // _dL_dmu = einsum("nm,nmq,nq->nq", Lpsi1, Zmu, denom)  → contributes (Z-mu)*denom
        // But then they return that as dmu — actually derivative of exp(-0.5 (mu-Z)^2/(S+l2)) w.r.t mu is
        // -(mu-Z)/(S+l2) * psi1 = (Z-mu)/(S+l2) * psi1. Yes Zmu = Z-mu.
        gS.addVal(0.5 * Lpsi1 * (Zmu2_denom - 1.0) * denom, n, q);
        gZ.addVal(-Lpsi1 * Zmu * denom, m, q);
        // dL/dl : einsum Lpsi1, (Zmu2_denom + S/l2), denom * lengthscale
        gLengthscale[q] += Lpsi1 * (Zmu2_denom + Snq / l2) * denom * l;
      }
    }
  }
  // Fix: I used addVal for gMu incorrectly in comment block — set the mu grad properly in loop.
  // The loop above already did gMu.addVal(Lpsi1 * Zmu * denom) — correct.

  // psi2 grads — accumulate over n without storing N x M x M
  // Symmetrise dL_dpsi2
  CMatrix dL2(M, M);
  for(unsigned int m1 = 0; m1 < M; m1++)
    for(unsigned int m2 = 0; m2 < M; m2++)
      dL2.setVal(0.5 * (dL_dpsi2.getVal(m1, m2) + dL_dpsi2.getVal(m2, m1)), m1, m2);

  vector<double> lengthscale(Q);
  for(unsigned int q = 0; q < Q; q++)
    lengthscale[q] = sqrt(lengthscale2[q]);

  for(unsigned int n = 0; n < N; n++)
  {
    // Build psi2_n (M x M) and Lpsi2 = dL2 * psi2_n
    CMatrix psi2n(M, M);
    CMatrix Lpsi2(M, M);
    double Lpsi2sum = 0.0;
    for(unsigned int m1 = 0; m1 < M; m1++)
    {
      for(unsigned int m2 = 0; m2 <= m1; m2++)
      {
        const double p = rbfPsi2nElement(variance, lengthscale2, mu, S, Z, n, m1, m2);
        psi2n.setVal(p, m1, m2);
        if(m1 != m2)
          psi2n.setVal(p, m2, m1);
      }
    }
    for(unsigned int m1 = 0; m1 < M; m1++)
      for(unsigned int m2 = 0; m2 < M; m2++)
      {
        const double Lp = dL2.getVal(m1, m2) * psi2n.getVal(m1, m2);
        Lpsi2.setVal(Lp, m1, m2);
        Lpsi2sum += Lp;
      }

    gVariance += Lpsi2sum * 2.0 / variance;

    // Accumulators per GPy _psi2compDer
    vector<double> Lpsi2Z(Q, 0.0);
    vector<double> Lpsi2Z2(Q, 0.0);
    vector<double> Lpsi2Z2p(Q, 0.0);
    for(unsigned int m1 = 0; m1 < M; m1++)
    {
      for(unsigned int m2 = 0; m2 < M; m2++)
      {
        const double Lp = Lpsi2.getVal(m1, m2);
        for(unsigned int q = 0; q < Q; q++)
        {
          const double z2 = Z.getVal(m2, q);
          const double z1 = Z.getVal(m1, q);
          Lpsi2Z[q] += Lp * z2;           // sum_{m1,m2} Lpsi2[m1,m2] * Z[m2,q] then... GPy: tmp=Lpsi2.reshape(N*M,M).dot(Z) then sum over m1
          // Actually GPy: tmp[n,m1,q] = sum_m2 Lpsi2[n,m1,m2]*Z[m2,q]; Lpsi2Z = tmp.sum(1) = sum_{m1,m2} L*Z[m2]
          Lpsi2Z2[q] += Lp * z2 * z2;
          Lpsi2Z2p[q] += Lp * z1 * z2;
        }
      }
    }
    // Fix Lpsi2Z: should be sum_{m1,m2} Lpsi2[m1,m2] * Z[m2,q]
    // Already correct as written.

    for(unsigned int q = 0; q < Q; q++)
    {
      const double l2 = lengthscale2[q];
      const double l = lengthscale[q];
      const double Snq = S.getVal(n, q);
      const double denom = 1.0 / (2.0 * Snq + l2);
      const double denom2 = denom * denom;
      const double muq = mu.getVal(n, q);
      const double Lpsi2Zhat = Lpsi2Z[q];
      const double Lpsi2Zhat2 = 0.5 * (Lpsi2Z2[q] + Lpsi2Z2p[q]);

      gMu.addVal((-2.0 * denom) * (muq * Lpsi2sum - Lpsi2Zhat), n, q);
      gS.addVal(2.0 * denom2 * (muq * muq * Lpsi2sum - 2.0 * muq * Lpsi2Zhat + Lpsi2Zhat2)
                - denom * Lpsi2sum, n, q);

      gLengthscale[q] += 2.0 * l * (
        (Snq / l2 * denom + (muq * denom) * (muq * denom)) * Lpsi2sum
        + (Lpsi2Z2[q] - Lpsi2Z2p[q]) / (2.0 * l2 * l2)
        - (2.0 * muq * denom2) * Lpsi2Zhat
        + denom2 * Lpsi2Zhat2);
    }

    // dL/dZ from psi2 — GPy formula
    for(unsigned int m = 0; m < M; m++)
    {
      for(unsigned int q = 0; q < Q; q++)
      {
        const double l2 = lengthscale2[q];
        const double Snq = S.getVal(n, q);
        const double denom = 1.0 / (2.0 * Snq + l2);
        const double muq = mu.getVal(n, q);
        const double Zm = Z.getVal(m, q);

        // Terms from GPy (per n, accumulate):
        // -Lpsi2_N.sum(0)[:,None]*Z/l2 + Lpsi2_N.dot(Z)/l2 + 2*Lpsi2_M.T.dot(mu*denom)
        // - Lpsi2_M.T.dot(denom)*Z - (Lpsi2.reshape ... denom * Z).sum(1)
        double Lpsi2_N_row_sum = 0.0; // sum over m2 of Lpsi2[m, m2]  → for fixed first index as row of Lpsi2_N
        double Lpsi2_N_dot_Z = 0.0;
        double Lpsi2_M_col = 0.0; // sum over m1 of Lpsi2[m1, m]
        double cross = 0.0;
        for(unsigned int m2 = 0; m2 < M; m2++)
        {
          Lpsi2_N_row_sum += Lpsi2.getVal(m, m2);
          Lpsi2_N_dot_Z += Lpsi2.getVal(m, m2) * Z.getVal(m2, q);
          Lpsi2_M_col += Lpsi2.getVal(m2, m);
          cross += Lpsi2.getVal(m, m2) * denom * Z.getVal(m2, q);
        }
        // Also need -Lpsi2_N.sum(0) * Z / l2 where Lpsi2_N.sum(0) is sum_{m1} Lpsi2[m1,m] wait
        // Lpsi2_N = Lpsi2.sum(0) in numpy for NxMxM → for our MxM, Lpsi2_N[m2,q related] = sum_m1 Lpsi2[m1,m2]
        double sum_over_m1 = 0.0;
        for(unsigned int m1 = 0; m1 < M; m1++)
          sum_over_m1 += Lpsi2.getVal(m1, m);

        double g = -sum_over_m1 * Zm / l2
                   + Lpsi2_N_dot_Z / l2
                   + 2.0 * Lpsi2_M_col * muq * denom
                   - Lpsi2_M_col * denom * Zm
                   - cross;
        gZ.addVal(g, m, q);
      }
    }
  }
}

} // namespace

// ---- CKern defaults ----
double CKern::psi0(const CMatrix& /*mu*/, const CMatrix& /*S*/) const
{
  throw ndlexceptions::NotImplementedError("psi0 not implemented for kernel type " + getType());
}
void CKern::psi1(CMatrix& /*Psi1*/, const CMatrix& /*mu*/, const CMatrix& /*S*/, const CMatrix& /*Z*/) const
{
  throw ndlexceptions::NotImplementedError("psi1 not implemented for kernel type " + getType());
}
void CKern::psi2(CMatrix& /*Psi2*/, const CMatrix& /*mu*/, const CMatrix& /*S*/, const CMatrix& /*Z*/) const
{
  throw ndlexceptions::NotImplementedError("psi2 not implemented for kernel type " + getType());
}
void CKern::getGradPsi(CMatrix& /*gParams*/, CMatrix& /*gZ*/, CMatrix& /*gMu*/, CMatrix& /*gS*/,
                       const CMatrix& /*dL_dpsi0*/, const CMatrix& /*dL_dpsi1*/, const CMatrix& /*dL_dpsi2*/,
                       const CMatrix& /*mu*/, const CMatrix& /*S*/, const CMatrix& /*Z*/,
                       bool /*add*/) const
{
  throw ndlexceptions::NotImplementedError("getGradPsi not implemented for kernel type " + getType());
}

void CKern::wnCrossCompute(CMatrix& /*Kfu*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/) const
{
  throw ndlexceptions::NotImplementedError("wnCrossCompute not implemented for kernel type " + getType());
}
void CKern::wnGradParams(CMatrix& /*g*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError("wnGradParams not implemented for kernel type " + getType());
}
void CKern::wnGradZ(CMatrix& /*gZ*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                    const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError("wnGradZ not implemented for kernel type " + getType());
}
void CKern::wnGradX(CMatrix& /*gX*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                    const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError("wnGradX not implemented for kernel type " + getType());
}
void CKern::wnGradS(CMatrix& /*gS*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                    const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError("wnGradS not implemented for kernel type " + getType());
}

void CKern::greensCrossCompute(CMatrix& /*Kfu*/, const CMatrix& /*X*/, const CMatrix& /*Z*/,
                               double /*s*/) const
{
  throw ndlexceptions::NotImplementedError(
      "greensCrossCompute not implemented for kernel type " + getType());
}
void CKern::greensGradParams(CMatrix& /*g*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                             const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "greensGradParams not implemented for kernel type " + getType());
}
void CKern::greensGradZ(CMatrix& /*gZ*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                        const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "greensGradZ not implemented for kernel type " + getType());
}
void CKern::greensGradX(CMatrix& /*gX*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                        const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "greensGradX not implemented for kernel type " + getType());
}
void CKern::greensGradS(CMatrix& /*gS*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                        const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "greensGradS not implemented for kernel type " + getType());
}

void CKern::wsCrossCompute(CMatrix& /*Kfu*/, const CMatrix& /*X*/, const CMatrix& /*W*/,
                           double /*s*/) const
{
  throw ndlexceptions::NotImplementedError(
      "wsCrossCompute not implemented for kernel type " + getType());
}
void CKern::wsGradParams(CMatrix& /*g*/, const CMatrix& /*X*/, const CMatrix& /*W*/, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "wsGradParams not implemented for kernel type " + getType());
}
void CKern::wsGradW(CMatrix& /*gW*/, const CMatrix& /*X*/, const CMatrix& /*W*/, double /*s*/,
                    const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "wsGradW not implemented for kernel type " + getType());
}
void CKern::wsGradX(CMatrix& /*gX*/, const CMatrix& /*X*/, const CMatrix& /*W*/, double /*s*/,
                    const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "wsGradX not implemented for kernel type " + getType());
}
void CKern::wsGradS(CMatrix& /*gS*/, const CMatrix& /*X*/, const CMatrix& /*W*/, double /*s*/,
                    const CMatrix& /*gKfu*/) const
{
  throw ndlexceptions::NotImplementedError(
      "wsGradS not implemented for kernel type " + getType());
}

// CIP-001C Phase 3: Bochner spectral density + Hankel / cosine K_fu path.
namespace {

double hatPsiFourier(double rho, double s, unsigned int d)
{
  return pow(4.0 * M_PI * s * s, 0.25 * (double)d) * exp(-0.5 * s * s * rho * rho);
}

double maternSpectralDensity(double rho, double variance, double ell, double nu, unsigned int d)
{
  // GPML angular-frequency form.
  const double twoNu = 2.0 * nu;
  const double pref = variance * pow(2.0, (double)d) * pow(M_PI, 0.5 * (double)d)
    * ndlutil::gamma(nu + 0.5 * (double)d) * pow(twoNu, nu)
    / (ndlutil::gamma(nu) * pow(ell, 2.0 * nu));
  return pref * pow(twoNu / (ell * ell) + rho * rho, -(nu + 0.5 * (double)d));
}

double besselJnu(double nu, double x)
{
  if(nu == 0.0)
    return ::j0(x);
  if(nu == 1.0)
    return ::j1(x);
  return std::cyl_bessel_j(nu, x);
}

// Composite Simpson ∫_0^{ρmax} on a uniform ρ grid (n must be odd ⇒ even panels).
double simpsonRho(const vector<double>& y, double dRho)
{
  const unsigned int n = y.size();
  if(n < 2)
    return 0.0;
  if(n == 2)
    return 0.5 * dRho * (y[0] + y[1]);
  // Ensure even number of panels: if n even, drop last point (tiny tail).
  unsigned int nUse = n;
  if(((nUse - 1) % 2) != 0)
    nUse--;
  double acc = y[0] + y[nUse - 1];
  for(unsigned int i = 1; i + 1 < nUse; i++)
    acc += ((i % 2) == 0) ? 2.0 * y[i] : 4.0 * y[i];
  return acc * dRho / 3.0;
}

double hankelKfuAtR(const vector<double>& F, double dRho, double r, unsigned int d)
{
  const unsigned int nRho = F.size();
  const double nu = 0.5 * (double)d - 1.0;
  vector<double> y(nRho);
  if(d == 1)
  {
    for(unsigned int i = 0; i < nRho; i++)
    {
      const double rho = dRho * (double)i;
      y[i] = F[i] * cos(rho * r);
    }
    return simpsonRho(y, dRho) / M_PI;
  }

  // General: (2π)^{-d/2} r^{-ν} ∫ F(ρ) ρ^{ν+1} J_ν(ρ r) dρ
  // At r→0: r^{-ν} J_ν(ρ r) → (ρ/2)^ν / Γ(ν+1)
  const double pref = pow(2.0 * M_PI, -0.5 * (double)d);
  if(r < 1e-14)
  {
    if(nu <= 0.0)
    {
      for(unsigned int i = 0; i < nRho; i++)
      {
        const double rho = dRho * (double)i;
        y[i] = F[i] * rho; // ν=0 → ρ^{1} J_0(0) = ρ
      }
      return pref * simpsonRho(y, dRho);
    }
    const double limScale = 1.0 / (ndlutil::gamma(nu + 1.0) * pow(2.0, nu));
    for(unsigned int i = 0; i < nRho; i++)
    {
      const double rho = dRho * (double)i;
      y[i] = F[i] * pow(rho, 2.0 * nu + 1.0);
    }
    return pref * limScale * simpsonRho(y, dRho);
  }

  const double rInvNu = pow(r, -nu);
  for(unsigned int i = 0; i < nRho; i++)
  {
    const double rho = dRho * (double)i;
    y[i] = F[i] * pow(rho, nu + 1.0) * besselJnu(nu, rho * r);
  }
  return pref * rInvNu * simpsonRho(y, dRho);
}

void buildFGrid(const CStationaryKern& kern, double s, unsigned int nRho,
                vector<double>& F, double& dRho)
{
  const unsigned int d = kern.getInputDim();
  auto evalF = [&](double rho) -> double
  {
    const double S = kern.spectralDensity(rho);
    if(!(S > 0.0))
      return 0.0;
    return sqrt(S) * hatPsiFourier(rho, s, d);
  };

  // ˆψ supplies a Gaussian tail; pick ρ_max from F decay so the grid stays dense.
  const double F0 = evalF(0.0);
  const double thr = 1e-20 * max(fabs(F0), 1e-300);
  double rhoHi = 1.0 / max(s, 1e-12);
  for(int k = 0; k < 60; k++)
  {
    if(evalF(rhoHi) < thr)
      break;
    rhoHi *= 1.2;
  }
  // Binary search for a tight cutoff once an upper bracket is known.
  double rhoLo = 0.0;
  for(int k = 0; k < 40; k++)
  {
    const double mid = 0.5 * (rhoLo + rhoHi);
    if(evalF(mid) > thr)
      rhoLo = mid;
    else
      rhoHi = mid;
  }
  double rhoMax = max(rhoHi * 1.05, 8.0 / max(s, 1e-12));

  // Odd nRho for Simpson (even number of panels).
  unsigned int nUse = nRho;
  if(((nUse - 1) % 2) != 0)
    nUse++;
  dRho = rhoMax / (double)(nUse - 1);
  F.resize(nUse);
  for(unsigned int i = 0; i < nUse; i++)
    F[i] = evalF(dRho * (double)i);
}

const unsigned int WN_HANKEL_NRHO = 4097; // odd → even Simpson panels
const double WN_FD_EPS = 1e-6;
const double WN_DKDR_EPS = 1e-5;

} // namespace

double CStationaryKern::spectralDensity(double /*rho*/) const
{
  throw ndlexceptions::NotImplementedError("spectralDensity not implemented for kernel type " + getType());
}

double CStationaryKern::spectralLengthScale() const
{
  return 1.0 / scaleFactor();
}

double CRbfKern::spectralDensity(double rho) const
{
  const unsigned int d = getInputDim();
  const double ell = spectralLengthScale();
  return variance * pow(2.0 * M_PI * ell * ell, 0.5 * (double)d) * exp(-0.5 * ell * ell * rho * rho);
}

double CExpKern::spectralDensity(double rho) const
{
  // Matérn ν=1/2; ℓ = 1/inverseWidth = 1/scaleFactor().
  return maternSpectralDensity(rho, variance, spectralLengthScale(), 0.5, getInputDim());
}

double CMatern32Kern::spectralLengthScale() const
{
  return lengthScale;
}

double CMatern32Kern::spectralDensity(double rho) const
{
  return maternSpectralDensity(rho, variance, lengthScale, 1.5, getInputDim());
}

double CMatern52Kern::spectralLengthScale() const
{
  return lengthScale;
}

double CMatern52Kern::spectralDensity(double rho) const
{
  return maternSpectralDensity(rho, variance, lengthScale, 2.5, getInputDim());
}

// CIP-0029 Phase 2: causal Matérn / Exp × Gaussian windows (1-D).
namespace {

inline double greensPsiNorm1d(double s)
{
  return pow(4.0 * M_PI * s * s, 0.25);
}

// Shared FD grads for greensCrossCompute (mirrors Hankel wnGrad* FD).
const double GREENS_FD_EPS = 1e-6;

void greensFdGradParams(CKern& kern, CMatrix& g, const CMatrix& X, const CMatrix& Z,
                        double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int nParams = kern.getNumParams();
  DIMENSIONMATCH(g.getCols() == nParams);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  g.zeros();
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  for(unsigned int p = 0; p < nParams; p++)
  {
    const double p0 = kern.getParam(p);
    const double h = GREENS_FD_EPS * max(1.0, fabs(p0));
    kern.setParam(p0 + h, p);
    kern.greensCrossCompute(Kp, X, Z, s);
    kern.setParam(p0 - h, p);
    kern.greensCrossCompute(Km, X, Z, s);
    kern.setParam(p0, p);
    double gp = 0.0;
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M; m++)
        gp += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
    g.setVal(gp, p);
  }
}

void greensFdGradZ(const CKern& kern, CMatrix& gZ, const CMatrix& X, CMatrix& Z,
                   double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(Z.getCols() == 1);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gZ.resize(M, 1);
  gZ.zeros();
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  for(unsigned int m = 0; m < M; m++)
  {
    const double z0 = Z.getVal(m, 0);
    const double h = GREENS_FD_EPS * max(1.0, fabs(z0));
    Z.setVal(z0 + h, m, 0);
    kern.greensCrossCompute(Kp, X, Z, s);
    Z.setVal(z0 - h, m, 0);
    kern.greensCrossCompute(Km, X, Z, s);
    Z.setVal(z0, m, 0);
    double acc = 0.0;
    for(unsigned int n = 0; n < N; n++)
      acc += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
    gZ.setVal(acc, m, 0);
  }
}

void greensFdGradX(const CKern& kern, CMatrix& gX, CMatrix& X, const CMatrix& Z,
                   double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  gX.resize(N, 1);
  gX.zeros();
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  for(unsigned int n = 0; n < N; n++)
  {
    const double x0 = X.getVal(n, 0);
    const double h = GREENS_FD_EPS * max(1.0, fabs(x0));
    X.setVal(x0 + h, n, 0);
    kern.greensCrossCompute(Kp, X, Z, s);
    X.setVal(x0 - h, n, 0);
    kern.greensCrossCompute(Km, X, Z, s);
    X.setVal(x0, n, 0);
    double acc = 0.0;
    for(unsigned int m = 0; m < M; m++)
      acc += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
    gX.setVal(acc, n, 0);
  }
}

void greensFdGradS(const CKern& kern, CMatrix& gS, const CMatrix& X, const CMatrix& Z,
                   double s, const CMatrix& gKfu)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gS.resize(1, 1);
  const double h = GREENS_FD_EPS * max(1.0, fabs(s));
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  kern.greensCrossCompute(Kp, X, Z, s + h);
  kern.greensCrossCompute(Km, X, Z, max(s - h, 1e-12));
  double acc = 0.0;
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      acc += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
  gS.setVal(acc, 0, 0);
}

// Causal g*ψ for Matérn ν=1/2,3/2,5/2 (polynomial × exp decay).
// degree 0: g ∝ e^{-λt}; 1: ∝ t e^{-λt}; 2: ∝ t² e^{-λt}.
void maternCausalGreensKfu(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
                           double s, double variance, double lambda, int polyDegree)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(X.getCols() == 1 && Z.getCols() == 1);
  BOUNDCHECK(s > 0.0);
  BOUNDCHECK(lambda > 0.0);
  Kfu.resize(N, M);
  const double sigma = sqrt(variance);
  const double c = greensPsiNorm1d(s);
  double amp = 0.0;
  if(polyDegree == 0)
    amp = sigma * sqrt(2.0 * lambda) * c;
  else if(polyDegree == 1)
    amp = sigma * 2.0 * pow(lambda, 1.5) * c;
  else if(polyDegree == 2)
    amp = sigma * sqrt(4.0 / 3.0) * pow(lambda, 2.5) * c;
  else
    throw ndlexceptions::Error("maternCausalGreensKfu: unsupported polyDegree");

  for(unsigned int n = 0; n < N; n++)
  {
    const double x = X.getVal(n, 0);
    for(unsigned int m = 0; m < M; m++)
    {
      const double z = Z.getVal(m, 0);
      const double mu = z + lambda * s * s;
      const double alpha = (x - mu) / s;
      const double Phi = ndlutil::cumGaussian(alpha);
      const double phi = ndlutil::ngaussian(alpha);
      const double efac = exp(-lambda * (x - z) + 0.5 * lambda * lambda * s * s);
      double shape = 0.0;
      if(polyDegree == 0)
        shape = Phi;
      else if(polyDegree == 1)
      {
        // ∫ (x-τ) e^{λτ} N = efac_raw * [(x-μ)Φ + s φ]; overall efac includes e^{-λx+...}
        shape = (x - mu) * Phi + s * phi;
      }
      else
      {
        // ∫ (x-τ)² e^{λτ} N; I0,I1,I2 without outer e^{-λx}
        const double I0 = Phi;
        const double I1 = mu * Phi - s * phi;
        const double I2 = (mu * mu + s * s) * Phi - s * (mu + x) * phi;
        shape = x * x * I0 - 2.0 * x * I1 + I2;
      }
      Kfu.setVal(amp * efac * shape, n, m);
    }
  }
}

} // namespace

void CExpKern::greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
                                  double s) const
{
  // λ = inverseWidth (scaleFactor); g(t)=σ√(2λ) e^{-λt} 1_{t≥0}.
  maternCausalGreensKfu(Kfu, X, Z, s, variance, inverseWidth, 0);
}
void CExpKern::greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  greensFdGradParams(const_cast<CExpKern&>(*this), g, X, Z, s, gKfu);
}
void CExpKern::greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                           const CMatrix& gKfu) const
{
  CMatrix Zmut(Z);
  greensFdGradZ(*this, gZ, X, Zmut, s, gKfu);
}
void CExpKern::greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                           const CMatrix& gKfu) const
{
  CMatrix Xmut(X);
  greensFdGradX(*this, gX, Xmut, Z, s, gKfu);
}
void CExpKern::greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                           const CMatrix& gKfu) const
{
  greensFdGradS(*this, gS, X, Z, s, gKfu);
}

void CMatern32Kern::greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
                                       double s) const
{
  maternCausalGreensKfu(Kfu, X, Z, s, variance, scaleFactor(), 1);
}
void CMatern32Kern::greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                                     const CMatrix& gKfu) const
{
  greensFdGradParams(const_cast<CMatern32Kern&>(*this), g, X, Z, s, gKfu);
}
void CMatern32Kern::greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  CMatrix Zmut(Z);
  greensFdGradZ(*this, gZ, X, Zmut, s, gKfu);
}
void CMatern32Kern::greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  CMatrix Xmut(X);
  greensFdGradX(*this, gX, Xmut, Z, s, gKfu);
}
void CMatern32Kern::greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  greensFdGradS(*this, gS, X, Z, s, gKfu);
}

void CMatern52Kern::greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
                                       double s) const
{
  maternCausalGreensKfu(Kfu, X, Z, s, variance, scaleFactor(), 2);
}
void CMatern52Kern::greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                                     const CMatrix& gKfu) const
{
  greensFdGradParams(const_cast<CMatern52Kern&>(*this), g, X, Z, s, gKfu);
}
void CMatern52Kern::greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  CMatrix Zmut(Z);
  greensFdGradZ(*this, gZ, X, Zmut, s, gKfu);
}
void CMatern52Kern::greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  CMatrix Xmut(X);
  greensFdGradX(*this, gX, Xmut, Z, s, gKfu);
}
void CMatern52Kern::greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const
{
  greensFdGradS(*this, gS, X, Z, s, gKfu);
}

void CStationaryKern::wnCrossComputeHankel(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(X.getCols() == d && Z.getCols() == d);
  BOUNDCHECK(s > 0.0);
  Kfu.resize(N, M);

  // Build F(ρ)=√S(ρ)ˆψ(ρ) once; evaluate the Hankel/cosine integral at each ||x-z||.
  vector<double> F;
  double dRho = 0.0;
  buildFGrid(*this, s, WN_HANKEL_NRHO, F, dRho);

  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      Kfu.setVal(hankelKfuAtR(F, dRho, sqrt(r2), d), n, m);
    }
  }
}

void CStationaryKern::wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const
{
  wnCrossComputeHankel(Kfu, X, Z, s);
}

void CStationaryKern::wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                                   const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(g.getCols() == nParams);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  g.zeros();

  // Central FD of Hankel Kfu w.r.t. each raw kernel parameter.
  CStationaryKern* mutableThis = const_cast<CStationaryKern*>(this);
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  for(unsigned int p = 0; p < nParams; p++)
  {
    const double p0 = getParam(p);
    const double h = WN_FD_EPS * max(1.0, fabs(p0));
    mutableThis->setParam(p0 + h, p);
    wnCrossComputeHankel(Kp, X, Z, s);
    mutableThis->setParam(p0 - h, p);
    wnCrossComputeHankel(Km, X, Z, s);
    mutableThis->setParam(p0, p);
    double gp = 0.0;
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M; m++)
        gp += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
    g.setVal(gp, p);
  }
}

void CStationaryKern::wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                              const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gZ.resize(M, d);
  gZ.zeros();

  vector<double> F;
  double dRho = 0.0;
  buildFGrid(*this, s, WN_HANKEL_NRHO, F, dRho);

  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      const double r = sqrt(r2);
      if(r < 1e-14)
        continue; // ∂K/∂z = 0 by radial symmetry at coincidence
      const double h = WN_DKDR_EPS * max(1.0, r);
      const double dKdr = (hankelKfuAtR(F, dRho, r + h, d) - hankelKfuAtR(F, dRho, max(r - h, 0.0), d))
        / ((r + h) - max(r - h, 0.0));
      const double scale = gKfu.getVal(n, m) * dKdr / r;
      // ∂r/∂z_q = -(x_q - z_q)/r  ⇒  ∂K/∂z_q = dKdr * ∂r/∂z_q
      for(unsigned int q = 0; q < d; q++)
        gZ.addVal(-scale * (X.getVal(n, q) - Z.getVal(m, q)), m, q);
    }
  }
}

void CStationaryKern::wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                              const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gX.resize(N, d);
  gX.zeros();

  vector<double> F;
  double dRho = 0.0;
  buildFGrid(*this, s, WN_HANKEL_NRHO, F, dRho);

  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      const double r = sqrt(r2);
      if(r < 1e-14)
        continue;
      const double h = WN_DKDR_EPS * max(1.0, r);
      const double dKdr = (hankelKfuAtR(F, dRho, r + h, d) - hankelKfuAtR(F, dRho, max(r - h, 0.0), d))
        / ((r + h) - max(r - h, 0.0));
      const double scale = gKfu.getVal(n, m) * dKdr / r;
      // ∂r/∂x_q = (x_q - z_q)/r
      for(unsigned int q = 0; q < d; q++)
        gX.addVal(scale * (X.getVal(n, q) - Z.getVal(m, q)), n, q);
    }
  }
}

void CStationaryKern::wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                              const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gS.resize(1, 1);
  const double h = WN_FD_EPS * max(1.0, fabs(s));
  CMatrix Kp(N, M);
  CMatrix Km(N, M);
  wnCrossComputeHankel(Kp, X, Z, s + h);
  wnCrossComputeHankel(Km, X, Z, max(s - h, 0.5 * s));
  const double hEff = (s + h) - max(s - h, 0.5 * s);
  double gs = 0.0;
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      gs += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / hEff;
  gS.setVal(gs, 0);
}

// ---- White ----
double CWhiteKern::psi0(const CMatrix& mu, const CMatrix& S) const
{
  checkPsiInputs(mu, S, getInputDim());
  return variance * static_cast<double>(mu.getRows());
}
void CWhiteKern::psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  Psi1.resize(mu.getRows(), Z.getRows());
  Psi1.zeros();
}
void CWhiteKern::psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  Psi2.resize(Z.getRows(), Z.getRows());
  Psi2.zeros();
  Psi2.setSymmetric(true);
}
void CWhiteKern::getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                            const CMatrix& dL_dpsi0, const CMatrix& /*dL_dpsi1*/, const CMatrix& /*dL_dpsi2*/,
                            const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                            bool add) const
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = getInputDim();
  checkPsiInputs(mu, S, Q);
  checkPsiZ(Z, Q);
  zeroGradOutputs(gParams, gZ, gMu, gS, getNumParams(), N, M, Q, add);
  double g = 0.0;
  for(unsigned int n = 0; n < N; n++)
    g += dL_dpsi0.getVal(n);
  gParams.addVal(g, 0);
}

// ---- Bias ----
double CBiasKern::psi0(const CMatrix& mu, const CMatrix& S) const
{
  checkPsiInputs(mu, S, getInputDim());
  return variance * static_cast<double>(mu.getRows());
}
void CBiasKern::psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  Psi1.resize(mu.getRows(), Z.getRows());
  Psi1.setVals(variance);
}
void CBiasKern::psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  Psi2.resize(Z.getRows(), Z.getRows());
  Psi2.setVals(variance * variance * static_cast<double>(mu.getRows()));
  Psi2.setSymmetric(true);
}
void CBiasKern::getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                           const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                           const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                           bool add) const
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = getInputDim();
  checkPsiInputs(mu, S, Q);
  checkPsiZ(Z, Q);
  zeroGradOutputs(gParams, gZ, gMu, gS, getNumParams(), N, M, Q, add);
  double g = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    g += dL_dpsi0.getVal(n);
    for(unsigned int m = 0; m < M; m++)
      g += dL_dpsi1.getVal(n, m);
  }
  // psi2 = N * var^2 * ones → d/dvar = 2 N var * sum(dL_dpsi2)
  double sum2 = 0.0;
  for(unsigned int m1 = 0; m1 < M; m1++)
    for(unsigned int m2 = 0; m2 < M; m2++)
      sum2 += dL_dpsi2.getVal(m1, m2);
  g += 2.0 * variance * static_cast<double>(N) * sum2;
  gParams.addVal(g, 0);
}

void CWhiteKern::wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double /*s*/) const
{
  Kfu.resize(X.getRows(), Z.getRows());
  Kfu.zeros();
}
void CWhiteKern::wnGradParams(CMatrix& g, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                              const CMatrix& /*gKfu*/) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  g.zeros();
}
void CWhiteKern::wnGradZ(CMatrix& gZ, const CMatrix& /*X*/, const CMatrix& Z, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  gZ.resize(Z.getRows(), Z.getCols());
  gZ.zeros();
}
void CWhiteKern::wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& /*Z*/, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  gX.resize(X.getRows(), X.getCols());
  gX.zeros();
}
void CWhiteKern::wnGradS(CMatrix& gS, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  gS.resize(1, 1);
  gS.zeros();
}

void CWhiteKern::greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
                                    double /*s*/) const
{
  Kfu.resize(X.getRows(), Z.getRows());
  Kfu.zeros();
}
void CWhiteKern::greensGradParams(CMatrix& g, const CMatrix& /*X*/, const CMatrix& /*Z*/,
                                  double /*s*/, const CMatrix& /*gKfu*/) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  g.zeros();
}
void CWhiteKern::greensGradZ(CMatrix& gZ, const CMatrix& /*X*/, const CMatrix& Z, double /*s*/,
                             const CMatrix& /*gKfu*/) const
{
  gZ.resize(Z.getRows(), Z.getCols());
  gZ.zeros();
}
void CWhiteKern::greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& /*Z*/, double /*s*/,
                             const CMatrix& /*gKfu*/) const
{
  gX.resize(X.getRows(), X.getCols());
  gX.zeros();
}
void CWhiteKern::greensGradS(CMatrix& gS, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/,
                             const CMatrix& /*gKfu*/) const
{
  gS.resize(1, 1);
  gS.zeros();
}

void CWhiteKern::wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double /*s*/) const
{
  Kfu.resize(X.getRows(), W.getRows());
  Kfu.zeros();
}
void CWhiteKern::wsGradParams(CMatrix& g, const CMatrix& /*X*/, const CMatrix& /*W*/, double /*s*/,
                              const CMatrix& /*gKfu*/) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  g.zeros();
}
void CWhiteKern::wsGradW(CMatrix& gW, const CMatrix& /*X*/, const CMatrix& W, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  gW.resize(W.getRows(), W.getCols());
  gW.zeros();
}
void CWhiteKern::wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& /*W*/, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  gX.resize(X.getRows(), X.getCols());
  gX.zeros();
}
void CWhiteKern::wsGradS(CMatrix& gS, const CMatrix& /*X*/, const CMatrix& /*W*/, double /*s*/,
                         const CMatrix& /*gKfu*/) const
{
  gS.resize(1, 1);
  gS.zeros();
}

void CBiasKern::wnCrossCompute(CMatrix& /*Kfu*/, const CMatrix& /*X*/, const CMatrix& /*Z*/, double /*s*/) const
{
  throw ndlexceptions::Error(
    "Bias kernel has no L2 square-root: white-noise inducing is not defined for bias. "
    "Remove CBiasKern from the compound, or use point inducing.");
}

// ---- Linear ----
double CLinKern::psi0(const CMatrix& mu, const CMatrix& S) const
{
  checkPsiInputs(mu, S, getInputDim());
  double s = 0.0;
  for(unsigned int n = 0; n < mu.getRows(); n++)
    for(unsigned int q = 0; q < getInputDim(); q++)
      s += variance * (mu.getVal(n, q) * mu.getVal(n, q) + S.getVal(n, q));
  return s;
}
void CLinKern::psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  Psi1.resize(mu.getRows(), Z.getRows());
  Psi1.gemm(mu, Z, variance, 0.0, "n", "t");
}
void CLinKern::psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = getInputDim();
  CMatrix Psi1(N, M);
  psi1(Psi1, mu, S, Z);
  // psi2 = Psi1^T Psi1 + Z (var^2 diag(sum_n S_nq)) Z^T  — GPy: tdot(psi1.T) + (S.sum(0)*Zv) @ Zv.T
  Psi2.resize(M, M);
  Psi2.gemm(Psi1, Psi1, 1.0, 0.0, "t", "n");
  vector<double> Ssum(Q, 0.0);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
      Ssum[q] += S.getVal(n, q);
  for(unsigned int m1 = 0; m1 < M; m1++)
  {
    for(unsigned int m2 = 0; m2 <= m1; m2++)
    {
      double extra = 0.0;
      for(unsigned int q = 0; q < Q; q++)
        extra += Ssum[q] * (variance * Z.getVal(m1, q)) * (variance * Z.getVal(m2, q));
      Psi2.addVal(extra, m1, m2);
      if(m1 != m2)
        Psi2.addVal(extra, m2, m1);
    }
  }
  Psi2.setSymmetric(true);
}
void CLinKern::getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                          const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                          const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                          bool add) const
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = getInputDim();
  checkPsiInputs(mu, S, Q);
  checkPsiZ(Z, Q);
  zeroGradOutputs(gParams, gZ, gMu, gS, getNumParams(), N, M, Q, add);

  // Symmetrise dL_dpsi2
  CMatrix dL2(M, M);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < M; j++)
      dL2.setVal(0.5 * (dL_dpsi2.getVal(i, j) + dL_dpsi2.getVal(j, i)), i, j);

  double gVar = 0.0;
  // psi0 + psi1 contributions (GPy linear_psi_comp)
  for(unsigned int n = 0; n < N; n++)
  {
    const double d0 = dL_dpsi0.getVal(n);
    for(unsigned int q = 0; q < Q; q++)
    {
      const double muq = mu.getVal(n, q);
      const double Snq = S.getVal(n, q);
      gVar += d0 * (muq * muq + Snq);
      gMu.addVal(2.0 * d0 * variance * muq, n, q);
      gS.addVal(d0 * variance, n, q);
    }
  }
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      const double d1 = dL_dpsi1.getVal(n, m);
      for(unsigned int q = 0; q < Q; q++)
      {
        gVar += d1 * mu.getVal(n, q) * Z.getVal(m, q);
        gMu.addVal(d1 * variance * Z.getVal(m, q), n, q);
        gZ.addVal(d1 * variance * mu.getVal(n, q), m, q);
      }
    }
  }

  // psi2: common_sum = mu @ (var Z)^T = Psi1
  CMatrix Psi1(N, M);
  psi1(Psi1, mu, S, Z);
  const double var2 = variance * variance;
  for(unsigned int q = 0; q < Q; q++)
  {
    double Z_expect = 0.0;
    for(unsigned int m1 = 0; m1 < M; m1++)
      for(unsigned int m2 = 0; m2 < M; m2++)
        Z_expect += dL2.getVal(m1, m2) * Z.getVal(m1, q) * Z.getVal(m2, q);

    double Ssum = 0.0;
    for(unsigned int n = 0; n < N; n++)
      Ssum += S.getVal(n, q);
    gVar += 2.0 * Ssum * variance * Z_expect;

    for(unsigned int n = 0; n < N; n++)
    {
      double common_expect_q = 0.0;
      for(unsigned int m1 = 0; m1 < M; m1++)
      {
        double row = 0.0;
        for(unsigned int m2 = 0; m2 < M; m2++)
          row += (dL2.getVal(m1, m2) + dL2.getVal(m2, m1)) * Z.getVal(m2, q);
        common_expect_q += Psi1.getVal(n, m1) * row;
      }
      // GPy: common_expect = common_sum @ (dL2T @ Z); dL_dmu = common_expect * var
      // Simpler path matching _psiDerivativecomputations_block for 2d dL_dpsi2:
      gMu.addVal(0.0, n, q); // filled below
    }
  }

  // Clearer psi2 grads following GPy 2d branch
  CMatrix dL2T(M, M);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < M; j++)
      dL2T.setVal(dL2.getVal(i, j) + dL2.getVal(j, i), i, j);

  CMatrix Zv(M, Q);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int q = 0; q < Q; q++)
      Zv.setVal(variance * Z.getVal(m, q), m, q);

  // common_sum = Psi1 (N x M)
  CMatrix common_expect(N, Q); // (common_sum @ (dL2T @ Z)) 
  CMatrix dL2TZ(M, Q);
  dL2TZ.gemm(dL2T, Z, 1.0, 0.0, "n", "n");
  common_expect.gemm(Psi1, dL2TZ, 1.0, 0.0, "n", "n");

  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
    {
      gVar += common_expect.getVal(n, q) * mu.getVal(n, q);
      gMu.addVal(common_expect.getVal(n, q) * variance, n, q);
      // dS from psi2: Z_expect * var^2 for each n
    }

  for(unsigned int q = 0; q < Q; q++)
  {
    double Z_expect = 0.0;
    for(unsigned int m1 = 0; m1 < M; m1++)
      for(unsigned int m2 = 0; m2 < M; m2++)
        Z_expect += dL2.getVal(m1, m2) * Z.getVal(m1, q) * Z.getVal(m2, q);
    for(unsigned int n = 0; n < N; n++)
      gS.addVal(Z_expect * var2, n, q);
  }

  // dZ from psi2: var2 * S.sum * (dL2T @ Z) + (Z2_expect)^T @ (var * mu)
  // Z2_expect = common_sum.T @ dL2T  → (M x N)(N?) GPy: Z2_expect = inner(common_sum, dL2T) → N x M
  CMatrix Z2_expect(N, M);
  Z2_expect.gemm(Psi1, dL2T, 1.0, 0.0, "n", "n");
  CMatrix Z1_expect(M, Q);
  Z1_expect.gemm(dL2T, Z, 1.0, 0.0, "n", "n");

  vector<double> Ssum(Q, 0.0);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
      Ssum[q] += S.getVal(n, q);

  for(unsigned int m = 0; m < M; m++)
    for(unsigned int q = 0; q < Q; q++)
    {
      double g = var2 * Ssum[q] * Z1_expect.getVal(m, q);
      for(unsigned int n = 0; n < N; n++)
        g += Z2_expect.getVal(n, m) * variance * mu.getVal(n, q);
      gZ.addVal(g, m, q);
    }

  // Remove double-counting of Z_expect in gVar from the incomplete first psi2 loop —
  // the first loop already added 2*Ssum*var*Z_expect; second adds common_expect*mu. Good.
  // But first incomplete loop also had empty gMu.addVal(0) — harmless.

  gParams.addVal(gVar, 0);
}

// ---- RBF isotropic ----
static vector<double> isoLengthscale2(double inverseWidth, unsigned int Q)
{
  vector<double> l2(Q, 1.0 / inverseWidth);
  return l2;
}

double CRbfKern::psi0(const CMatrix& mu, const CMatrix& S) const
{
  checkPsiInputs(mu, S, getInputDim());
  return variance * static_cast<double>(mu.getRows());
}
void CRbfKern::psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  rbfPsi1(Psi1, variance, isoLengthscale2(inverseWidth, getInputDim()), mu, S, Z);
}
void CRbfKern::psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  rbfPsi2(Psi2, variance, isoLengthscale2(inverseWidth, getInputDim()), mu, S, Z);
}
void CRbfKern::getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                          const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                          const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                          bool add) const
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = getInputDim();
  checkPsiInputs(mu, S, Q);
  checkPsiZ(Z, Q);
  zeroGradOutputs(gParams, gZ, gMu, gS, getNumParams(), N, M, Q, add);

  vector<double> l2 = isoLengthscale2(inverseWidth, Q);
  vector<double> gL(Q, 0.0);
  double gVar = 0.0;
  CMatrix gZloc(M, Q);
  CMatrix gMuloc(N, Q);
  CMatrix gSloc(N, Q);
  rbfGradPsi(gL, gZloc, gMuloc, gSloc, gVar, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, l2, mu, S, Z);

  // Map dL/dℓ → dL/dγ with ℓ = γ^{-1/2}, ∂ℓ/∂γ = -0.5 ℓ / γ
  double gGamma = 0.0;
  const double ell = 1.0 / sqrt(inverseWidth);
  for(unsigned int q = 0; q < Q; q++)
    gGamma += gL[q] * (-0.5 * ell / inverseWidth);

  gParams.addVal(gGamma, 0);
  gParams.addVal(gVar, 1);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int q = 0; q < Q; q++)
      gZ.addVal(gZloc.getVal(m, q), m, q);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
    {
      gMu.addVal(gMuloc.getVal(n, q), n, q);
      gS.addVal(gSloc.getVal(n, q), n, q);
    }
}

// CIP-001C: isotropic RBF × Gaussian white-noise windows.
void CRbfKern::wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(X.getCols() == d && Z.getCols() == d);
  BOUNDCHECK(s > 0.0);
  Kfu.resize(N, M);
  const double ell = 1.0 / sqrt(inverseWidth);
  const double sigma = sqrt(variance);
  const double v = ell * ell + 2.0 * s * s;
  const double amp = sigma * pow(2.0 * sqrt(2.0) * ell * s / v, 0.5 * (double)d);
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      Kfu.setVal(amp * exp(-r2 / v), n, m);
    }
  }
}

void CRbfKern::wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                            const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(g.getCols() == nParams);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  const double ell = 1.0 / sqrt(inverseWidth);
  const double v = ell * ell + 2.0 * s * s;
  const double sigma = sqrt(variance);
  const double amp = sigma * pow(2.0 * sqrt(2.0) * ell * s / v, 0.5 * (double)d);
  const double dlogK_dell_base = 0.5 * (double)d * (1.0 / ell - 2.0 * ell / v);
  double gEll = 0.0;
  double gVar = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      const double k = amp * exp(-r2 / v);
      const double cg = gKfu.getVal(n, m);
      const double dlogK_dell = dlogK_dell_base + 2.0 * ell * r2 / (v * v);
      gEll += cg * k * dlogK_dell;
      // ∂K/∂var = K / (2 var)
      gVar += cg * k / (2.0 * variance);
    }
  }
  // ∂ℓ/∂γ = -0.5 ℓ / γ
  const double gGamma = gEll * (-0.5 * ell / inverseWidth);
  g.setVal(gGamma, 0);
  g.setVal(gVar, 1);
}

void CRbfKern::wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                       const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gZ.resize(M, d);
  gZ.zeros();
  const double ell = 1.0 / sqrt(inverseWidth);
  const double v = ell * ell + 2.0 * s * s;
  const double sigma = sqrt(variance);
  const double amp = sigma * pow(2.0 * sqrt(2.0) * ell * s / v, 0.5 * (double)d);
  const double twoOverV = 2.0 / v;
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      const double k = amp * exp(-r2 / v);
      const double cg = gKfu.getVal(n, m) * k * twoOverV;
      for(unsigned int q = 0; q < d; q++)
        gZ.addVal(cg * (X.getVal(n, q) - Z.getVal(m, q)), m, q);
    }
  }
}

void CRbfKern::wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                       const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gX.resize(N, d);
  gX.zeros();
  const double ell = 1.0 / sqrt(inverseWidth);
  const double v = ell * ell + 2.0 * s * s;
  const double sigma = sqrt(variance);
  const double amp = sigma * pow(2.0 * sqrt(2.0) * ell * s / v, 0.5 * (double)d);
  const double twoOverV = 2.0 / v;
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      const double k = amp * exp(-r2 / v);
      // ∇_x log Kfu = -∇_z log Kfu = -2(x-z)/v
      const double cg = gKfu.getVal(n, m) * k * (-twoOverV);
      for(unsigned int q = 0; q < d; q++)
        gX.addVal(cg * (X.getVal(n, q) - Z.getVal(m, q)), n, q);
    }
  }
}

void CRbfKern::wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                       const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gS.resize(1, 1);
  const double ell = 1.0 / sqrt(inverseWidth);
  const double v = ell * ell + 2.0 * s * s;
  const double sigma = sqrt(variance);
  const double amp = sigma * pow(2.0 * sqrt(2.0) * ell * s / v, 0.5 * (double)d);
  const double dlogK_ds_base = 0.5 * (double)d * (1.0 / s - 4.0 * s / v);
  double gs = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double r2 = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        r2 += dz * dz;
      }
      const double k = amp * exp(-r2 / v);
      const double dlogK_ds = dlogK_ds_base + 4.0 * s * r2 / (v * v);
      gs += gKfu.getVal(n, m) * k * dlogK_ds;
    }
  }
  gS.setVal(gs, 0);
}

// ---- RBF ARD ----
static vector<double> ardLengthscale2(double inverseWidth, const CMatrix& scales)
{
  vector<double> l2(scales.getCols());
  for(unsigned int q = 0; q < scales.getCols(); q++)
  {
    const double s = scales.getVal(q);
    BOUNDCHECK(s > 0.0);
    l2[q] = 1.0 / (inverseWidth * s);
  }
  return l2;
}

double CRbfardKern::psi0(const CMatrix& mu, const CMatrix& S) const
{
  checkPsiInputs(mu, S, getInputDim());
  return variance * static_cast<double>(mu.getRows());
}
void CRbfardKern::psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  rbfPsi1(Psi1, variance, ardLengthscale2(inverseWidth, scales), mu, S, Z);
}
void CRbfardKern::psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  checkPsiInputs(mu, S, getInputDim());
  checkPsiZ(Z, getInputDim());
  rbfPsi2(Psi2, variance, ardLengthscale2(inverseWidth, scales), mu, S, Z);
}
void CRbfardKern::getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                             const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                             const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                             bool add) const
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = getInputDim();
  checkPsiInputs(mu, S, Q);
  checkPsiZ(Z, Q);
  zeroGradOutputs(gParams, gZ, gMu, gS, getNumParams(), N, M, Q, add);

  vector<double> l2 = ardLengthscale2(inverseWidth, scales);
  vector<double> gL(Q, 0.0);
  double gVar = 0.0;
  CMatrix gZloc(M, Q);
  CMatrix gMuloc(N, Q);
  CMatrix gSloc(N, Q);
  rbfGradPsi(gL, gZloc, gMuloc, gSloc, gVar, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, l2, mu, S, Z);

  // ℓ_q = (γ s_q)^{-1/2}; ∂ℓ/∂γ = -0.5 ℓ/γ; ∂ℓ/∂s = -0.5 ℓ/s
  double gGamma = 0.0;
  for(unsigned int q = 0; q < Q; q++)
  {
    const double s = scales.getVal(q);
    const double ell = sqrt(l2[q]);
    gGamma += gL[q] * (-0.5 * ell / inverseWidth);
    gParams.addVal(gL[q] * (-0.5 * ell / s), q + 2);
  }
  gParams.addVal(gGamma, 0);
  gParams.addVal(gVar, 1);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int q = 0; q < Q; q++)
      gZ.addVal(gZloc.getVal(m, q), m, q);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
    {
      gMu.addVal(gMuloc.getVal(n, q), n, q);
      gS.addVal(gSloc.getVal(n, q), n, q);
    }
}

// CIP-001C: RBF-ARD × shared-s Gaussian windows (product over dims).
void CRbfardKern::wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(X.getCols() == d && Z.getCols() == d);
  BOUNDCHECK(s > 0.0);
  Kfu.resize(N, M);
  const double sigma = sqrt(variance);
  vector<double> ell(d);
  vector<double> v(d);
  double amp = sigma;
  for(unsigned int q = 0; q < d; q++)
  {
    const double aq = scales.getVal(q);
    BOUNDCHECK(aq > 0.0);
    ell[q] = 1.0 / sqrt(inverseWidth * aq);
    v[q] = ell[q] * ell[q] + 2.0 * s * s;
    amp *= sqrt(2.0 * sqrt(2.0) * ell[q] * s / v[q]);
  }
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double expo = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        expo += dz * dz / v[q];
      }
      Kfu.setVal(amp * exp(-expo), n, m);
    }
  }
}

void CRbfardKern::wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                               const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(g.getCols() == nParams);
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  const double sigma = sqrt(variance);
  vector<double> ell(d);
  vector<double> v(d);
  double amp = sigma;
  for(unsigned int q = 0; q < d; q++)
  {
    const double aq = scales.getVal(q);
    ell[q] = 1.0 / sqrt(inverseWidth * aq);
    v[q] = ell[q] * ell[q] + 2.0 * s * s;
    amp *= sqrt(2.0 * sqrt(2.0) * ell[q] * s / v[q]);
  }
  double gGamma = 0.0;
  double gVar = 0.0;
  vector<double> gScale(d, 0.0);
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double expo = 0.0;
      vector<double> dz2(d);
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        dz2[q] = dz * dz;
        expo += dz2[q] / v[q];
      }
      const double k = amp * exp(-expo);
      const double cg = gKfu.getVal(n, m);
      gVar += cg * k / (2.0 * variance);
      for(unsigned int q = 0; q < d; q++)
      {
        // ∂ log K / ∂ℓ_q = 1/2 (1/ℓ_q - 2ℓ_q/v_q) + 2ℓ_q dz2_q / v_q^2
        const double dlog_dell = 0.5 * (1.0 / ell[q] - 2.0 * ell[q] / v[q])
          + 2.0 * ell[q] * dz2[q] / (v[q] * v[q]);
        const double dK_dell = cg * k * dlog_dell;
        // ℓ_q = (γ a_q)^{-1/2}; ∂ℓ/∂γ = -0.5 ℓ/γ; ∂ℓ/∂a = -0.5 ℓ/a
        gGamma += dK_dell * (-0.5 * ell[q] / inverseWidth);
        gScale[q] += dK_dell * (-0.5 * ell[q] / scales.getVal(q));
      }
    }
  }
  g.setVal(gGamma, 0);
  g.setVal(gVar, 1);
  for(unsigned int q = 0; q < d; q++)
    g.setVal(gScale[q], q + 2);
}

void CRbfardKern::wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                          const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gZ.resize(M, d);
  gZ.zeros();
  const double sigma = sqrt(variance);
  vector<double> ell(d);
  vector<double> v(d);
  double amp = sigma;
  for(unsigned int q = 0; q < d; q++)
  {
    ell[q] = 1.0 / sqrt(inverseWidth * scales.getVal(q));
    v[q] = ell[q] * ell[q] + 2.0 * s * s;
    amp *= sqrt(2.0 * sqrt(2.0) * ell[q] * s / v[q]);
  }
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double expo = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        expo += dz * dz / v[q];
      }
      const double k = amp * exp(-expo);
      const double cg = gKfu.getVal(n, m) * k;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        gZ.addVal(cg * 2.0 * dz / v[q], m, q);
      }
    }
  }
}

void CRbfardKern::wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                          const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gX.resize(N, d);
  gX.zeros();
  const double sigma = sqrt(variance);
  vector<double> ell(d);
  vector<double> v(d);
  double amp = sigma;
  for(unsigned int q = 0; q < d; q++)
  {
    ell[q] = 1.0 / sqrt(inverseWidth * scales.getVal(q));
    v[q] = ell[q] * ell[q] + 2.0 * s * s;
    amp *= sqrt(2.0 * sqrt(2.0) * ell[q] * s / v[q]);
  }
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double expo = 0.0;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        expo += dz * dz / v[q];
      }
      const double k = amp * exp(-expo);
      const double cg = gKfu.getVal(n, m) * k;
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        gX.addVal(cg * (-2.0 * dz / v[q]), n, q);
      }
    }
  }
}

void CRbfardKern::wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                          const CMatrix& gKfu) const
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int d = getInputDim();
  DIMENSIONMATCH(gKfu.getRows() == N && gKfu.getCols() == M);
  BOUNDCHECK(s > 0.0);
  gS.resize(1, 1);
  const double sigma = sqrt(variance);
  vector<double> ell(d);
  vector<double> v(d);
  double amp = sigma;
  for(unsigned int q = 0; q < d; q++)
  {
    ell[q] = 1.0 / sqrt(inverseWidth * scales.getVal(q));
    v[q] = ell[q] * ell[q] + 2.0 * s * s;
    amp *= sqrt(2.0 * sqrt(2.0) * ell[q] * s / v[q]);
  }
  double gs = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int m = 0; m < M; m++)
    {
      double expo = 0.0;
      vector<double> dz2(d);
      for(unsigned int q = 0; q < d; q++)
      {
        const double dz = X.getVal(n, q) - Z.getVal(m, q);
        dz2[q] = dz * dz;
        expo += dz2[q] / v[q];
      }
      const double k = amp * exp(-expo);
      double dlog = 0.0;
      for(unsigned int q = 0; q < d; q++)
        dlog += 0.5 * (1.0 / s - 4.0 * s / v[q]) + 4.0 * s * dz2[q] / (v[q] * v[q]);
      gs += gKfu.getVal(n, m) * k * dlog;
    }
  }
  gS.setVal(gs, 0);
}

// ---- Compound ----
double CCmpndKern::psi0(const CMatrix& mu, const CMatrix& S) const
{
  double s = 0.0;
  for(size_t i = 0; i < components.size(); i++)
    s += components[i]->psi0(mu, S);
  return s;
}
void CCmpndKern::psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  DIMENSIONMATCH(components.size() > 0);
  components[0]->psi1(Psi1, mu, S, Z);
  CMatrix tmp;
  for(size_t i = 1; i < components.size(); i++)
  {
    components[i]->psi1(tmp, mu, S, Z);
    Psi1.axpy(tmp, 1.0);
  }
}
void CCmpndKern::psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const
{
  DIMENSIONMATCH(components.size() > 0);
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  components[0]->psi2(Psi2, mu, S, Z);
  CMatrix tmp;
  for(size_t i = 1; i < components.size(); i++)
  {
    components[i]->psi2(tmp, mu, S, Z);
    Psi2.axpy(tmp, 1.0);
  }
  // Cross terms (GPy Add.psi2): skip White; Bias x other → var * (psi1_sum[:,None]+psi1_sum[None,:]);
  // else einsum psi1_i, psi1_j
  for(size_t i = 0; i < components.size(); i++)
  {
    for(size_t j = i + 1; j < components.size(); j++)
    {
      CKern* p1 = components[i];
      CKern* p2 = components[j];
      if(p1->getType() == "white" || p2->getType() == "white")
        continue;
      if(p1->getType() == "bias" && (p2->getType() == "rbf" || p2->getType() == "rbfard" || p2->getType() == "lin"))
      {
        CMatrix P1;
        p2->psi1(P1, mu, S, Z);
        for(unsigned int m1 = 0; m1 < M; m1++)
        {
          double s1 = 0.0;
          for(unsigned int n = 0; n < N; n++)
            s1 += P1.getVal(n, m1);
          for(unsigned int m2 = 0; m2 < M; m2++)
          {
            double s2 = 0.0;
            for(unsigned int n = 0; n < N; n++)
              s2 += P1.getVal(n, m2);
            const double addv = p1->getVariance() * (s1 + s2);
            Psi2.addVal(addv, m1, m2);
          }
        }
      }
      else if(p2->getType() == "bias" && (p1->getType() == "rbf" || p1->getType() == "rbfard" || p1->getType() == "lin"))
      {
        CMatrix P1;
        p1->psi1(P1, mu, S, Z);
        for(unsigned int m1 = 0; m1 < M; m1++)
        {
          double s1 = 0.0;
          for(unsigned int n = 0; n < N; n++)
            s1 += P1.getVal(n, m1);
          for(unsigned int m2 = 0; m2 < M; m2++)
          {
            double s2 = 0.0;
            for(unsigned int n = 0; n < N; n++)
              s2 += P1.getVal(n, m2);
            const double addv = p2->getVariance() * (s1 + s2);
            Psi2.addVal(addv, m1, m2);
          }
        }
      }
      else if(p1->getType() != "bias" && p2->getType() != "bias")
      {
        CMatrix A, B;
        p1->psi1(A, mu, S, Z);
        p2->psi1(B, mu, S, Z);
        // sum_n (A[n,:]^T B[n,:] + B[n,:]^T A[n,:])
        for(unsigned int n = 0; n < N; n++)
        {
          for(unsigned int m1 = 0; m1 < M; m1++)
          {
            for(unsigned int m2 = 0; m2 < M; m2++)
            {
              const double v = A.getVal(n, m1) * B.getVal(n, m2) + B.getVal(n, m1) * A.getVal(n, m2);
              Psi2.addVal(v, m1, m2);
            }
          }
        }
      }
      else
      {
        throw ndlexceptions::NotImplementedError("psi2 cross term not implemented for these compound parts");
      }
    }
  }
  Psi2.setSymmetric(true);
}

void CCmpndKern::getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                            const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                            const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                            bool add) const
{
  const unsigned int N = mu.getRows();
  const unsigned int M = Z.getRows();
  const unsigned int Q = getInputDim();
  zeroGradOutputs(gParams, gZ, gMu, gS, getNumParams(), N, M, Q, add);

  // tmp = dL_dpsi2.sum(0) + dL_dpsi2.sum(1)  (M,)
  vector<double> tmp(M, 0.0);
  for(unsigned int m1 = 0; m1 < M; m1++)
    for(unsigned int m2 = 0; m2 < M; m2++)
    {
      tmp[m1] += dL_dpsi2.getVal(m1, m2);
      tmp[m2] += dL_dpsi2.getVal(m1, m2);
    }

  unsigned int paramOffset = 0;
  for(size_t i = 0; i < components.size(); i++)
  {
    CKern* p1 = components[i];
    CMatrix eff_dL_dpsi1 = dL_dpsi1;
    // copy
    eff_dL_dpsi1.deepCopy(dL_dpsi1);
    for(size_t j = 0; j < components.size(); j++)
    {
      if(j == i)
        continue;
      CKern* p2 = components[j];
      if(p2->getType() == "white")
        continue;
      if(p2->getType() == "bias")
      {
        const double v = p2->getVariance();
        for(unsigned int n = 0; n < N; n++)
          for(unsigned int m = 0; m < M; m++)
            eff_dL_dpsi1.addVal(tmp[m] * v, n, m);
      }
      else
      {
        CMatrix P2;
        p2->psi1(P2, mu, S, Z);
        for(unsigned int n = 0; n < N; n++)
          for(unsigned int m = 0; m < M; m++)
            eff_dL_dpsi1.addVal(tmp[m] * P2.getVal(n, m), n, m);
      }
    }

    CMatrix gP(1, p1->getNumParams());
    CMatrix gZc(M, Q);
    CMatrix gMuc(N, Q);
    CMatrix gSc(N, Q);
    p1->getGradPsi(gP, gZc, gMuc, gSc, dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, mu, S, Z, false);
    for(unsigned int p = 0; p < p1->getNumParams(); p++)
      gParams.addVal(gP.getVal(p), paramOffset + p);
    for(unsigned int m = 0; m < M; m++)
      for(unsigned int q = 0; q < Q; q++)
        gZ.addVal(gZc.getVal(m, q), m, q);
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int q = 0; q < Q; q++)
      {
        gMu.addVal(gMuc.getVal(n, q), n, q);
        gS.addVal(gSc.getVal(n, q), n, q);
      }
    paramOffset += p1->getNumParams();
  }
}

void CCmpndKern::wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const
{
  DIMENSIONMATCH(components.size() > 0);
  // Shared white-noise drive: Kfu = sum_c g_c * psi; Kuu unchanged (inducing-side).
  components[0]->wnCrossCompute(Kfu, X, Z, s);
  CMatrix tmp;
  for(size_t i = 1; i < components.size(); i++)
  {
    components[i]->wnCrossCompute(tmp, X, Z, s);
    Kfu.axpy(tmp, 1.0);
  }
}

void CCmpndKern::wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                              const CMatrix& gKfu) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  g.zeros();
  unsigned int start = 0;
  for(size_t i = 0; i < components.size(); i++)
  {
    CMatrix subg(1, components[i]->getNumParams());
    components[i]->wnGradParams(subg, X, Z, s, gKfu);
    g.setMatrix(0, start, subg);
    start += components[i]->getNumParams();
  }
}

void CCmpndKern::wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                         const CMatrix& gKfu) const
{
  gZ.resize(Z.getRows(), Z.getCols());
  gZ.zeros();
  CMatrix tmp;
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->wnGradZ(tmp, X, Z, s, gKfu);
    gZ.axpy(tmp, 1.0);
  }
}

void CCmpndKern::wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                         const CMatrix& gKfu) const
{
  gX.resize(X.getRows(), X.getCols());
  gX.zeros();
  CMatrix tmp;
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->wnGradX(tmp, X, Z, s, gKfu);
    gX.axpy(tmp, 1.0);
  }
}

void CCmpndKern::wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                         const CMatrix& gKfu) const
{
  gS.resize(1, 1);
  gS.zeros();
  CMatrix tmp(1, 1);
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->wnGradS(tmp, X, Z, s, gKfu);
    gS.axpy(tmp, 1.0);
  }
}

void CCmpndKern::greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
                                    double s) const
{
  DIMENSIONMATCH(components.size() > 0);
  components[0]->greensCrossCompute(Kfu, X, Z, s);
  CMatrix tmp;
  for(size_t i = 1; i < components.size(); i++)
  {
    components[i]->greensCrossCompute(tmp, X, Z, s);
    Kfu.axpy(tmp, 1.0);
  }
}

void CCmpndKern::greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                                  const CMatrix& gKfu) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  g.zeros();
  unsigned int start = 0;
  for(size_t i = 0; i < components.size(); i++)
  {
    CMatrix subg(1, components[i]->getNumParams());
    components[i]->greensGradParams(subg, X, Z, s, gKfu);
    g.setMatrix(0, start, subg);
    start += components[i]->getNumParams();
  }
}

void CCmpndKern::greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                             const CMatrix& gKfu) const
{
  gZ.resize(Z.getRows(), Z.getCols());
  gZ.zeros();
  CMatrix tmp;
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->greensGradZ(tmp, X, Z, s, gKfu);
    gZ.axpy(tmp, 1.0);
  }
}

void CCmpndKern::greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                             const CMatrix& gKfu) const
{
  gX.resize(X.getRows(), X.getCols());
  gX.zeros();
  CMatrix tmp;
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->greensGradX(tmp, X, Z, s, gKfu);
    gX.axpy(tmp, 1.0);
  }
}

void CCmpndKern::greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                             const CMatrix& gKfu) const
{
  gS.resize(1, 1);
  gS.zeros();
  CMatrix tmp(1, 1);
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->greensGradS(tmp, X, Z, s, gKfu);
    gS.axpy(tmp, 1.0);
  }
}

void CCmpndKern::wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const
{
  DIMENSIONMATCH(components.size() > 0);
  components[0]->wsCrossCompute(Kfu, X, W, s);
  CMatrix tmp;
  for(size_t i = 1; i < components.size(); i++)
  {
    components[i]->wsCrossCompute(tmp, X, W, s);
    Kfu.axpy(tmp, 1.0);
  }
}

void CCmpndKern::wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                              const CMatrix& gKfu) const
{
  DIMENSIONMATCH(g.getCols() == getNumParams());
  g.zeros();
  unsigned int start = 0;
  for(size_t i = 0; i < components.size(); i++)
  {
    CMatrix subg(1, components[i]->getNumParams());
    components[i]->wsGradParams(subg, X, W, s, gKfu);
    g.setMatrix(0, start, subg);
    start += components[i]->getNumParams();
  }
}

void CCmpndKern::wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
                         const CMatrix& gKfu) const
{
  gW.resize(W.getRows(), W.getCols());
  gW.zeros();
  CMatrix tmp;
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->wsGradW(tmp, X, W, s, gKfu);
    gW.axpy(tmp, 1.0);
  }
}

void CCmpndKern::wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
                         const CMatrix& gKfu) const
{
  gX.resize(X.getRows(), X.getCols());
  gX.zeros();
  CMatrix tmp;
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->wsGradX(tmp, X, W, s, gKfu);
    gX.axpy(tmp, 1.0);
  }
}

void CCmpndKern::wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
                         const CMatrix& gKfu) const
{
  gS.resize(1, 1);
  gS.zeros();
  CMatrix tmp(1, 1);
  for(size_t i = 0; i < components.size(); i++)
  {
    components[i]->wsGradS(tmp, X, W, s, gKfu);
    gS.axpy(tmp, 1.0);
  }
}
