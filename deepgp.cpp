#include "deepgp.h"
#include "CKern.h"
#include "ndlstrutil.h"
#include "ndlutil.h"
#include <cstdlib>
#include <memory>
#include <sstream>

using namespace std;

int main(int argc, char* argv[])
{
  CCldeepgp command(argc, argv);
  command.setFlags(true);
  command.setVerbosity(2);
  command.setMode("deepgp");
  try
  {
    while(command.isFlags())
    {
      string argument = command.getCurrentArgument();
      if(argv[command.getCurrentArgumentNo()][0] == '-')
      {
        if(command.isCurrentArg("-?", "--?") || command.isCurrentArg("-h", "--help"))
        {
          command.helpInfo();
          command.exitNormal();
        }
        else if(command.isCurrentArg("-v", "--verbosity"))
        {
          command.incrementArgument();
          command.setVerbosity(command.getIntFromCurrentArgument());
        }
        else if(command.isCurrentArg("-s", "--seed"))
        {
          command.incrementArgument();
          command.setSeed(command.getIntFromCurrentArgument());
        }
        else
          command.unrecognisedFlag();
      }
      else if(argument == "learn")
        command.learn();
      else if(argument == "display")
        command.display();
      else
        command.exitError("Invalid deepgp command provided.");
      command.incrementArgument();
    }
    command.exitError("No deepgp command provided.");
  }
  catch(ndlexceptions::FileFormatError err)
  {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::FileReadError err)
  {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::FileWriteError err)
  {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::FileError err)
  {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::Error err)
  {
    command.exitError(err.getMessage());
  }
  catch(std::bad_alloc)
  {
    command.exitError("Out of memory.");
  }
  catch(std::exception)
  {
    command.exitError("Unhandled exception.");
  }
}

CCldeepgp::CCldeepgp(int arc, char** arv) : CClctrl(arc, arv)
{
}

namespace {

void centreColumns(CMatrix& Y)
{
  CMatrix mean = meanCol(Y);
  for(unsigned int i = 0; i < Y.getRows(); i++)
    for(unsigned int j = 0; j < Y.getCols(); j++)
      Y.setVal(Y.getVal(i, j) - mean.getVal(j), i, j);
}

CKern* makeLayerKern(const string& type, unsigned int din,
                     double rbfInvWidth, double variance)
{
  CKern* k = 0;
  if(type == "rbf")
  {
    CRbfKern* rbf = new CRbfKern(din);
    rbf->setInitParam();
    if(rbfInvWidth > 0.0)
      rbf->setInverseWidth(rbfInvWidth);
    if(variance > 0.0)
      rbf->setVariance(variance);
    k = rbf;
  }
  else if(type == "mlp")
  {
    CMlpKern* mlp = new CMlpKern(din);
    mlp->setInitParam();
    if(variance > 0.0 && mlp->getNumParams() >= 3)
      mlp->setParam(variance, 2);
    k = mlp;
  }
  else if(type == "lin" || type == "linear")
  {
    CLinKern* lin = new CLinKern(din);
    lin->setInitParam();
    if(variance > 0.0)
      lin->setVariance(variance);
    k = lin;
  }
  else
    throw ndlexceptions::Error(
      "Unknown kernel for deepgp: " + type + ". Use rbf, mlp, lin.");
  return k;
}

vector<unsigned int> parseUintList(const string& s, unsigned int fallback,
                                   unsigned int nExpect)
{
  vector<unsigned int> out;
  if(s.find(',') == string::npos)
  {
    unsigned int v = fallback;
    if(s.size() > 0)
      v = (unsigned int)atoi(s.c_str());
    out.assign(nExpect, v);
    return out;
  }
  size_t start = 0;
  while(start < s.size())
  {
    size_t comma = s.find(',', start);
    string tok = (comma == string::npos) ? s.substr(start)
                                         : s.substr(start, comma - start);
    out.push_back((unsigned int)atoi(tok.c_str()));
    if(comma == string::npos)
      break;
    start = comma + 1;
  }
  if(out.size() != nExpect)
    throw ndlexceptions::Error(
      "Expected " + ndlstrutil::itoa((int)nExpect)
      + " comma-separated values, got " + ndlstrutil::itoa((int)out.size()));
  return out;
}

string peekModelBaseType(const string& modelFileName)
{
  ifstream in(modelFileName.c_str());
  if(!in.is_open())
    throw ndlexceptions::FileReadError(modelFileName);
  try
  {
    CStreamInterface::readVersionFromStream(in);
    return CStreamInterface::getBaseTypeStream(in);
  }
  catch(ndlexceptions::StreamFormatError& err)
  {
    throw ndlexceptions::FileFormatError(modelFileName, err);
  }
}

} // namespace

void CCldeepgp::learn()
{
  incrementArgument();
  setMode("learn");

  int numLayers = 2;
  int hiddenDim = -1;
  string mList = "";
  int numInducing = 10;
  string inducingType = "points";
  bool centreData = true;
  int iters = 100;
  int batchSize = -1;
  int numSamples = 1;
  double natGradGamma = 0.0;
  double adamLr = 1e-2;
  double chi = 1.0;
  double beta = 1.0;
  bool dropResidual = false;
  string kernelType = "rbf";
  string modelFileName = "deepgp_model";

  while(isFlags())
  {
    if(isCurrentArgumentFlag())
    {
      if(isCurrentArg("-?", "--?") || isCurrentArg("-h", "--help"))
      {
        helpInfo();
        exitNormal();
      }
      else if(isCurrentArg("-L", "--num-layers"))
      {
        incrementArgument();
        numLayers = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("--hidden", "--hidden"))
      {
        incrementArgument();
        hiddenDim = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-M", "--num-inducing"))
      {
        incrementArgument();
        mList = getCurrentArgument();
        if(mList.find(',') == string::npos)
          numInducing = atoi(mList.c_str());
      }
      else if(isCurrentArg("--inducing", "--inducing"))
      {
        incrementArgument();
        inducingType = getCurrentArgument();
      }
      else if(isCurrentArg("--residual", "--residual"))
      {
        incrementArgument();
        string v = getCurrentArgument();
        if(v == "off" || v == "0" || v == "false")
          dropResidual = true;
        else if(v == "on" || v == "1" || v == "true")
          dropResidual = false;
        else
          exitError("--residual must be on or off.");
      }
      else if(isCurrentArg("--batch", "--batch"))
      {
        incrementArgument();
        batchSize = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("--samples", "--samples"))
      {
        incrementArgument();
        numSamples = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("--chi", "--chi"))
      {
        incrementArgument();
        chi = getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("--beta", "--beta"))
      {
        incrementArgument();
        beta = getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("--natgrad-gamma", "--natgrad-gamma"))
      {
        incrementArgument();
        natGradGamma = getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("--adam-lr", "--adam-lr"))
      {
        incrementArgument();
        adamLr = getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("-k", "--kernel"))
      {
        incrementArgument();
        kernelType = getCurrentArgument();
      }
      else if(isCurrentArg("-C", "--Centre-data"))
      {
        incrementArgument();
        centreData = getBoolFromCurrentArgument();
      }
      else if(isCurrentArg("-#", "--#iterations"))
      {
        incrementArgument();
        iters = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-f", "--file-format"))
      {
        incrementArgument();
        setFileFormat(getIntFromCurrentArgument());
      }
      else
        unrecognisedFlag();
      incrementArgument();
    }
    else
      setFlags(false);
  }

  if(getCurrentArgumentNo() >= argc)
    exitError("There are not enough input parameters.");
  string trainDataFileName = getCurrentArgument();
  if((getCurrentArgumentNo() + 1) < argc)
    modelFileName = argv[getCurrentArgumentNo() + 1];

  if(numLayers < 1)
    exitError("-L must be >= 1.");
  if(inducingType != "points" && inducingType != "weightspace" && inducingType != "hermite")
    exitError("--inducing must be points, weightspace or hermite.");
  if(numSamples < 1)
    exitError("--samples must be positive.");
  if(!(chi > 0.0) || !(beta > 0.0))
    exitError("--chi and --beta must be positive.");
  if(getFileFormat() != 0)
    exitError("deepgp currently writes only the native text .model format (-f 0).");

  CMatrix X;
  CMatrix y;
  readData(X, y, trainDataFileName);
  if(centreData)
    centreColumns(y);

  const unsigned int Din = X.getCols();
  const unsigned int Dout = y.getCols();
  if(hiddenDim < 0)
    hiddenDim = (int)Din;

  vector<unsigned int> dims((unsigned int)numLayers + 1);
  dims[0] = Din;
  dims[(unsigned int)numLayers] = Dout;
  for(unsigned int ell = 1; ell < (unsigned int)numLayers; ell++)
    dims[ell] = (unsigned int)hiddenDim;

  vector<unsigned int> Mvec;
  try
  {
    if(mList.find(',') != string::npos)
      Mvec = parseUintList(mList, (unsigned int)numInducing,
                           (unsigned int)numLayers);
    else
      Mvec.assign((unsigned int)numLayers, (unsigned int)numInducing);
  }
  catch(ndlexceptions::Error& e)
  {
    exitError(e.getMessage());
  }
  for(unsigned int ell = 0; ell < Mvec.size(); ell++)
    if(Mvec[ell] < 1 || Mvec[ell] > X.getRows())
      exitError("-M values must be in 1..N.");

  if(batchSize < 0)
    batchSize = (int)X.getRows();
  if(batchSize < 1 || (unsigned int)batchSize > X.getRows())
    exitError("--batch must be in 1..N.");

  vector<CKern*> kerns;
  vector<unique_ptr<CKern> > owned;
  for(unsigned int ell = 0; ell < (unsigned int)numLayers; ell++)
  {
    CKern* k = makeLayerKern(kernelType, dims[ell], -1.0, -1.0);
    owned.emplace_back(k);
    kerns.push_back(k);
  }

  ndlutil::init_genrand((unsigned long)getSeed());
  CDeepGp model(X, y, kerns, dims, Mvec, beta, chi, getVerbosity());
  for(unsigned int ell = 0; ell < (unsigned int)numLayers; ell++)
  {
    if(inducingType != "points")
      model.setLayerInducingType(ell, inducingType);
  }
  if(dropResidual)
    model.setDropResidual(true);
  model.setNumSamples((unsigned int)numSamples);
  model.setNatGradGamma(natGradGamma);
  model.setSeed((uint64_t)getSeed());

  CAdam adam(adamLr);
  adam.setBatchSize((unsigned int)batchSize);
  adam.setMaxIters((unsigned int)iters);
  adam.setSeed((uint64_t)getSeed());

  if(getVerbosity() > 0)
  {
    cout << "Training deep GP: N=" << X.getRows()
         << " Din=" << Din << " Dout=" << Dout
         << " L=" << numLayers << " hidden=" << hiddenDim
         << " inducing=" << inducingType
         << " residual=" << (dropResidual ? "off" : "on")
         << " batch=" << batchSize << " samples=" << numSamples
         << " chi=" << chi << " iters=" << iters << endl;
  }

  model.installDefaultTextProgress(
      cout, getVerbosity(), "adam", [&model]() {
        ostringstream oss;
        model.display(oss);
        return oss.str();
      });
  adam.maximise(model);
  model.clearProgressObserver();

  string comment = "Run as:";
  for(int i = 0; i < argc; i++)
  {
    comment += " ";
    comment += argv[i];
  }
  comment += " with seed " + ndlstrutil::itoa(getSeed()) + ".";
  model.toFile(modelFileName, comment);

  if(getVerbosity() > 0)
    model.display(cout);
  exitNormal();
}

void CCldeepgp::display()
{
  incrementArgument();
  setMode("display");
  while(isFlags())
  {
    if(isCurrentArgumentFlag())
    {
      if(isCurrentArg("-?", "--?") || isCurrentArg("-h", "--help"))
      {
        helpInfo();
        exitNormal();
      }
      else
        unrecognisedFlag();
      incrementArgument();
    }
    else
      setFlags(false);
  }
  if(getCurrentArgumentNo() >= argc)
    exitError("There are not enough input parameters.");
  string modelFileName = getCurrentArgument();
  const string baseType = peekModelBaseType(modelFileName);
  if(baseType != "deepgp")
    exitError("Expected deepgp model file (baseType deepgp).");

  CMatrix Xtmp(1, 1), Ytmp(1, 1);
  Xtmp.zeros();
  Ytmp.zeros();
  CRbfKern ktmp(1);
  ktmp.setInitParam();
  vector<CKern*> kerns(1, &ktmp);
  CDeepGp model(Xtmp, Ytmp, kerns, vector<unsigned int>{1u, 1u},
                vector<unsigned int>{1u}, 1.0, 1.0, getVerbosity());
  model.fromFile(modelFileName);
  model.display(cout);
  exitNormal();
}

void CCldeepgp::helpHeader()
{
  helpInfo();
}

void CCldeepgp::helpInfo()
{
  string fname = "deepgp";
  cout << endl;
  cout << fname << ": Deep Gaussian process by doubly stochastic VI (CIP-0035)."
       << endl;
  cout << "Usage: " << fname
       << " [global] learn [options] datafile [model]" << endl;
  cout << "       " << fname << " display model" << endl;
  cout << endl;
  cout << "Global: -v/--verbosity, -s/--seed, -h/--help" << endl;
  cout << "learn:" << endl;
  cout << "  -L N            number of GP layers (default 2)" << endl;
  cout << "  --hidden D      hidden width for intermediate dims (default Din)"
       << endl;
  cout << "  -M M|M1,M2,...  inducing count per layer (default 10)" << endl;
  cout << "  --inducing T    points | weightspace | hermite (default points)" << endl;
  cout << "  --residual on|off  Nyström residual; off = finite-network bound"
       << endl;
  cout << "  --batch B       minibatch size (default N)" << endl;
  cout << "  --samples K     reparameterisation samples (default 1)" << endl;
  cout << "  --chi c         hidden variance scale (default 1)" << endl;
  cout << "  --beta b        observation precision init (default 1)" << endl;
  cout << "  --natgrad-gamma final-layer CVI step in [0,1] (default 0)" << endl;
  cout << "  --adam-lr a     Adam learning rate (default 1e-2)" << endl;
  cout << "  -k TYPE         rbf | mlp | lin (default rbf)" << endl;
  cout << "  -C bool         centre outputs (default true)" << endl;
  cout << "  -# iters        Adam iterations (default 100)" << endl;
  cout << endl;
  cout << "Stream type is deepgp. See docs/tutorials/deepgp.md." << endl;
}
