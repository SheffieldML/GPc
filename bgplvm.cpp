#include "bgplvm.h"

using namespace std;

int main(int argc, char* argv[])
{
  CClbgplvm command(argc, argv);
  command.setFlags(true);
  command.setVerbosity(2);
  command.setMode("bgplvm");
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
      else if(argument == "gnuplot")
        command.gnuplot();
      else
        command.exitError("Invalid bgplvm command provided.");
      command.incrementArgument();
    }
    command.exitError("No bgplvm command provided.");
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

CClbgplvm::CClbgplvm(int arc, char** arv) : CClctrl(arc, arv)
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

CKern* makeKernComponent(const string& type, unsigned int q,
                         double rbfInvWidth, double variance)
{
  CKern* k = 0;
  if(type == "rbf")
  {
    CRbfKern* rbf = new CRbfKern(q);
    rbf->setInitParam();
    if(rbfInvWidth > 0.0)
      rbf->setInverseWidth(rbfInvWidth);
    if(variance > 0.0)
      rbf->setVariance(variance);
    k = rbf;
  }
  else if(type == "lin" || type == "linear")
  {
    CLinKern* lin = new CLinKern(q);
    lin->setInitParam();
    if(variance > 0.0)
      lin->setVariance(variance);
    k = lin;
  }
  else if(type == "bias")
  {
    CBiasKern* bias = new CBiasKern(q);
    bias->setInitParam();
    if(variance > 0.0)
      bias->setVariance(variance);
    k = bias;
  }
  else if(type == "white")
  {
    CWhiteKern* white = new CWhiteKern(q);
    white->setInitParam();
    if(variance > 0.0)
      white->setVariance(variance);
    k = white;
  }
  else
    throw ndlexceptions::Error(
      "Unknown or unsupported kernel for Bayesian GPLVM (need psi): " + type
      + ". Use rbf, lin, bias, white.");
  return k;
}

void writeLatentText(const string& base, const CMatrix& mu, const CMatrix& S)
{
  mu.toUnheadedFile(base + "_mu.dat");
  S.toUnheadedFile(base + "_S.dat");
}

} // namespace

void CClbgplvm::learn()
{
  incrementArgument();
  setMode("learn");

  vector<string> kernelTypes;
  vector<double> rbfInvWidths;
  vector<double> variances;
  string optimiser = "scg";
  bool centreData = true;
  int iters = 100;
  int latentDim = 2;
  int numInducing = -1;
  string modelFileName = "bgplvm_model";
  string latentOutBase = "";
  bool dumpLatent = false;

  while(isFlags())
  {
    if(isCurrentArgumentFlag())
    {
      if(isCurrentArg("-?", "--?") || isCurrentArg("-h", "--help"))
      {
        helpInfo();
        exitNormal();
      }
      else if(isCurrentArg("-x", "--latent-dim"))
      {
        incrementArgument();
        latentDim = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-M", "--num-inducing"))
      {
        incrementArgument();
        numInducing = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-C", "--Centre-data"))
      {
        incrementArgument();
        centreData = getBoolFromCurrentArgument();
      }
      else if(isCurrentArg("-O", "--optimiser"))
      {
        incrementArgument();
        optimiser = getCurrentArgument();
      }
      else if(isCurrentArg("-k", "--kernel"))
      {
        incrementArgument();
        kernelTypes.push_back(getCurrentArgument());
        rbfInvWidths.push_back(-1.0);
        variances.push_back(-1.0);
      }
      else if(isCurrentArg("-g", "--gamma"))
      {
        incrementArgument();
        if(kernelTypes.size() == 0)
          exitError("Inverse width (-g) must follow a -k rbf.");
        if(kernelTypes.back() != "rbf")
          exitError("Inverse width (-g) only valid for RBF.");
        // Match ./gplvm: CLI gamma is half inverse-width.
        rbfInvWidths.back() = 2.0 * getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("-v", "--variance"))
      {
        incrementArgument();
        if(kernelTypes.size() == 0)
          exitError("Variance (-v) must follow a -k.");
        variances.back() = getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("-#", "--#iterations"))
      {
        incrementArgument();
        iters = getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-o", "--output-latent"))
      {
        incrementArgument();
        latentOutBase = getStringFromCurrentArgument();
        dumpLatent = true;
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

  CMatrix Y;
  CMatrix labs;
  readData(Y, labs, trainDataFileName);
  if(centreData)
    centreColumns(Y);

  if(numInducing < 0)
    numInducing = (int)Y.getRows() < 10 ? (int)Y.getRows() : 10;
  if(numInducing < 1 || (unsigned int)numInducing > Y.getRows())
    exitError("num-inducing (-M) must be in 1..N.");
  if(latentDim < 1)
    exitError("latent-dim (-x) must be positive.");

  if(kernelTypes.size() == 0)
  {
    kernelTypes.push_back("rbf");
    rbfInvWidths.push_back(-1.0);
    variances.push_back(-1.0);
  }

  CMatrix Xmeta(1, (unsigned int)latentDim);
  Xmeta.zeros();
  CCmpndKern kern(Xmeta);
  for(size_t i = 0; i < kernelTypes.size(); i++)
  {
    CKern* comp = makeKernComponent(kernelTypes[i], (unsigned int)latentDim,
                                    rbfInvWidths[i], variances[i]);
    kern.addKern(comp);
  }

  CBayesGplvm model(&kern, Y, (unsigned int)latentDim,
                    (unsigned int)numInducing, getVerbosity());

  // SVM-light first field is often a class label (oil 1/2/3); store for gnuplot.
  if(labs.getCols() >= 1 && labs.getRows() == Y.getRows())
  {
    vector<int> classLabs(Y.getRows());
    bool anyNonZero = false;
    for(unsigned int i = 0; i < Y.getRows(); i++)
    {
      classLabs[i] = (int)labs.getVal(i, 0);
      if(classLabs[i] != 0)
        anyNonZero = true;
    }
    if(anyNonZero)
      model.setLabels(classLabs);
  }

  if(optimiser == "scg")
    model.setDefaultOptimiser(CBayesGplvm::SCG);
  else if(optimiser == "conjgrad")
    model.setDefaultOptimiser(CBayesGplvm::CG);
  else if(optimiser == "quasinew" || optimiser == "lbfgs")
    model.setDefaultOptimiser(CBayesGplvm::BFGS);
  else if(optimiser == "graddesc")
  {
    model.setDefaultOptimiser(CBayesGplvm::GD);
    model.setLearnRate(1e-4);
    model.setMomentum(0.9);
  }
  else
    exitError("Unrecognised optimiser (use scg, conjgrad, quasinew/lbfgs, graddesc).");

  if(getVerbosity() > 0)
  {
    cout << "Training Bayesian GPLVM: N=" << Y.getRows()
         << " D=" << Y.getCols() << " q=" << latentDim
         << " M=" << numInducing << " iters=" << iters << endl;
  }
  model.optimise((unsigned int)iters);

  string comment = "Run as:";
  for(int i = 0; i < argc; i++)
  {
    comment += " ";
    comment += argv[i];
  }
  comment += " with seed " + ndlstrutil::itoa(getSeed()) + ".";
  if(getFileFormat() != 0)
    exitError("bgplvm currently writes only the native text .model format (-f 0).");
  model.toFile(modelFileName, comment);

  if(dumpLatent)
  {
    if(latentOutBase.size() == 0)
      latentOutBase = modelFileName;
    writeLatentText(latentOutBase, model.getMu(), model.getS());
  }

  if(getVerbosity() > 0)
    model.display(cout);
  exitNormal();
}

void CClbgplvm::display()
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

  CMatrix Ytmp(1, 1);
  CRbfKern kernTmp(1);
  kernTmp.setInitParam();
  CBayesGplvm model(&kernTmp, Ytmp, 1, 1, getVerbosity());
  model.fromFile(modelFileName);
  model.display(cout);
  exitNormal();
}

void CClbgplvm::gnuplot()
{
  incrementArgument();
  setMode("gnuplot");
  double pointSize = 2;
  int resolution = 80;
  string name = "bgplvm";
  string modelFileName = "bgplvm_model";
  string labelFileName = "";
  while(isFlags())
  {
    if(isCurrentArgumentFlag())
    {
      if(isCurrentArg("-?", "--?") || isCurrentArg("-h", "--help"))
      {
        helpInfo();
        exitNormal();
      }
      else if(isCurrentArg("-l", "--labels"))
      {
        incrementArgument();
        labelFileName = getStringFromCurrentArgument();
      }
      else if(isCurrentArg("-p", "--point-size"))
      {
        incrementArgument();
        pointSize = getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("-r", "--resolution"))
      {
        incrementArgument();
        resolution = getIntFromCurrentArgument();
      }
      else
        unrecognisedFlag();
      incrementArgument();
    }
    else
      setFlags(false);
  }
  if(getCurrentArgumentNo() < argc)
    modelFileName = argv[getCurrentArgumentNo()];
  if((getCurrentArgumentNo() + 1) < argc)
    name = argv[getCurrentArgumentNo() + 1];

  CMatrix Ytmp(1, 1);
  CRbfKern kernTmp(1);
  kernTmp.setInitParam();
  CBayesGplvm model(&kernTmp, Ytmp, 1, 1, getVerbosity());
  model.fromFile(modelFileName);
  if(model.getLatentDim() != 2)
    exitError("Plotting is only implemented for 2 dimensional latent spaces.");

  if(labelFileName.length() > 0)
  {
    ifstream in(labelFileName.c_str());
    string line;
    if(!in)
      throw ndlexceptions::FileReadError(labelFileName);
    vector<int> labels;
    while(getline(in, line))
    {
      int index = atol(line.c_str());
      if(index < 0)
        throw ndlexceptions::FileFormatError(labelFileName);
      labels.push_back(index);
    }
    in.close();
    if(labels.size() != model.getNumData())
      exitError("Incorrect number of labels.");
    model.setLabels(labels);
  }

  vector<bool> labelFileWritten;
  int maxVal = 0;
  int minVal = 0;
  const CMatrix& Xlat = model.getMu();
  if(model.isLabels())
  {
    maxVal = model.getMaxLabelVal();
    minVal = model.getMinLabelVal();
    for(int j = minVal; j <= maxVal; j++)
    {
      vector<int> storeIndex;
      for(unsigned int i = 0; i < model.getNumData(); i++)
        if(model.getLabel(i) == j)
          storeIndex.push_back((int)i);
      if(storeIndex.size() > 0)
      {
        CMatrix Xtemp(storeIndex.size(), 3, 0.1);
        for(size_t i = 0; i < storeIndex.size(); i++)
          for(unsigned int k = 0; k < 2; k++)
            Xtemp.setVal(Xlat.getVal(storeIndex[i], k), (unsigned int)i, k);
        Xtemp.toUnheadedFile(name + "_latent_data" + ndlstrutil::itoa(j) + ".dat");
        labelFileWritten.push_back(true);
      }
      else
        labelFileWritten.push_back(false);
    }
  }
  else
  {
    CMatrix Xext(Xlat.getRows(), 3, 0.1);
    for(unsigned int i = 0; i < Xlat.getRows(); i++)
      for(unsigned int j = 0; j < 2; j++)
        Xext.setVal(Xlat.getVal(i, j), i, j);
    Xext.toUnheadedFile(name + "_latent_data.dat");
  }

  CMatrix minVals(1, Xlat.getCols());
  CMatrix maxVals(1, Xlat.getCols());
  Xlat.maxRow(maxVals);
  Xlat.minRow(minVals);
  const int numx = resolution;
  const int numy = resolution;
  double xspan = maxVals.getVal(0, 0) - minVals.getVal(0, 0);
  double yspan = maxVals.getVal(0, 1) - minVals.getVal(0, 1);
  double xdiff = 1.1 * xspan / (double)(numx - 1);
  double ydiff = 1.1 * yspan / (double)(numy - 1);
  CMatrix Xgrid(numx * numy, 2);
  vector<CMatrix*> matrixOut;
  int i;
  int j;
  double y;
  double x;
  for(i = 0, y = minVals.getVal(0, 1) - 0.05 * yspan; i < numy; y += ydiff, i++)
  {
    matrixOut.push_back(new CMatrix(numx, 3));
    for(j = 0, x = minVals.getVal(0, 0) - 0.05 * xspan; j < numx; x += xdiff, j++)
    {
      Xgrid.setVal(x, i * numy + j, 0);
      matrixOut[i]->setVal(x, j, 0);
      Xgrid.setVal(y, i * numy + j, 1);
      matrixOut[i]->setVal(y, j, 1);
    }
  }
  CMatrix means(Xgrid.getRows(), model.getOutputDim());
  CMatrix variances(Xgrid.getRows(), model.getOutputDim());
  model.posteriorMeanVar(means, variances, Xgrid);
  for(i = 0; i < numy; i++)
    for(j = 0; j < numx; j++)
      matrixOut[i]->setVal(-log(variances.getVal(i * numy + j)), j, 2);

  string matrixFile = name + "_variance_matrix.dat";
  ofstream out(matrixFile.c_str());
  if(!out)
    throw ndlexceptions::FileWriteError(matrixFile);
  out << setiosflags(ios::scientific);
  out << setprecision(17);
  out << "# Prepared plot of Bayesian GPLVM model file" << endl;
  for(i = 0; i < numy; i++)
  {
    matrixOut[i]->toUnheadedStream(out);
    out << endl;
    delete matrixOut[i];
  }
  out.close();

  string plotFileName = name + "_plot.gp";
  ofstream outGnuplot(plotFileName.c_str());
  if(!outGnuplot)
    throw ndlexceptions::FileWriteError(plotFileName);
  writeGnuplotTermSelect(outGnuplot);
  writeGnuplotStyleHeader(outGnuplot);
  outGnuplot << "set palette defined (0 \"#eeeeec\", 1 \"#2e3436\")" << endl;
  outGnuplot << "set pm3d map" << endl;
  outGnuplot << "set pm3d explicit" << endl;
  outGnuplot << "unset key" << endl;
  outGnuplot << "splot \"" << name << "_variance_matrix.dat\" with pm3d notitle";
  const char* labelCols[] = {
    "#204a87", "#a40000", "#4e9a06", "#ce5c00",
    "#c4a000", "#5c3566", "#8f5902", "#2e3436"
  };
  const int nLabelCols = 8;
  if(model.isLabels())
  {
    for(int lab = minVal, idx = 0; lab <= maxVal; lab++, idx++)
      if(labelFileWritten[idx])
        outGnuplot << ", \"" << name << "_latent_data" << lab
                   << ".dat\" with points pt 7 ps " << pointSize
                   << " lc rgb \"" << labelCols[(lab - minVal) % nLabelCols] << "\"";
  }
  else
    outGnuplot << ", \"" << name << "_latent_data.dat\" with points ls 3 ps " << pointSize;
  outGnuplot << endl;
  writeGnuplotFinish(outGnuplot);
  outGnuplot.close();
  if(getVerbosity() > 0)
    cout << "Wrote " << plotFileName << " (open with: gnuplot " << plotFileName << ")" << endl;
  exitNormal();
}

void CClbgplvm::helpHeader()
{
  cout << "Bayesian GP-LVM (bgplvm) Command Line Tool." << endl;
  cout << "CIP-0017: variational q(X) with psi statistics." << endl;
}

void CClbgplvm::helpInfo()
{
  string command = getMode();
  if(command == "bgplvm")
  {
    helpHeader();
    helpUsage("bgplvm [options] command [command-options]");
    cout << "Commands:" << endl;
    helpArgument("learn", "Train a Bayesian GP-LVM from SVM-light data.");
    helpArgument("display", "Show a saved bgplvm .model file.");
    helpArgument("gnuplot", "Write gnuplot scripts for a 2-D latent visualisation.");
    helpDescriptor("For help on a specific command type bgplvm command --h");
    cout << endl;
    cout << "Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-v, --verbosity [0..3]", "Verbosity level (default is set to 2).");
    helpArgument("-s, --seed long", "Set random seed (default is to use current time in seconds).");
  }
  else if(command == "learn")
  {
    helpHeader();
    helpUsage("bgplvm [options] learn [command-options] example_file [model_file]");
    helpDescriptor("Learn a Bayesian GP-LVM (VarDTC + KL). Default optimiser is SCG.");
    cout << "Arguments:" << endl;
    helpArgument("example_file", "SVM-light training file (observations Y).");
    helpArgument("model_file", "Output model path (default bgplvm_model).");
    cout << endl;
    cout << "Command Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-x, --latent-dim int", "Latent dimension q (default 2).");
    helpArgument("-M, --num-inducing int", "Inducing points M (default min(10,N)).");
    helpArgument("-k, --kernel string", "Add kernel with psi support: rbf, lin, bias, white. Repeatable. Default: rbf.");
    helpArgument("-g, --gamma float", "RBF gamma as in ./gplvm (inverseWidth=2*gamma). Follows -k rbf.");
    helpArgument("-v, --variance float", "Kernel variance; follows -k.");
    helpArgument("-#, --#iterations int", "Optimiser iterations (default 100).");
    helpArgument("-O, --optimiser string", "scg (default), conjgrad, quasinew/lbfgs, graddesc.");
    helpArgument("-C, --Centre-data bool", "Mean-centre observation columns (default 1).");
    helpArgument("-o, --output-latent string", "Write mu/S to basename_mu.dat and basename_S.dat.");
    helpArgument("-f, --file-format int", "Model format; only 0 (text) supported.");
  }
  else if(command == "display")
  {
    helpHeader();
    helpUsage("bgplvm [options] display model_file");
    helpDescriptor("Display parameters and ELBO for a saved Bayesian GP-LVM model.");
  }
  else if(command == "gnuplot")
  {
    helpHeader();
    helpUsage("bgplvm [options] gnuplot [command-options] model_file [name]");
    helpDescriptor("Write latent scatter (variational mean mu) and predictive-variance background for q=2.");
    cout << "Arguments:" << endl;
    helpArgument("model_file", "Saved Bayesian GPLVM model.");
    helpArgument("name", "Stub for output files (default bgplvm).");
    cout << endl;
    cout << "Command Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-l, --labels string", "Optional labels file (one integer per line); overrides labels stored in the model.");
    helpArgument("-p, --point-size float", "Point size (default 2).");
    helpArgument("-r, --resolution int", "Grid resolution for variance map (default 80).");
  }
}
