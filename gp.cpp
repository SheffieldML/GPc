#include "gp.h"

int main(int argc, char* argv[])
{
  CClgp command(argc, argv);
  command.setFlags(true);
  command.setVerbosity(2);
  command.setMode("gp");
  try {    
    while(command.isFlags())
    {
      string argument = command.getCurrentArgument();
      if(argv[command.getCurrentArgumentNo()][0]=='-')
      {
	if (command.isCurrentArg("-?", "--?")) 
	{ 
	  command.helpInfo(); 
	  command.exitNormal();
	}
	else if(command.isCurrentArg("-h", "--help"))
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
	else if(argument=="learn") // learning a model.
	  command.learn();	  
	else if(argument=="relearn") // initialise with an old model.
	  command.relearn();	  
	//else if(argument=="test") // test with a model.
	//	  command.test();	  
	//else if(argument=="log-likelihood") // compute model log likelihood.
	// command.logLikelihood();	  
	//else if(argument=="predict") 
	//  command.predict();
	else if(argument=="display") 
	  command.display();
	else if(argument=="gnuplot") 
	  command.gnuplot();
	else  
	  command.exitError("Invalid gp command provided.");
      command.incrementArgument();
    }
    command.exitError("No gp command provided.");
  }
  catch(ndlexceptions::CommandLineError err) {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::FileFormatError err) {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::FileReadError err) {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::FileWriteError err) {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::FileError err) {
    command.exitError(err.getMessage());
  }
  catch(ndlexceptions::Error err) {
    command.exitError(err.getMessage());
  }
  catch(std::bad_alloc err) {
    command.exitError("Out of memory.");
  }
  catch(std::exception err) {
    command.exitError("Unhandled exception.");
  }
}
CClgp::CClgp(int arc, char** arv) : CClctrl(arc, arv)
{
}
void CClgp::learn()
{
  incrementArgument();
  setMode("learn");
  
  enum {
    KERNEL_USAGE_BACK,
    KERNEL_USAGE_FWD,
    KERNEL_USAGE_DYN
  };

  double tol=1e-6;
  string optimiser="scg";

  vector<string> kernelTypes;
  vector<unsigned int> kernelUsageFlag;
  vector<double> ratQuadAlphas;
  vector<double> rbfInvWidths;
  vector<double> weightVariances;
  vector<double> biasVariances;
  vector<double> variances;
  vector<double> degrees;
  vector<bool> selectInputs;
  bool centreData=true;
  bool scaleData=false;
  bool outputScaleLearnt=false;
  int activeSetSize = -1;
  int approxType = -1;
  string approxTypeStr = "ftc";
  bool labelsProvided = true;
  double signalVariance = 0.0;
  vector<int> labels;
  int iters=1000;
  string modelFileName="gp_model";
  while(isFlags()) {
    if(isCurrentArgumentFlag()) {
      if (isCurrentArg("-?", "--?")) {
        helpInfo(); 
        exitNormal();
      }
      else if (isCurrentArg("-h", "--help")) {
        helpInfo(); 
        exitNormal();
      }      
      else if (isCurrentArg("-C", "--Centre-data")) {
        incrementArgument(); 
        centreData=getBoolFromCurrentArgument();
      }
      else if (isCurrentArg("-L", "--Learn-scales")) {
        incrementArgument();
        outputScaleLearnt=getBoolFromCurrentArgument();
      }
      else if (isCurrentArg("-S", "--Scale-data")) {
        incrementArgument(); 
        scaleData=getBoolFromCurrentArgument();
      }
      else if(isCurrentArg("-a", "--active-set-size")) {
	incrementArgument();
	activeSetSize = getIntFromCurrentArgument();
      }
      else if (isCurrentArg("-A", "--Approximation-type")) {
	incrementArgument();
	approxTypeStr = getCurrentArgument();
      }
      else if (isCurrentArg("-k", "--kernel")) {
        incrementArgument(); 
        kernelTypes.push_back(getCurrentArgument()); 
        kernelUsageFlag.push_back(KERNEL_USAGE_FWD);
        ratQuadAlphas.push_back(-1.0);
		rbfInvWidths.push_back(-1.0);
        weightVariances.push_back(-1.0);
        biasVariances.push_back(-1.0);
        variances.push_back(-1.0);
        degrees.push_back(-1.0);
        selectInputs.push_back(false);
      }
      else if (isCurrentArg("-g", "--gamma")) {
        incrementArgument(); 
        if(kernelTypes.size()==0)
          exitError("Inverse width specification must come after covariance function type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="rbf" && kernelTypes[kernelTypes.size()-1]!="exp" && kernelTypes[kernelTypes.size()-1]!="ratquad")
          exitError("Inverse width parameter only valid for RBF, exponential and rational quadratic covariance function.");
        rbfInvWidths[rbfInvWidths.size()-1]=2*getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-d", "--degree")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("Polynomial degree specification must come after covariance function type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="poly")
          exitError("Polynomial degree parameter only valid for poly covariance function.");
        degrees[degrees.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-w", "--weight")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("`Weight variance' parameter specification must come after covariance function type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="poly" 
           && kernelTypes[kernelTypes.size()-1]!="mlp")
          exitError("`Weight variance' parameter only valid for polynomial and MLP covariance function.");
        weightVariances[weightVariances.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-b", "--bias")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("`Bias variance' parameter specification must come after covariance function type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="poly" 
           && kernelTypes[kernelTypes.size()-1]!="mlp")
          exitError("`Bias variance' parameter only valid for polynomial and MLP covariance function.");
        biasVariances[biasVariances.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-v", "--variance")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("Variance parameter specification must come after covariance function type is specified.");
        variances[variances.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-i", "--input-select")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("Input selection flag must come after covariance function type is specified.");
        selectInputs[selectInputs.size()-1]=getBoolFromCurrentArgument();
      }
      else if(isCurrentArg("-O", "--optimiser")) {
	incrementArgument(); 
	optimiser=getCurrentArgument(); 
	    }
     else if (isCurrentArg("-#", "--#iterations")) {
        incrementArgument();
        iters=getIntFromCurrentArgument();
      }
      else if (isCurrentArg("-f", "--file-format")) {
        incrementArgument();
        setFileFormat(getIntFromCurrentArgument());
      }
      else {
        unrecognisedFlag();
      }
      incrementArgument();
    }
    else
      setFlags(false);
  }
  if(getCurrentArgumentNo()>=argc) 
    exitError("There are not enough input parameters.");
  string trainDataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc) 
    modelFileName=argv[getCurrentArgumentNo()+1];
  string dataFile="";
  CMatrix X;
  CMatrix y;
  readData(X, y, trainDataFileName);
  int inputDim = X.getCols();    
  
  // create covariance function.
  CCmpndKern kern(X);
  vector<CKern*> kernels;
  for(int i=0; i<kernelTypes.size(); i++) {
    CMatrix *M = 0;
    M = &X;
    if(kernelTypes[i]=="lin") {
      if(selectInputs[i])
        kernels.push_back(new CLinardKern(*M));
      else
        kernels.push_back(new CLinKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 0); // set variance parameter as specified.
    }
    else if(kernelTypes[i]=="poly") {
      if(selectInputs[i]) {
        kernels.push_back(new CPolyardKern(*M));
        if(degrees[i]!=-1.0)
          ((CPolyardKern*)kernels[i])->setDegree(degrees[i]);
      }
      else {
        kernels.push_back(new CPolyKern(*M));	
        if(degrees[i]!=-1.0)
          ((CPolyKern*)kernels[i])->setDegree(degrees[i]);
      }
      if(weightVariances[i]!=-1.0)
        kernels[i]->setParam(weightVariances[i], 0); // set `weight variance' parameter as specified.
      if(biasVariances[i]!=-1.0)
        kernels[i]->setParam(biasVariances[i], 1); // set `bias variance' parameter as specified.
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 2); // set variance parameter as specified.
    }
    else if(kernelTypes[i]=="rbf") {
      if(selectInputs[i])
        kernels.push_back(new CRbfardKern(*M));
      else
        kernels.push_back(new CRbfKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setParam(rbfInvWidths[i], 0); /// set rbf inverse width as specified.
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 1); /// set variance parameter as specified.
	}
    else if(kernelTypes[i]=="exp") {
      if(selectInputs[i])
        exitError("Exponential covariance function not available with input selection yet.");
      else
        kernels.push_back(new CExpKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setParam(rbfInvWidths[i], 0); /// set exp inverse width as specified.
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 1); /// set variance parameter as specified.
	}
    else if(kernelTypes[i]=="ratquad") {
      if(selectInputs[i])
        exitError("Rational quadratic covariance function not available with input selection yet.");
      else
        kernels.push_back(new CRatQuadKern(*M));
      if(ratQuadAlphas[i]!=-1.0)
        kernels[i]->setParam(ratQuadAlphas[i], 0); /// set rat quad length scale as specified.
	  if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setParam(1/sqrt(rbfInvWidths[i]), 1); /// set rat quad length scale as specified.
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 2); /// set variance parameter as specified.
    }
    else if(kernelTypes[i] == "mlp") {
      if(selectInputs[i])
        kernels.push_back(new CMlpardKern(*M));
      else
        kernels.push_back(new CMlpKern(*M));
      if(weightVariances[i]!=-1.0)
        kernels[i]->setParam(weightVariances[i], 0); // set `weight variance' parameter as specified.
      if(biasVariances[i]!=-1.0)
        kernels[i]->setParam(biasVariances[i], 1); // set `bias variance' parameter as specified.
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 2); // set variance parameter as specified.
    }
    else if(kernelTypes[i] == "bias" && kernelUsageFlag[i]!=KERNEL_USAGE_FWD) {
      // fwd covariance function always has bias component
      kernels.push_back(new CBiasKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 0); // set variance parameter as specified.
    }
    else if(kernelTypes[i] == "white" && kernelUsageFlag[i]!=KERNEL_USAGE_FWD) {
      // fwd covariance function always includes a white noise component
      kernels.push_back(new CWhiteKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 0); // set variance parameter as specified.
    }
    else {
      exitError("Unknown covariance function type: " + kernelTypes[i]);
    }
    switch (kernelUsageFlag[i]) {
    case KERNEL_USAGE_FWD:
      kern.addKern(kernels[i]);
      break;
    case KERNEL_USAGE_BACK:
      break;
    case KERNEL_USAGE_DYN:
      break;
    }
  }
  // if no covariance function was specified, add an RBF.
  if(kern.getNumKerns()==0)
  {
    CKern* defaultKern = new CRbfKern(X);
    kern.addKern(defaultKern);
  }
  CKern* biasKern = new CBiasKern(X);
  CKern* whiteKern = new CWhiteKern(X);
  kern.addKern(biasKern);
  kern.addKern(whiteKern);

  if(approxTypeStr=="ftc") {
    approxType = CGp::FTC;
    activeSetSize = -1;
  }
  else {
    if(approxTypeStr=="dtc") {
      approxType = CGp::DTC;
    }
    else if(approxTypeStr=="dtcvar") {
      approxType = CGp::DTCVAR;
      cout << "Warning: numerical stabilities exist in DTCVAR approximation." << endl;
    }
    else if(approxTypeStr=="fitc") {
      approxType = CGp::FITC;
      exitError("FITC Approximation currently not working.");
    }
    else if(approxTypeStr=="pitc") {
      approxType = CGp::PITC;
    }
    else {
      exitError("Unknown sparse approximation type: " + approxTypeStr + ".");
    }
    if(activeSetSize==-1)
      exitError("You must choose an active set size (option -a) for the command learn.");
  }


  
  CGaussianNoise noise(&y);
  noise.setBias(0.0);
  // Remove scales and center if necessary.
  CMatrix scale(1, y.getCols(), 1.0);
  CMatrix bias(1, y.getCols(), 0.0);
  if(centreData)
    bias.deepCopy(meanCol(y));
  if(scaleData)
    scale.deepCopy(stdCol(y));
  

  CGp* pmodel;
  CMatrix bK(1,1,0.0);
  pmodel = new CGp(&kern, &noise, &X, approxType, activeSetSize, getVerbosity());
  if(optimiser=="scg")
    pmodel->setDefaultOptimiser(CGp::SCG);
  else if(optimiser=="conjgrad")
    pmodel->setDefaultOptimiser(CGp::CG);
  else if(optimiser=="graddesc")
    pmodel->setDefaultOptimiser(CGp::GD);
  else if(optimiser=="quasinew")
    pmodel->setDefaultOptimiser(CGp::BFGS);
  else
    exitError("Unrecognised optimiser type: " + optimiser);
  pmodel->setBetaVal(1); //
  pmodel->setScale(scale);
  pmodel->setBias(bias);
  pmodel->updateM();
  pmodel->setOutputScaleLearnt(outputScaleLearnt);
  //writeGpToFile(*pmodel, "c:\\gp_model", "Write for testing of model");  
  pmodel->optimise(iters);
  string comment="";
  switch(getFileFormat()) {
  case 0: /// GP file format.
    comment = "Run as:";
    for(int i=0; i<argc; i++) {
      comment+=" ";
      comment+=argv[i];
    }
    comment += " with seed " + ndlstrutil::itoa(getSeed()) + ".";
    writeGpToFile(*pmodel, modelFileName, comment);
    break;
  case 1: /// Matlab file format.
#ifdef _NDLMATLAB
    // Write matlab output.
    pmodel->writeMatlabFile(modelFileName, "gpInfo");
    pmodel->getKernel()->updateMatlabFile(modelFileName, "kern");
    X.updateMatlabFile(modelFileName, "X");
    y.updateMatlabFile(modelFileName, "y");
#else 
    exitError("Error MATLAB not incorporated at compile time.");
#endif
    break;
  default:
    exitError("Unrecognised file format number.");
      
  }
  exitNormal();
}

void CClgp::relearn()
{
  incrementArgument();
  setMode("relearn");
  string optimiser="scg";

  int iters=1000;
  string modelFileName="gp_model";
  string newModelFileName="gp_model";

  while(isFlags()) {
    if(isCurrentArgumentFlag()) {
      if(isCurrentArg("-?", "--?")) {
	helpInfo(); 
	exitNormal();
      }
      else if(isCurrentArg("-h", "--help")) {
	helpInfo(); 
	exitNormal();
      }
      else if(isCurrentArg("-O", "--optimiser")) {
	incrementArgument(); 
	optimiser=getCurrentArgument(); 
      }
      else if(isCurrentArg("-#", "--#iterations")) {
	incrementArgument();
	iters=getIntFromCurrentArgument();
      }	
      else {
	unrecognisedFlag();
      }
      incrementArgument();
    }
    else {
      setFlags(false);
    }
  }
  if(getCurrentArgumentNo()>=argc) 
    exitError("There are not enough input parameters.");
  
  string trainDataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc) 
    modelFileName=argv[getCurrentArgumentNo()+1];
  if((getCurrentArgumentNo()+2)<argc) 
    newModelFileName=argv[getCurrentArgumentNo()+2];
  CMatrix X;
  CMatrix y;
  readData(X, y, trainDataFileName);
  CGp* pmodel=readGpFromFile(modelFileName, getVerbosity());
  pmodel->py = &y;
  pmodel->updateM();
  pmodel->pX = &X;
  
  if(optimiser=="scg")
    pmodel->setDefaultOptimiser(CGp::SCG);
  else if(optimiser=="conjgrad")
    pmodel->setDefaultOptimiser(CGp::CG);
  else if(optimiser=="graddesc")
    pmodel->setDefaultOptimiser(CGp::GD);
  else if(optimiser=="quasinew")
    pmodel->setDefaultOptimiser(CGp::BFGS);
  else
    exitError("Unrecognised optimiser type: " + optimiser);
  if(pmodel->getInputDim()!=X.getCols())
    throw ndlexceptions::Error(trainDataFileName + ": input data is not of correct dimension");  
  
  pmodel->optimise(iters);
  string comment="";
  switch(getFileFormat()) {
  case 0: /// GP file format.
    comment = "Run as:";
    for(int i=0; i<argc; i++) {
      comment+=" ";
      comment+=argv[i];
    }
    comment += " with seed " + ndlstrutil::itoa(getSeed()) + ".";
    writeGpToFile(*pmodel, newModelFileName, comment);
    break;
  case 1: /// Matlab file format.
#ifdef _NDLMATLAB
    // Write matlab output.
    pmodel->writeMatlabFile(newModelFileName, "gpInfo");
    pmodel->getKernel()->updateMatlabFile(newModelFileName, "kern");
    X.updateMatlabFile(newModelFileName, "X");
    y.updateMatlabFile(newModelFileName, "y");
#else 
    exitError("Error MATLAB not incorporated at compile time.");
#endif
    break;
  default:
    exitError("Unrecognised file format number.");
    
  }
  
  exitNormal();
}

void CClgp::display()
{
  incrementArgument();
  setMode("display");
  while(isFlags()) {
    if(isCurrentArgumentFlag()) {
      if(getCurrentArgumentLength()!=2) {
        unrecognisedFlag();
      }
      if (isCurrentArg("-?", "--?") ||isCurrentArg("-h", "--help")) {
        helpInfo(); 
        exitNormal();
      }
      else {
        unrecognisedFlag();
      }
      incrementArgument();
    }
    else
      setFlags(false);
  }
  string modelFileName = "";
  if(getCurrentArgumentNo()>=argc)
    modelFileName="gp_model";
  else
    modelFileName=getCurrentArgument();
  CGp* pmodel=readGpFromFile(modelFileName, getVerbosity());
  pmodel->display(cout);
  exitNormal();
}

void CClgp::gnuplot()
{
  incrementArgument();
  setMode("gnuplot");
  double pointSize = 2;
  double lineWidth = 2;
  int resolution = 80;
  string name = "gp";
  string modelFileName="gp_model";
  string labelFileName="";
  while(isFlags()) {
    if(isCurrentArgumentFlag()) {
      int j=1;
      if(getCurrentArgumentLength()!=2)
	unrecognisedFlag();
      else if(isCurrentArg("-?","--?") || isCurrentArg("-h","--help")) {
	helpInfo(); 
	exitNormal();
      }
      else if (isCurrentArg("-l", "--labels")) {
	incrementArgument();
	labelFileName=getStringFromCurrentArgument(); 
      }
      else if (isCurrentArg("-p", "--point-size")) {
	incrementArgument();
	pointSize=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-r", "--resolution")) {
	incrementArgument();
	resolution=getIntFromCurrentArgument(); 
      }
      else {
	unrecognisedFlag();
      }
      incrementArgument();
    }
    else
      setFlags(false);
    }
  if(getCurrentArgumentNo()>=argc) 
    exitError("There are not enough input parameters.");
  string dataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc) 
    modelFileName=argv[getCurrentArgumentNo()+1];
  if((getCurrentArgumentNo()+2)<argc) 
    name=argv[getCurrentArgumentNo()+2];
  string outputFileName=name+"_plot_data";
  if((getCurrentArgumentNo()+3)<argc) 
    outputFileName=argv[getCurrentArgumentNo()+2];
  CMatrix y;
  CMatrix X;
  readData(X, y, dataFileName);
  CGp* pmodel=readGpFromFile(modelFileName, getVerbosity());
  pmodel->py=&y;
  pmodel->updateM();
  pmodel->pX=&X;

  /// START HERE
  if(pmodel->getNoiseType()!="gaussian" && pmodel->getInputDim()!=2) {
    exitError("Incorrect number of model inputs.");
  }
  if(pmodel->getNoiseType()=="gaussian" && pmodel->getInputDim()>2) {
    exitError("Incorrect number of model inputs.");
  }

  if(X.getCols()!=pmodel->getInputDim()) {
    exitError("Incorrect dimension of input data.");
  }
  if(pmodel->getNoiseType()=="probit" || pmodel->getNoiseType()=="ncnm") {
    pmodel->X_u.toUnheadedFile(name+"_inducing_set.dat");
    int numPos = 0;
    int numNeg = 0;
    int numUnlab = 0;
    for(int i=0; i<X.getRows(); i++) {
      if(y.getVal(i)==1.0)
	numPos++;
      else if(y.getVal(i)==-1.0)
	numNeg++;
      else
	numUnlab++;
    }
    CMatrix XPos(numPos, X.getCols()+1);
    CMatrix XNeg(numNeg, X.getCols()+1);
    CMatrix XUnlab(numUnlab, X.getCols()+1);
    int posIndex = 0;
    int negIndex = 0;
    int unlabIndex = 0;
    for(int i=0; i<X.getRows(); i++) {
      if(y.getVal(i)==1.0) {
	for(int j=0; j<X.getCols(); j++)
	  XPos.setVal(X.getVal(i, j), posIndex, j);
	XPos.setVal(0, posIndex, XPos.getCols()-1);
	posIndex++;
      }
      else if(y.getVal(i)==-1.0) {
	for(int j=0; j<X.getCols(); j++)
	  XNeg.setVal(X.getVal(i, j), negIndex, j);
	XNeg.setVal(0, negIndex, XNeg.getCols()-1);
	negIndex++;
      }
      else {
	for(int j=0; j<X.getCols(); j++)
	  XUnlab.setVal(X.getVal(i, j), unlabIndex, j);
	XUnlab.setVal(0, unlabIndex, XUnlab.getCols()-1);
	unlabIndex++;
      }
    }
    if(numPos>0)
      XPos.toUnheadedFile(name+"_positive.dat");
    if(numNeg>0)
      XNeg.toUnheadedFile(name+"_negative.dat");
    if(numUnlab>0)
      XUnlab.toUnheadedFile(name+"_unlabelled.dat");
    CMatrix minVals(1, X.getCols());
    CMatrix maxVals(1, X.getCols());
    X.maxRow(maxVals);
    X.minRow(minVals);
    int numx=resolution;
    int numy=resolution;
    double xspan=maxVals.getVal(0, 0)-minVals.getVal(0, 0);
    double xdiff=xspan/(numx-1);
    double yspan= maxVals.getVal(0, 1)-minVals.getVal(0, 1);
    double ydiff=yspan/(numy-1);
    CMatrix Xgrid(numx*numy, 2);
    vector<CMatrix*> probOut; 
    double y = minVals.getVal(0, 1);
    double x = 0.0;
    for(int i=0; i<numy; y+=ydiff, i++) {
      probOut.push_back(new CMatrix(numx, 3));
      x=minVals.getVal(0, 0);
      for(int j=0; j<numx; x+=xdiff, j++) {
	Xgrid.setVal(x, i*numy+j, 0);
	probOut[i]->setVal(x, j, 0);
	Xgrid.setVal(y, i*numy+j, 1);
	probOut[i]->setVal(y, j, 1);
      }
    }
    CMatrix yTest(Xgrid.getRows(), pmodel->getOutputDim());
    yTest.ones();
    CMatrix probs(yTest.getRows(), yTest.getCols());
//TODO need to fill in for classification    pmodel->likelihoods(probs, yTest, Xgrid);
    for(int i=0; i<numy; i++)
      for(int j=0; j<numx; j++) {
	probOut[i]->setVal(probs.getVal(i*numy+j), j, 2);
      }
    string matrixFile = name + "_prob_matrix.dat";
    ofstream out(matrixFile.c_str());
    if(!out) throw ndlexceptions::FileWriteError(matrixFile);
    out << setiosflags(ios::scientific);
    out << setprecision(17);
    out << "# Prepared plot of model file " << endl;
    for(int i=0; i<numy; i++) {
      probOut[i]->toUnheadedStream(out);
      out << endl;
    }
    out.close();
    string plotFileName = name + "_plot.gp";
    ofstream outGnuplot(plotFileName.c_str());
    if(!outGnuplot) throw ndlexceptions::FileWriteError(plotFileName);
    outGnuplot << "set nosurface" << endl;
    outGnuplot << "set contour base" << endl;
    outGnuplot << "set cntrparam levels discrete 0.5" << endl;
    outGnuplot << "set term table # set output type to tables" << endl;
    outGnuplot << "set out '" << name << "_decision.dat'" << endl;
    outGnuplot << "splot \"" << name << "_prob_matrix.dat\"" << endl;
    outGnuplot << "set cntrparam levels discrete 0.25, 0.75" << endl;
    outGnuplot << "set out '" << name << "_contours.dat'" << endl;
    outGnuplot << "splot \"" << name << "_prob_matrix.dat\"" << endl;
    outGnuplot << "reset" << endl;
    outGnuplot << "set term x11" << endl;
    outGnuplot << "plot ";
    if(numPos>0)
      outGnuplot << "\"" << name << "_positive.dat\" with points ps " << pointSize << ", ";
    if(numNeg>0)
      outGnuplot << "\"" << name << "_negative.dat\" with points ps " << pointSize << ", ";
    outGnuplot << "\"" << name << "_active_set.dat\" with points ps " << pointSize*2 << ", ";
    if(numUnlab>0)
      outGnuplot << "\"" << name << "_unlabelled.dat\" with points ps " << pointSize << ", ";
    outGnuplot << "\"" << name << "_decision.dat\" with lines lw " << lineWidth << ", \"" << name << "_contours.dat\" with lines lw " << lineWidth << endl;
    outGnuplot << "pause -1" << endl;
    outGnuplot.close();
    
    
  }
  else if(pmodel->getNoiseType()=="gaussian") {
    switch(pmodel->getApproximationType())
    {
    case CGp::DTC:
    case CGp::DTCVAR:
    case CGp::FITC:
    case CGp::PITC:
      CMatrix scatterActive(pmodel->X_u.getRows(), pmodel->X_u.getCols()+1);
      CMatrix scatterOut(pmodel->X_u.getRows(), pmodel->getOutputDim());     
      pmodel->out(scatterOut, pmodel->X_u);
      for(int i=0; i<pmodel->X_u.getRows(); i++) 
      {
        for(int j=0; j<pmodel->X_u.getCols(); j++)
	  scatterActive.setVal(pmodel->X_u.getVal(i, j), i, j);
        scatterActive.setVal(scatterOut.getVal(i, 0), i, pmodel->X_u.getCols());
      }
      
      scatterActive.toUnheadedFile(name+"_active_set.dat");
    }
    CMatrix scatterData(X.getRows(), X.getCols()+1);
    for(int i=0; i<X.getRows(); i++) 
    {
      for(int j=0; j<X.getCols(); j++)
      {
	scatterData.setVal(X.getVal(i, j), i, j);
      }
      scatterData.setVal(y.getVal(i, 0), i, X.getCols());
    }
    scatterData.toUnheadedFile(name+"_scatter_data.dat");
    CMatrix minVals(1, X.getCols());
    CMatrix maxVals(1, X.getCols());
    X.maxRow(maxVals);
    X.minRow(minVals);
    
    if(pmodel->pX->getCols()==2) // two dimensional inputs
    {
      int numx=resolution;
      int numy=resolution;
      double xspan=maxVals.getVal(0, 0)-minVals.getVal(0, 0);
      double xdiff=xspan/(numx-1);
      double yspan= maxVals.getVal(0, 1)-minVals.getVal(0, 1);
      double ydiff=yspan/(numy-1);
      CMatrix Xgrid(numx*numy, 2);
      vector<CMatrix*> regressOut; 
      double x;
      double y;
      int i;
      int j;
      for(i=0, y=minVals.getVal(0, 1); i<numy; y+=ydiff, i++) 
      {
	regressOut.push_back(new CMatrix(numx, 3));
	for(j=0,  x=minVals.getVal(0, 0); j<numx; x+=xdiff, j++) 
	{
	  Xgrid.setVal(x, i*numy+j, 0);
	  regressOut[i]->setVal(x, j, 0);
	  Xgrid.setVal(y, i*numy+j, 1);
	  regressOut[i]->setVal(y, j, 1);
	}
      }
      CMatrix outVals(Xgrid.getRows(), pmodel->getOutputDim());
      pmodel->out(outVals, Xgrid);
      for(int i=0; i<numy; i++)
      {
	for(int j=0; j<numx; j++) 
	{
	  regressOut[i]->setVal(outVals.getVal(i*numy+j), j, 2);
	}
      }
      string matrixFile = name + "_output_matrix.dat";
      ofstream out(matrixFile.c_str());
      if(!out) throw ndlexceptions::FileWriteError(matrixFile);
      out << setiosflags(ios::scientific);
      out << setprecision(17);
      out << "# Prepared plot of model file " << endl;
      for(int i=0; i<numy; i++) 
      {
	regressOut[i]->toUnheadedStream(out);
	out << endl;
      }
      out.close();
      string plotFileName = name + "_plot.gp";
      ofstream outGnuplot(plotFileName.c_str());
      if(!outGnuplot) throw ndlexceptions::FileWriteError(plotFileName);
      outGnuplot << "splot \"" << name << "_output_matrix.dat\"  with lines lw " << lineWidth;
      outGnuplot << ", \"" << name << "_scatter_data.dat\" with points ps " << pointSize;
      if(pmodel->isSparseApproximation())
      {
      	outGnuplot << ", \"" << name << "_active_set.dat\" with points ps " << pointSize << endl;
      }
      outGnuplot << "pause -1";
      outGnuplot.close();
    }
	 
    else if(pmodel->X_u.getCols()==1) // one dimensional input.
    {
      double outLap=0.25;
      int numx=resolution;
      double xspan=maxVals.getVal(0, 0)-minVals.getVal(0, 0);
      maxVals.setVal(maxVals.getVal(0, 0)+outLap*xspan, 0, 0);
      minVals.setVal(minVals.getVal(0, 0)-outLap*xspan, 0, 0);
      xspan=maxVals.getVal(0, 0)-minVals.getVal(0, 0);
      double xdiff=xspan/(numx-1);
      CMatrix Xinvals(numx, 1);
      CMatrix regressOut(numx, 2);
      CMatrix errorBarPlus(numx, 2);
      CMatrix errorBarMinus(numx, 2);
      double x;
      int j;
      for(j=0, x=minVals.getVal(0, 0); j<numx; x+=xdiff, j++) 
      {
	Xinvals.setVal(x, j, 0);
	regressOut.setVal(x, j, 0);
	errorBarPlus.setVal(x, j, 0);
	errorBarMinus.setVal(x, j, 0);
      }
      CMatrix outVals(Xinvals.getRows(), pmodel->getOutputDim());
      CMatrix stdVals(Xinvals.getRows(), pmodel->getOutputDim());
      pmodel->out(outVals, stdVals, Xinvals);
      for(int j=0; j<numx; j++) {
	double val = outVals.getVal(j);
	regressOut.setVal(val, j, 1);
	errorBarPlus.setVal(val + 2*stdVals.getVal(j), j, 1);
	errorBarMinus.setVal(val - 2*stdVals.getVal(j), j, 1);
      }
      string lineFile = name + "_line_data.dat";
      regressOut.toUnheadedFile(lineFile);
      string errorFile = name + "_error_bar_data.dat";
      ofstream out(errorFile.c_str());
      if(!out) throw ndlexceptions::FileWriteError(errorFile);
      out << setiosflags(ios::scientific);
      out << setprecision(17);
      out << "# Prepared plot of model file " << endl;
      errorBarPlus.toUnheadedStream(out);
      out << endl;
      errorBarMinus.toUnheadedStream(out);
      string plotFileName = name + "_plot.gp";
      ofstream outGnuplot(plotFileName.c_str());
      if(!outGnuplot) throw ndlexceptions::FileWriteError(plotFileName);
      outGnuplot << "plot \"" << name << "_line_data.dat\" with lines lw " << lineWidth;
      outGnuplot << ", \"" << name << "_scatter_data.dat\" with points ps " << pointSize;
      if(pmodel->isSparseApproximation())
      {
        outGnuplot << ", \"" << name << "_active_set.dat\" with points ps " << pointSize;
      }
      outGnuplot << ", \"" << name << "_error_bar_data.dat\" with lines lw " << lineWidth << endl;
      outGnuplot << "pause -1";
      outGnuplot.close();
    }      
    
  }
  else 
  {
    exitError("Unknown noise model for gnuplot output.");
  }
  exitNormal();
}
void CClgp::helpInfo()
{
  string command = getMode();
  if(command=="gp") {
    helpHeader();
    helpUsage("gp [options] command [command-options]");
    cout << "Commands:" << endl;
    helpArgument("learn", "For learning the GP model.");
    helpArgument("display", "For displaying the parameters of a learned GP model.");
    helpArgument("gnuplot", "For plotting the results in gnuplot.");
    helpDescriptor("For help on a specific command type gp command --h");
    cout << endl;
    cout << "Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-v, --verbosity [0..3]", "Verbosity level (default is set to 2).");      
    helpArgument("-s, --seed long", "Set random seed (default is to use current time in seconds).");      
  }
  else if(command=="learn") {
    helpHeader();
    helpUsage("gp [options] learn example_file [model_file]");
    helpDescriptor("This file is for learning a data set with an GP. By default 1000 iterations of scaled conjugate gradient are used.");
    cout << "Arguments:" << endl;
    helpArgument("example_file", "File with the training data. The data file is in the format used by SVM light.");
    helpArgument("model_file", "File to store the resulting model in. By default this is set to gp_model.");
    cout << endl;
    cout << "Command Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-O, --optimiser string", "Optimiser to be used, valid options are scg, conjgrad and graddesc. Default is scg.");
    helpArgument("-#, --#iterations int", "Number of iterations for optimisation by scaled conjugate gradient. Default is 1000.");
    helpArgument("-x, --input-dim int", "Dimension for input space.");
    helpArgument("-C, --Centre-data bool", "Centre the data by removing the mean value. Default value is 1 (true).");
    helpArgument("-L, --Learn-scales bool", "Whether or not to learn the scaling on the input data. Default value is 0 (false).");
    helpArgument("-S, --Scale-data bool", "Scale the data by setting the standard deviation in each direction to 1.0. Default value is 0 (false).");
    helpArgument("-A, --Approximation-type string", "Sparse approximation type. Either FTC (no approximation), DTC (projected process approximation), DTCVAR (variational sparse approximation).");// FITC (fully independent training conditional --- not fully tested).");//, PITC (partially independent training conditional. Default value is FTC (no approximation).");
    //      helpArgument("-dh, --dynamics-hyperparameter double", "Specify weighting of the prior on the dynamics hyperparameters.");
    string m=getMode();
    setMode("kern");
    helpInfo();
    setMode(m);
  }
  else if(command=="kern") {
    cout << endl << "Covariance Function options:" << endl;
    helpArgument("-k, --kernel string", "Type of covariance function function. Currently available options are lin (linear), poly (polynomial -- not recommended), rbf (radial basis function), exp (exponential), ratquad (rational quadratic) and mlp (multi-layer perceptron otherwise known as arcsin). If the covariance function is not specified it defaults to rbf.");
    helpArgument("-g, --gamma float", "Inverse width parameter in RBF, exponential and rational quadratic covariance function.");
    helpArgument("-@, --alpha float", "Alpha parameter in rational quadratic covariance function.");
    helpArgument("-v, --variance float", "Variance parameter for covariance function.");
    helpArgument("-w, --weight float", "Weight parameter for polynomial and MLP covariance function.");
    helpArgument("-b, --bias float", "Bias parameter for polynomial and MLP covariance function.");
    helpArgument("-d, --degree int", "Degree parameter for polynomial covariance function.");
  }    
  else if(command=="display") {
    helpHeader();
    helpUsage("gp [options] display [model_file]");
    helpDescriptor("Summarise the contents of a model file for quick viewing. The model is loaded in and the parameters of the GP are given.");
    cout << "Arguments:" << endl;
    helpArgument("model_file", "File containing the model to be tested. By default this is set to gp_model.");
    cout << "Command Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
  }
  else if(command=="gnuplot") {
    helpHeader();
    helpUsage("gp [options] gnuplot [data_file] [model_file] [name]");
    helpDescriptor("This command creates files for displaying the visualisation in gnuplot.");
    cout << "Arguments:" << endl;
    helpArgument("data_file", "The data to plot, it can be test or training data, whatever you want to visualise. It should be in SVM light format.");
    helpArgument("model_file", "The GP model you want to use for generating the plot. The default is gp_model.");
    helpArgument("name", "The gnuplot script will be output as \"name_plot.gp\". Data files needed etc. will also be preceded by \"name\". Default is \"gp\". Requires gnuplot vs 4.0 or later.");
    cout << "Command Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-l, --labels string", "Label file name for visualisation of the data. If you wish to add labels to the data you can specify a file containing a vector of those labels here. By default any labels in the original data file are used.");
    helpArgument("-p, --point-size float", "Size of the point markers in gnuplot.");
    helpArgument("-r, --resolution int", "Resolution of the mesh grid in 3-d plots. The grid will have the given number of points in each direction. Default is 80.");
    
  }
}  

void CClgp::helpHeader() {
  cout << endl << "GP Code: Version 0.1" << endl;
}
