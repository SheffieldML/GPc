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
	else if(argument=="sample") // HMC samples of the hyperparameter posterior (CIP-003B).
	  command.sample();
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
  vector<int> arccosOrders;
  vector<bool> selectInputs;
  vector<string> setSpecs;
  bool centreData=true;
  bool scaleData=false;
  bool outputScaleLearnt=false;
  int activeSetSize = -1;
  int approxType = -1;
  string approxTypeStr = "ftc";
  string sparseLinalgStr = "lq";
  string inducingTypeStr = "points";
  double windowWidth = -1.0;
  bool allowWhiteAndNoise = false;
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
      else if (isCurrentArg("-L", "--sparse-linalg")) {
	incrementArgument();
	sparseLinalgStr = getCurrentArgument();
      }
      else if (isCurrentArg("--inducing", "--inducing")) {
        incrementArgument();
        inducingTypeStr = getCurrentArgument();
      }
      else if (isCurrentArg("--window-width", "--window-width")) {
        incrementArgument();
        windowWidth = getDoubleFromCurrentArgument();
      }
      else if (isCurrentArg("--allow-white-and-noise", "--allow-white-and-noise")) {
        allowWhiteAndNoise=true;
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
        arccosOrders.push_back(-1);
        selectInputs.push_back(false);
      }
      else if (isCurrentArg("--arccos-order", "--arccos-order")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("Arc-cosine order must come after -k arccos.");
        if(kernelTypes[kernelTypes.size()-1]!="arccos"
           && kernelTypes[kernelTypes.size()-1]!="arccosine")
          exitError("--arccos-order only valid for arccos / arccosine covariance.");
        arccosOrders[arccosOrders.size()-1]=getIntFromCurrentArgument();
      }
      else if (isCurrentArg("-g", "--gamma")) {
        incrementArgument(); 
        if(kernelTypes.size()==0)
          exitError("Inverse width specification must come after covariance function type is specified.");
        const string& kt = kernelTypes[kernelTypes.size()-1];
        if(kt!="rbf" && kt!="rbfard" && kt!="eq" && kt!="exp" && kt!="ratquad" && kt!="ratquadard"
           && kt!="matern32" && kt!="matern52" && kt!="matern32ard" && kt!="matern52ard"
           && kt!="stdperiodic" && kt!="cosine" && kt!="sinc" && kt!="expquadcosine")
          exitError("Inverse width / lengthscale (-g) only valid for RBF/EQ/exp/ratquad/Matérn/stdperiodic/cosine/sinc/expquadcosine (and ARD variants).");
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
      else if (isCurrentArg("-P", "--set")) {
        incrementArgument();
        setSpecs.push_back(getCurrentArgument());
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
        kernels[i]->setByPath("variance", variances[i]);
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
        kernels[i]->setByPath("weight_variance", weightVariances[i]);
      if(biasVariances[i]!=-1.0)
        kernels[i]->setByPath("bias_variance", biasVariances[i]);
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="rbf" || kernelTypes[i]=="rbfard") {
      if(selectInputs[i] || kernelTypes[i]=="rbfard")
        kernels.push_back(new CRbfardKern(*M));
      else
        kernels.push_back(new CRbfKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("inverseWidth", rbfInvWidths[i]);
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
	}
    else if(kernelTypes[i]=="eq" || kernelTypes[i]=="ExpQuad" || kernelTypes[i]=="expquad") {
      if(selectInputs[i])
        exitError("EQ/ExpQuad covariance does not map to ARD via -i; use -k rbfard.");
      kernels.push_back(new CEqKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("inverseWidth", rbfInvWidths[i]);
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="exp") {
      if(selectInputs[i])
        exitError("Exponential covariance function not available with input selection yet.");
      else
        kernels.push_back(new CExpKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("inverseWidth", rbfInvWidths[i]);
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
	}
    else if(kernelTypes[i]=="ratquad" || kernelTypes[i]=="ratquadard") {
      if(selectInputs[i] || kernelTypes[i]=="ratquadard")
        kernels.push_back(new CRatQuadardKern(*M));
      else
        kernels.push_back(new CRatQuadKern(*M));
      if(ratQuadAlphas[i]!=-1.0)
        kernels[i]->setByPath("power", ratQuadAlphas[i]);
	  if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("lengthscale", 1/sqrt(rbfInvWidths[i]));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="matern32" || kernelTypes[i]=="matern32ard") {
      if(selectInputs[i] || kernelTypes[i]=="matern32ard")
        kernels.push_back(new CMatern32ardKern(*M));
      else
        kernels.push_back(new CMatern32Kern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("lengthscale", 1.0/sqrt(rbfInvWidths[i]));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="matern52" || kernelTypes[i]=="matern52ard") {
      if(selectInputs[i] || kernelTypes[i]=="matern52ard")
        kernels.push_back(new CMatern52ardKern(*M));
      else
        kernels.push_back(new CMatern52Kern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("lengthscale", 1.0/sqrt(rbfInvWidths[i]));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="stdperiodic") {
      if(selectInputs[i])
        exitError("stdperiodic does not map to ARD via -i.");
      kernels.push_back(new CStdPeriodicKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("lengthscale", 1.0/sqrt(rbfInvWidths[i]));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="cosine") {
      if(selectInputs[i])
        exitError("cosine does not map to ARD via -i.");
      kernels.push_back(new CCosineKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("lengthscale", 1.0/sqrt(rbfInvWidths[i]));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="sinc") {
      if(selectInputs[i])
        exitError("sinc does not map to ARD via -i.");
      kernels.push_back(new CSincKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("lengthscale", 1.0/sqrt(rbfInvWidths[i]));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="expquadcosine" || kernelTypes[i]=="ExpQuadCosine") {
      if(selectInputs[i])
        exitError("expquadcosine does not map to ARD via -i.");
      kernels.push_back(new CExpQuadCosineKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setByPath("lengthscale", 1.0/sqrt(rbfInvWidths[i]));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="brownian") {
      if(selectInputs[i])
        exitError("brownian does not map to ARD via -i.");
      kernels.push_back(new CBrownianKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i]=="ibrownian" || kernelTypes[i]=="integratedbrownian") {
      if(selectInputs[i])
        exitError("ibrownian does not map to ARD via -i.");
      kernels.push_back(new CIntegratedBrownianKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i] == "mlp") {
      if(selectInputs[i])
        kernels.push_back(new CMlpardKern(*M));
      else
        kernels.push_back(new CMlpKern(*M));
      if(weightVariances[i]!=-1.0)
        kernels[i]->setByPath("weight_variance", weightVariances[i]);
      if(biasVariances[i]!=-1.0)
        kernels[i]->setByPath("bias_variance", biasVariances[i]);
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i] == "arccos" || kernelTypes[i] == "arccosine") {
      if(selectInputs[i])
        exitError("Arc-cosine covariance does not map to ARD via -i yet.");
      CArcCosineKern* ak = new CArcCosineKern(*M);
      if(arccosOrders[i] >= 0)
        ak->setOrder((unsigned int)arccosOrders[i]);
      kernels.push_back(ak);
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i] == "bias" && kernelUsageFlag[i]!=KERNEL_USAGE_FWD) {
      // Explicit -k bias only (CIP-0027: no longer auto-appended).
      kernels.push_back(new CBiasKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
    }
    else if(kernelTypes[i] == "white" && kernelUsageFlag[i]!=KERNEL_USAGE_FWD) {
      // Explicit -k white only (CIP-0027: observation noise is likelihood.variance).
      kernels.push_back(new CWhiteKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setByPath("variance", variances[i]);
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
  // CIP-0027 Phase B: do not auto-append Bias or White; FTC noise is likelihood.variance.
  if(kern.getNumKerns()==0)
  {
    CKern* defaultKern = new CRbfKern(X);
    kern.addKern(defaultKern);
  }

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
      // DTCVAR matches GPy VarDTC on the spherical Gaussian path (see
      // backlog/bugs/2026-11-15_dtcvar-numerical-diagnostic.md). K_uu / A may
      // still need jitter under poor conditioning, as with DTC.
    }
    else if(approxTypeStr=="fitc") {
      // FITC works on the spherical Gaussian path (testCGp / StableFactor / LQ).
      // Prefer -A dtcvar for tutorials; FITC remains available for parity/bench.
      approxType = CGp::FITC;
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
  if(sparseLinalgStr=="legacy")
    pmodel->setSparseLinalgMode(CGp::SparseLinalgLegacy);
  else if(sparseLinalgStr=="stable")
    pmodel->setSparseLinalgMode(CGp::SparseLinalgStableFactor);
  else if(sparseLinalgStr=="lq")
    pmodel->setSparseLinalgMode(CGp::SparseLinalgLQ);
  else
    exitError("Unknown --sparse-linalg mode (use legacy, stable, or lq): " + sparseLinalgStr);
  if(pmodel->isSparseApproximation())
  {
    try
    {
      pmodel->setInducingType(inducingTypeStr);
    }
    catch(ndlexceptions::Error& err)
    {
      exitError(err.getMessage());
    }
    if((inducingTypeStr == "whitenoise" || inducingTypeStr == "greens"
        || inducingTypeStr == "weightspace")
       && windowWidth > 0.0)
    {
      try
      {
        pmodel->getInducing()->setWindowWidth(windowWidth);
      }
      catch(ndlexceptions::Error& err)
      {
        exitError(err.getMessage());
      }
    }
  }
  if(optimiser=="scg")
    pmodel->setDefaultOptimiser(CGp::SCG);
  else if(optimiser=="conjgrad")
    pmodel->setDefaultOptimiser(CGp::CG);
  else if(optimiser=="graddesc")
    pmodel->setDefaultOptimiser(CGp::GD);
  else if(optimiser=="quasinew" || optimiser=="lbfgs")
    pmodel->setDefaultOptimiser(CGp::BFGS);
  else
    exitError("Unrecognised optimiser type: " + optimiser);
  pmodel->setBetaVal(1); // sparse: β=1 ⇒ likelihood.variance=1 (CIP-0027 sync)
  if(pmodel->getApproximationType()==CGp::FTC && pmodel->isGaussianNoise())
    pmodel->setLikelihoodVariance(1.0); // GPy-like default observation noise
  // CIP-0027 Phase C: explicit -k white with free likelihood.variance needs allow,
  // otherwise fix likelihood variance (legacy White-owned noise).
  if(allowWhiteAndNoise)
    pmodel->setAllowWhiteAndLikelihoodNoise(true);
  else if(pmodel->getApproximationType()==CGp::FTC && pmodel->isGaussianNoise())
  {
    try
    {
      (void)kern.getByPath("white.variance");
      pmodel->fixLikelihoodVariance(true);
      if(getVerbosity()>0)
        cout << "CIP-0027: White present — fixing likelihood.variance "
             << "(use --allow-white-and-noise to train both)." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      // no White component
    }
  }
  pmodel->setScale(scale);
  pmodel->setBias(bias);
  pmodel->updateM();
  pmodel->setOutputScaleLearnt(outputScaleLearnt);
  for(size_t si=0; si<setSpecs.size(); si++)
  {
    const string& spec=setSpecs[si];
    const size_t eq=spec.find('=');
    if(eq==string::npos || eq==0 || eq+1>=spec.size())
      exitError("--set expects path=value (e.g. rbf.variance=1.0), got: "+spec);
    const string path=spec.substr(0, eq);
    const double val=atof(spec.substr(eq+1).c_str());
    try
    {
      pmodel->setByPath(path, val);
    }
    catch(ndlexceptions::Error& err)
    {
      exitError(string("--set failed for '")+path+"': "+err.getMessage());
    }
  }
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
  else if(optimiser=="quasinew" || optimiser=="lbfgs")
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

// CIP-003B: parse "gamma:a,b" or "gaussian:precision" into a prior.
static CDist* parsePriorSpec(const string& spec)
{
  const size_t colon=spec.find(':');
  if(colon==string::npos)
    throw ndlexceptions::Error("prior spec must look like gamma:a,b or gaussian:precision, got: "+spec);
  const string kind=spec.substr(0, colon);
  vector<double> vals;
  string rest=spec.substr(colon+1);
  size_t pos=0;
  while(pos<=rest.size())
  {
    const size_t comma=rest.find(',', pos);
    const string tok=rest.substr(pos, comma==string::npos ? string::npos : comma-pos);
    if(tok.empty())
      throw ndlexceptions::Error("empty prior parameter in: "+spec);
    vals.push_back(atof(tok.c_str()));
    if(comma==string::npos)
      break;
    pos=comma+1;
  }
  if(kind=="gamma")
  {
    if(vals.size()!=2 || !(vals[0]>0.0) || !(vals[1]>0.0))
      throw ndlexceptions::Error("gamma prior needs positive shape and rate: gamma:a,b");
    CGammaDist* d=new CGammaDist();
    d->setParam(vals[0], 0);
    d->setParam(vals[1], 1);
    return d;
  }
  if(kind=="gaussian")
  {
    if(vals.size()!=1 || !(vals[0]>0.0))
      throw ndlexceptions::Error("gaussian prior needs a positive precision: gaussian:precision");
    CGaussianDist* d=new CGaussianDist();
    d->setParam(vals[0], 0);
    return d;
  }
  throw ndlexceptions::Error("unknown prior type '"+kind+"' (use gamma or gaussian)");
}

void CClgp::sample()
{
  incrementArgument();
  setMode("sample");
  int numSamples=1000;
  int burnIn=200;
  int leapfrog=10;
  double stepSize=0.1;
  bool jacobian=true;
  bool allowImproper=false;
  bool constrained=true;
  vector<string> priorSpecs;
  vector<string> fixPaths;
  string modelFileName="gp_model";
  string samplesFileName="gp_samples.csv";

  while(isFlags()) {
    if(isCurrentArgumentFlag()) {
      if(isCurrentArg("-?", "--?") || isCurrentArg("-h", "--help")) {
	helpInfo();
	exitNormal();
      }
      else if(isCurrentArg("-n", "--num-samples")) {
	incrementArgument();
	numSamples=getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-b", "--burn-in")) {
	incrementArgument();
	burnIn=getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-e", "--step-size")) {
	incrementArgument();
	stepSize=getDoubleFromCurrentArgument();
      }
      else if(isCurrentArg("-l", "--leapfrog")) {
	incrementArgument();
	leapfrog=getIntFromCurrentArgument();
      }
      else if(isCurrentArg("-p", "--prior")) {
	incrementArgument();
	priorSpecs.push_back(getCurrentArgument());
      }
      else if(isCurrentArg("-F", "--fix")) {
	incrementArgument();
	fixPaths.push_back(getCurrentArgument());
      }
      else if(isCurrentArg("-J", "--no-jacobian")) {
	jacobian=false;
      }
      else if(isCurrentArg("-I", "--allow-improper")) {
	allowImproper=true;
      }
      else if(isCurrentArg("-U", "--unconstrained")) {
	constrained=false;
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
  if(numSamples<1 || burnIn<0 || leapfrog<1 || !(stepSize>0.0))
    exitError("sample: need num-samples>=1, burn-in>=0, leapfrog>=1 and step-size>0.");

  string trainDataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc)
    modelFileName=argv[getCurrentArgumentNo()+1];
  if((getCurrentArgumentNo()+2)<argc)
    samplesFileName=argv[getCurrentArgumentNo()+2];

  CMatrix X;
  CMatrix y;
  readData(X, y, trainDataFileName);
  CGp* pmodel=readGpFromFile(modelFileName, getVerbosity());
  // setTargetVals also points the noise model at y; relearn only sets pmodel->py,
  // which leaves the Gaussian noise gradient reading a stale target (pre-existing
  // segfault in relearn with a free likelihood.variance).
  pmodel->setTargetVals(y);
  pmodel->updateM();
  pmodel->pX = &X;
  if(pmodel->getInputDim()!=X.getCols())
    throw ndlexceptions::Error(trainDataFileName + ": input data is not of correct dimension");

  // Priors are not stored for noise in the model stream, so they come from the command line.
  for(size_t i=0; i<priorSpecs.size(); i++)
  {
    const size_t eq=priorSpecs[i].find('=');
    if(eq==string::npos || eq==0 || eq+1>=priorSpecs[i].size())
      exitError("--prior expects path=gamma:a,b or path=gaussian:precision, got: "+priorSpecs[i]);
    pmodel->addPriorByPath(priorSpecs[i].substr(0, eq), parsePriorSpec(priorSpecs[i].substr(eq+1)));
  }
  for(size_t i=0; i<fixPaths.size(); i++)
    pmodel->fix(fixPaths[i]);
  pmodel->setIncludeJacobian(jacobian);
  pmodel->setAllowImproper(allowImproper);

  // CHmc refuses improper targets, naming the parameters that lack priors.
  CHmc hmc(*pmodel, *pmodel, getSeed());
  hmc.setStepSize(stepSize);
  hmc.setNumLeapfrog((unsigned int)leapfrog);
  const bool toStdout=(samplesFileName=="-");
  ostream& info=toStdout ? cerr : cout;
  if(burnIn>0)
    hmc.adaptStepSize((unsigned int)burnIn);
  CMatrix samples, logPost;
  hmc.sample((unsigned int)numSamples, samples, logPost);
  if(getVerbosity()>0)
    info << "HMC: " << numSamples << " samples, step size " << hmc.getStepSize()
	 << ", " << leapfrog << " leapfrog steps, acceptance rate "
	 << hmc.acceptRate() << (jacobian ? "" : " (no Jacobian)") << endl;
  if(toStdout)
    writeSamplesCsv(*pmodel, samples, logPost, cout, constrained);
  else
  {
    ofstream out(samplesFileName.c_str());
    if(!out)
      throw ndlexceptions::FileWriteError(samplesFileName);
    writeSamplesCsv(*pmodel, samples, logPost, out, constrained);
    if(getVerbosity()>0)
      info << "Samples written to " << samplesFileName << endl;
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
    outGnuplot << "set table '" << name << "_decision.dat'" << endl;
    outGnuplot << "splot \"" << name << "_prob_matrix.dat\"" << endl;
    outGnuplot << "set cntrparam levels discrete 0.25, 0.75" << endl;
    outGnuplot << "set table '" << name << "_contours.dat'" << endl;
    outGnuplot << "splot \"" << name << "_prob_matrix.dat\"" << endl;
    outGnuplot << "unset table" << endl;
    outGnuplot << "reset" << endl;
    writeGnuplotTermSelect(outGnuplot);
    writeGnuplotStyleHeader(outGnuplot);
    outGnuplot << "plot ";
    bool firstPlot = true;
    if(numPos>0) {
      outGnuplot << "\"" << name << "_positive.dat\" with points ls 5 ps " << pointSize << " title \"positive\"";
      firstPlot = false;
    }
    if(numNeg>0) {
      if(!firstPlot) outGnuplot << ", ";
      outGnuplot << "\"" << name << "_negative.dat\" with points ls 6 ps " << pointSize << " title \"negative\"";
      firstPlot = false;
    }
    if(!firstPlot) outGnuplot << ", ";
    outGnuplot << "\"" << name << "_active_set.dat\" with points ls 4 ps " << pointSize*2 << " title \"active\"";
    if(numUnlab>0)
      outGnuplot << ", \"" << name << "_unlabelled.dat\" with points ls 7 ps " << pointSize << " title \"unlabelled\"";
    outGnuplot << ", \"" << name << "_decision.dat\" with lines ls 8 lw " << lineWidth << " title \"decision\"";
    outGnuplot << ", \"" << name << "_contours.dat\" with lines ls 9 lw " << lineWidth << " title \"0.25/0.75\"" << endl;
    writeGnuplotFinish(outGnuplot);
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
      writeGnuplotTermSelect(outGnuplot);
      writeGnuplotStyleHeader(outGnuplot);
      outGnuplot << "set hidden3d" << endl;
      outGnuplot << "splot \"" << name << "_output_matrix.dat\" with lines ls 1 lw " << lineWidth << " title \"mean\"";
      outGnuplot << ", \"" << name << "_scatter_data.dat\" with points ls 3 ps " << pointSize << " title \"data\"";
      if(pmodel->isSparseApproximation())
      {
      	outGnuplot << ", \"" << name << "_active_set.dat\" with points ls 4 ps " << pointSize << " title \"inducing\"";
      }
      outGnuplot << endl;
      writeGnuplotFinish(outGnuplot);
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
      // Three columns: x, ymin (−2σ), ymax (+2σ) for filledcurves.
      string errorFile = name + "_error_bar_data.dat";
      ofstream out(errorFile.c_str());
      if(!out) throw ndlexceptions::FileWriteError(errorFile);
      out << setiosflags(ios::scientific);
      out << setprecision(17);
      out << "# x ymin ymax (±2σ) for filledcurves" << endl;
      for(int j=0; j<numx; j++) {
	out << errorBarMinus.getVal(j, 0) << " "
	    << errorBarMinus.getVal(j, 1) << " "
	    << errorBarPlus.getVal(j, 1) << endl;
      }
      out.close();
      string plotFileName = name + "_plot.gp";
      ofstream outGnuplot(plotFileName.c_str());
      if(!outGnuplot) throw ndlexceptions::FileWriteError(plotFileName);
      writeGnuplotTermSelect(outGnuplot);
      writeGnuplotStyleHeader(outGnuplot);
      outGnuplot << "plot \"" << name << "_error_bar_data.dat\" using 1:2:3 with filledcurves fc rgb \"#729fcf\" title \"±2σ\"";
      outGnuplot << ", \"" << name << "_line_data.dat\" with lines ls 1 lw " << lineWidth << " title \"mean\"";
      outGnuplot << ", \"" << name << "_scatter_data.dat\" with points ls 3 ps " << pointSize << " title \"data\"";
      if(pmodel->isSparseApproximation())
      {
        outGnuplot << ", \"" << name << "_active_set.dat\" with points ls 4 ps " << pointSize << " title \"inducing\"";
      }
      outGnuplot << endl;
      writeGnuplotFinish(outGnuplot);
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
    helpArgument("sample", "For drawing HMC samples of the hyperparameter posterior of a learned exact GP (CIP-003B).");
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
    helpArgument("example_file", "Training data: dense CSV/TSV (last column = y; CIP-0022) preferred; SVM-light (.svml) still supported. Extension selects the reader.");
    helpArgument("model_file", "File to store the resulting model in. By default this is set to gp_model.");
    cout << endl;
    cout << "Command Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-O, --optimiser string", "Optimiser to be used, valid options are scg, conjgrad, quasinew (or lbfgs) and graddesc. Default is scg.");
    helpArgument("-#, --#iterations int", "Optimiser budget (setMaxIters). For scg/conjgrad/graddesc: algorithm iterations. For lbfgs/quasinew: both reverse-comm steps and function evaluations (GPy-aligned). Default 1000. See docs/optimisers.md.");
    helpArgument("-x, --input-dim int", "Dimension for input space.");
    helpArgument("-C, --Centre-data bool", "Centre the data by removing the mean value. Default value is 1 (true).");
    helpArgument("-L, --Learn-scales bool", "Whether or not to learn the scaling on the input data. Default value is 0 (false).");
    helpArgument("-S, --Scale-data bool", "Scale the data by setting the standard deviation in each direction to 1.0. Default value is 0 (false).");
    helpArgument("-A, --Approximation-type string", "Sparse approximation type. Either FTC (no approximation), DTC (projected process approximation), DTCVAR (variational sparse approximation).");// FITC (fully independent training conditional --- not fully tested).");//, PITC (partially independent training conditional. Default value is FTC (no approximation).");
    helpArgument("-L, --sparse-linalg string", "Sparse inducing-side linear algebra: lq (LQ of [I,A], CIP-001B, default), stable (whitened C + Murray adjoints), or legacy (explicit invK_uu). Prefer --sparse-linalg over -L (short -L collides with Learn-scales).");
    helpArgument("--inducing points|whitenoise|greens|weightspace", "Inducing-variable family. Default points. whitenoise: CIP-001C kernel-free K_uu (Gaussian windows). greens: CIP-0029 1-D causal-root features. weightspace: CIP-0028 MLP/arccos weight-space features.");
    helpArgument("--window-width float", "Fixed window width s (with --inducing whitenoise, greens, or weightspace). Default: median nearest-neighbour spacing of inducing centres.");
    helpArgument("-P, --set path=value", "Set a model parameter by GPy-style path after the model is built (CIP-0014). Examples: rbf.variance=1.0, kern.rbf.lengthscale=2.0, white.variance=0.01, beta=10 (sparse). Repeatable.");
    //      helpArgument("-dh, --dynamics-hyperparameter double", "Specify weighting of the prior on the dynamics hyperparameters.");
    string m=getMode();
    setMode("kern");
    helpInfo();
    setMode(m);
  }
  else if(command=="kern") {
    cout << endl << "Covariance Function options:" << endl;
    helpArgument("-k, --kernel string", "Covariance type (repeatable compound). Learn path: lin, poly, rbf, rbfard, eq/ExpQuad, exp, ratquad, ratquadard, matern32, matern52, matern32ard, matern52ard, stdperiodic, cosine, sinc, expquadcosine, brownian, ibrownian, mlp, arccos (CIP-0028/0038). Default: rbf only (CIP-0027: Bias/White opt-in via -k bias / -k white; FTC noise is likelihood.variance). Multi-output ICM/LCM + indexwhite: API demo (docs/tutorials/coregionalization_toy.md), not -k. Prefer -A dtcvar over FITC for sparse tutorials.");
    helpArgument("--allow-white-and-noise", "CIP-0027 Phase C: allow trainable White and free likelihood.variance together on FTC. Default refuses dual free noise (legacy -k white fixes likelihood.variance).");
    helpArgument("--arccos-order int", "Arc-cosine order 0 (angular / infinite-MLP) or 1 (ReLU). Only with -k arccos. Default 0.");
    helpArgument("-g, --gamma float", "Lengthscale control: sets inverseWidth for RBF/EQ/exp, or lengthscale=1/sqrt(2γ) for ratquad/Matérn/stdperiodic/cosine (and ARD variants). Prefer -P …lengthscale=… when matching GPy.");
    helpArgument("-@, --alpha float", "Alpha / power parameter for rational quadratic (prefer -P RatQuad.power=…).");
    helpArgument("-v, --variance float", "Variance parameter for covariance function.");
    helpArgument("-w, --weight float", "Weight parameter for polynomial and MLP covariance function.");
    helpArgument("-b, --bias float", "Bias parameter for polynomial and MLP covariance function.");
    helpArgument("-d, --degree int", "Degree parameter for polynomial covariance function.");
    helpArgument("-P, --set path=value", "Also listed under learn: GPy-style path set after model construction (rbf.variance, Mat32.lengthscale, ExpQuad.lengthscale, beta, scale.j).");
  }    
  else if(command=="sample") {
    helpHeader();
    helpUsage("gp [options] sample example_file [model_file] [samples_file]");
    helpDescriptor("Hamiltonian Monte Carlo over the free parameters of a learned exact (FTC) GP, sampled in the optimiser's unconstrained space. Priors on kernel parameters stored in the model are used; priors on likelihood.variance (and any others) are given with --prior. Every sampled parameter needs a prior or a bounded transform, otherwise the posterior is improper and the command refuses to run.");
    cout << "Arguments:" << endl;
    helpArgument("example_file", "Training data (dense CSV/TSV or SVM-light), as for learn.");
    helpArgument("model_file", "Learned model. Default gp_model.");
    helpArgument("samples_file", "Output CSV (iteration, log_posterior, one column per parameter), or - for standard output. Default gp_samples.csv.");
    cout << "Command Options:" << endl;
    helpArgument("-?, -h, --help", "This help.");
    helpArgument("-n, --num-samples int", "Number of samples after burn-in. Default 1000.");
    helpArgument("-b, --burn-in int", "Transitions used to adapt the step size (dual averaging, target acceptance 0.8) and then discarded. Default 200; 0 keeps the step size fixed.");
    helpArgument("-e, --step-size float", "Initial leapfrog step size. Default 0.1.");
    helpArgument("-l, --leapfrog int", "Leapfrog steps per transition. Default 10.");
    helpArgument("-p, --prior path=spec", "Attach a prior to a kernel parameter or likelihood.variance, e.g. rbf.lengthscale=gamma:2,1 or likelihood.variance=gamma:2,3. spec is gamma:shape,rate or gaussian:precision. Repeatable.");
    helpArgument("-F, --fix path", "Hold a parameter fixed (same paths as --prior). Repeatable.");
    helpArgument("-J, --no-jacobian", "Omit the log-Jacobian of the constraining transforms (GPy-style: samples the transformed-space density). Default includes it, so priors mean what they say on the constrained value.");
    helpArgument("-I, --allow-improper", "Run even if some parameter has neither a prior nor a bounded transform.");
    helpArgument("-U, --unconstrained", "Write the optimiser's unconstrained values rather than constrained ones.");
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
    helpArgument("name", "The gnuplot script will be output as \"name_plot.gp\". Data files needed etc. will also be preceded by \"name\". Default is \"gp\". Uses qt by default; set GPLOT_OUT='file.pdf|.svg|.png' (or GPLOT_PDF / GPLOT_SVG / GPLOT_PNG) for file output.");
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
