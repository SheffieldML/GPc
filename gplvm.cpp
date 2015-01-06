
#include "gplvm.h"

int main(int argc, char* argv[])
{
  CClgplvm command(argc, argv);
  command.setFlags(true);
  command.setVerbosity(2);
  command.setMode("gplvm");
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
	else if(argument=="display") 
	  command.display();
	else if(argument=="gnuplot") 
	  command.gnuplot();
	else  
	  command.exitError("Invalid gplvm command provided.");
	command.incrementArgument();
      }
    command.exitError("No gplvm command provided.");
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
  catch(std::bad_alloc err)
  {
    command.exitError("Out of memory.");
  }
  catch(std::exception err)
  {
    command.exitError("Unhandled exception.");
  }
}
CClgplvm::CClgplvm(int arc, char** arv) : CClctrl(arc, arv)
{
}
void CClgplvm::learn()
{
  incrementArgument();
  setMode("learn");
  
  enum {
    KERNEL_USAGE_BACK,
    KERNEL_USAGE_FWD,
    KERNEL_USAGE_DYN
  };

  double tol=1e-6;
  vector<string> kernelTypes;
  vector<unsigned int> kernelUsageFlag;
  vector<double> rbfInvWidths;
  vector<double> weightVariances;
  vector<double> biasVariances;
  vector<double> variances;
  vector<double> degrees;
  vector<bool> selectInputs;
  string labelledIndicesFile;
  bool labelledIndicesFlag=false;
  string optimiser="scg";
  bool centreData=true;
  bool scaleData=false;
  bool regulariseLatent=true;
  bool inputScaleLearnt=false;
  string initialisationType="pca";
  bool labelsProvided = true;
  bool dynamicsUsed = false;
  bool wangWeighting = false;
  bool dynamicScaling = false;
  double dynamicsRatio = 20;
  double dynamicsScale = 0.5;
  double signalVariance = 0.0;
  double dynamicsMParameter = 1.0;
  vector<int> labels;
  int iters=1000;
  int latentDim=2;
  string modelFileName="gplvm_model";
  while(isFlags())
  {
    if(isCurrentArgumentFlag())
    {
      if (isCurrentArg("-?", "--?")) {
        helpInfo(); 
        exitNormal();
      }
      else if (isCurrentArg("-h", "--help")) {
        helpInfo(); 
        exitNormal();
      }
      else if (isCurrentArg("-x", "--latent-dim")) {
        incrementArgument(); 
        latentDim = getIntFromCurrentArgument();
      }
      
      else if (isCurrentArg("-c", "--constrained")) {
        incrementArgument(); 
        kernelTypes.push_back(getCurrentArgument()); 
        kernelUsageFlag.push_back(KERNEL_USAGE_BACK);
        rbfInvWidths.push_back(-1.0);
        weightVariances.push_back(-1.0);
        biasVariances.push_back(-1.0);
        variances.push_back(-1.0);
        degrees.push_back(-1.0);
        selectInputs.push_back(false);
      }
      else if (isCurrentArg("-D", "--dynamics-kernel")) {
        incrementArgument(); 
	dynamicsUsed = true;
        kernelTypes.push_back(getCurrentArgument());
        kernelUsageFlag.push_back(KERNEL_USAGE_DYN);
        rbfInvWidths.push_back(-1.0);
        weightVariances.push_back(-1.0);
        biasVariances.push_back(-1.0);
        variances.push_back(-1.0);
        degrees.push_back(-1.0);
        selectInputs.push_back(false);
      }
//       else if (isCurrentArg("-dh", "--dynamics-hyperparameter")) {
// 	incrementArgument();
// 	if(!dynamicsUsed)
// 	  exitError("You need to declare a dynamics kernel before setting the parameter M.");
// 	dynamicsMParameter = getDoubleFromCurrentArgument();
// 	if(dynamicsRatio!=-1.0)
// 	  exitError("It doesn't make sense to use Wang-weighting when the dynamics signal to noise ratio is fixed.");
// 	wangWeighting=true;
	
//       }
//       else if (isCurrentArg("-ds", "--dynamics-scaling")) {
// 	incrementArgument();
// 	if(!dynamicsUsed)
// 	  exitError("You need to declare a dynamics kernel before setting dynamics scaling.");
// 	if(dynamicsRatio!=-1.0)
// 	  exitError("It doesn't make sense to use log-likelihood scaling on the dynamics when the dynamics signal to noise ratio is fixed.");
// 	dynamicScaling = getBoolFromCurrentArgument();
//       }
      else if (isCurrentArg("-dr", "--dynamics-ratio")) {
	incrementArgument();
	if(!dynamicsUsed)
	  exitError("You need to declare a dynamics kernel before setting the dynamics signal to noise ratio. Default is 10.");
	dynamicsRatio = getDoubleFromCurrentArgument();
	if(wangWeighting)
	  exitError("It doesn't make sense to use Wang-weighting with the dynamics ratio paratmeter set.");
      }
      else if (isCurrentArg("-ds", "--dynamics-scale")) {
	incrementArgument();
	if(!dynamicsUsed)
	  exitError("You need to declare a dynamics kernel before setting the dynamics scale.");
	dynamicsScale = getDoubleFromCurrentArgument();
	if(wangWeighting)
	  exitError("It doesn't make sense to use Wang-weighting with the dynamics scale paratmeter set.");
      }
      else if (isCurrentArg("-C", "--Centre-data")) {
        incrementArgument(); 
        centreData=getBoolFromCurrentArgument();
      }
      else if (isCurrentArg("-I", "--Initialise")) {
        incrementArgument(); 
        initialisationType=getStringFromCurrentArgument();
      }
      else if (isCurrentArg("-L", "--Learn-scales")) {
        incrementArgument();
        inputScaleLearnt=getBoolFromCurrentArgument();
      }
      else if (isCurrentArg("-R", "--Regularise")) {
        incrementArgument();
        regulariseLatent=getBoolFromCurrentArgument();
      }
      else if (isCurrentArg("-S", "--Scale-data")) {
        incrementArgument(); 
        scaleData=getBoolFromCurrentArgument();
      }
      else if (isCurrentArg("-l", "--labelled-indices-file")) {
        incrementArgument();
        labelledIndicesFile=getCurrentArgument();
        labelledIndicesFlag = true;
      }
      else if (isCurrentArg("-O", "--optimiser")) {
	incrementArgument(); 
	optimiser=getCurrentArgument(); 
      }
      else if (isCurrentArg("-k", "--kernel")) {
        incrementArgument(); 
        kernelTypes.push_back(getCurrentArgument()); 
        kernelUsageFlag.push_back(KERNEL_USAGE_FWD);
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
          exitError("Inverse width specification must come after kernel type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="rbf")
          exitError("Inverse width parameter only valid for RBF kernel.");
        rbfInvWidths[rbfInvWidths.size()-1]=2*getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-d", "--degree")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("Polynomial degree specification must come after kernel type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="poly")
          exitError("Polynomial degree parameter only valid for poly kernel.");
        degrees[degrees.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-w", "--weight")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("`Weight variance' parameter specification must come after kernel type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="poly" 
           && kernelTypes[kernelTypes.size()-1]!="mlp")
          exitError("`Weight variance' parameter only valid for polynomial and MLP kernel.");
        weightVariances[weightVariances.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-b", "--bias")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("`Bias variance' parameter specification must come after kernel type is specified.");
        if(kernelTypes[kernelTypes.size()-1]!="poly" 
           && kernelTypes[kernelTypes.size()-1]!="mlp")
          exitError("`Bias variance' parameter only valid for polynomial and MLP kernel.");
        biasVariances[biasVariances.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-v", "--variance")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("Variance parameter specification must come after kernel type is specified.");
        variances[variances.size()-1]=getDoubleFromCurrentArgument(); 
      }
      else if (isCurrentArg("-i", "--input-select")) {
        incrementArgument();
        if(kernelTypes.size()==0)
          exitError("Input selection flag must come after kernel type is specified.");
        selectInputs[selectInputs.size()-1]=getBoolFromCurrentArgument();
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
  CMatrix Y;
  CMatrix labs;
  readData(Y, labs, trainDataFileName);

  vector<int> labelledIndices;
  if(labelledIndicesFlag)
  {
    ifstream in(labelledIndicesFile.c_str());
    string line;
    if(!in) throw ndlexceptions::FileReadError(labelledIndicesFile);
    while(getline(in, line))
    {
      int index = atol(line.c_str())-1;
      if(index<0 || index>labs.getRows()) throw ndlexceptions::FileFormatError(labelledIndicesFile);	  
      labelledIndices.push_back(index);
    }
    in.close();
  }
  
  if(labelledIndicesFlag)
  {
    CMatrix newlabs(labelledIndices.size(), labs.getCols());
    CMatrix newY(labelledIndices.size(), Y.getCols());
    for(int i=0; i<labelledIndices.size(); i++)
    {
      newlabs.copyRowRow(i, labs, labelledIndices[i]);
      newY.copyRowRow(i, Y, labelledIndices[i]);
    }
    labs = newlabs;
    Y = newY;
    if(getVerbosity()>0)
      cout << "Reduced data set ... contains " << labs.getRows() << " points." << endl;
    
    
  }  
  if(labs.getCols()>1)
  {
    cout << "Ignoring data labels." << endl;
    labelsProvided=false;
  }
  if(labelsProvided)
  {
    for(int i=0; i<labs.getRows(); i++)
    {
      double val = labs.getVal(i);
      int intVal = (int)val;
      if((val - (double)intVal)!=0)
      {
        cout << "Ignoring data labels." << endl;
        labelsProvided=false;
        break;
      }
      else
	labels.push_back(intVal);
    }
  }
  
  
  CMatrix X(Y.getRows(), latentDim);
  
  // create kernel.
  CCmpndKern kern(X);
  CCmpndKern backKern(X);
  CCmpndKern dynKern(X);
  vector<CKern*> kernels;
  for(int i=0; i<kernelTypes.size(); i++)
  {
    CMatrix *M = 0;
    if(kernelUsageFlag[i]!=KERNEL_USAGE_BACK)
      M = &X;
    else
      M = &Y;

    if(kernelTypes[i]=="lin")
    {
      if(selectInputs[i])
        kernels.push_back(new CLinardKern(*M));
      else
        kernels.push_back(new CLinKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 0); // set variance parameter as specified.
    }
    else if(kernelTypes[i]=="poly")
    {
      if(selectInputs[i])
      {
        kernels.push_back(new CPolyardKern(*M));
        if(degrees[i]!=-1.0)
          ((CPolyardKern*)kernels[i])->setDegree(degrees[i]);
      }
      else
      {
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
    else if(kernelTypes[i]=="rbf")
    {
      if(selectInputs[i])
        kernels.push_back(new CRbfardKern(*M));
      else
        kernels.push_back(new CRbfKern(*M));
      if(rbfInvWidths[i]!=-1.0)
        kernels[i]->setParam(rbfInvWidths[i], 0); /// set rbf inverse width as specified.
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 1); /// set variance parameter as specified.
    }
    else if(kernelTypes[i] == "mlp")
    {
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
    else if(kernelTypes[i] == "bias" && kernelUsageFlag[i]!=KERNEL_USAGE_FWD)
    {
      // fwd kernel always has bias component
      kernels.push_back(new CBiasKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 0); // set variance parameter as specified.
    }
    else if(kernelTypes[i] == "white" && kernelUsageFlag[i]!=KERNEL_USAGE_FWD)
    {
      // fwd kernel always includes a white noise component
      kernels.push_back(new CWhiteKern(*M));
      if(variances[i]!=-1.0)
        kernels[i]->setParam(variances[i], 0); // set variance parameter as specified.
    }
    else
    {
      exitError("Unknown kernel type: " + kernelTypes[i]);
    }
    switch (kernelUsageFlag[i]) 
    {
      case KERNEL_USAGE_FWD:
	if(wangWeighting)
	  addWangPrior(kernels[i], dynamicsMParameter);
        kern.addKern(kernels[i]);
        break;
      case KERNEL_USAGE_BACK:
        backKern.addKern(kernels[i]);
        break;
      case KERNEL_USAGE_DYN:
	if(wangWeighting)
	  addWangPrior(kernels[i], 1.0);
	signalVariance+=kernels[i]->getVariance();
        dynKern.addKern(kernels[i]);
        break;
    }
  }
  // if no kernel was specified, add an RBF.
  if(kern.getNumKerns()==0)
  {
    CKern* defaultKern = new CRbfKern(X);
    if(wangWeighting)
      addWangPrior(defaultKern, dynamicsMParameter);
    kern.addKern(defaultKern);
  }
  CKern* biasKern = new CBiasKern(X);
  if(wangWeighting)
    addWangPrior(biasKern, dynamicsMParameter);
  CKern* whiteKern = new CWhiteKern(X);
  if(wangWeighting)
    addWangPrior(whiteKern, dynamicsMParameter);
  kern.addKern(biasKern);
  kern.addKern(whiteKern);

      

  // if a dynamics kernel was specified add bias and white
  if(dynKern.getNumKerns()>0)
    {
      biasKern = new CBiasKern(X);
      if(wangWeighting)
	addWangPrior(biasKern, 1.0);
      whiteKern = new CWhiteKern(X);
      if(wangWeighting)
	addWangPrior(whiteKern, 1.0);
      dynKern.addKern(biasKern);
      signalVariance+=biasKern->getVariance();
      dynKern.setVariance(dynamicsScale*dynamicsScale);
      if(dynamicsRatio!=-1.0)
	whiteKern->setParam(dynamicsScale/(dynamicsRatio*dynamicsRatio), 0);
      dynKern.addKern(whiteKern);
    }
  
  CScaleNoise noise(&Y);
  
  // Remove scales and center if necessary.
  CMatrix params(1, 2*Y.getCols());
  noise.getParams(params);
  if(!centreData)
    for(int j=0; j<Y.getCols(); j++)
      params.setVal(0.0, j);
  if(!scaleData)
    for(int j=0; j<Y.getCols(); j++)
      params.setVal(1.0, j+Y.getCols());
  noise.setParams(params);

  CGplvm* pmodel;
  CMatrix bK(1,1,0.0);
  bool hasBack = (backKern.getNumKerns()!=0);
  bool hasDyn  = (dynKern.getNumKerns()!=0);
  if(!hasBack && !hasDyn) {
    pmodel = new CGplvm(&kern, &noise, latentDim, getVerbosity());
  }
  else if(!hasBack && hasDyn) {
    pmodel = new CGplvm(&kern, &dynKern, &noise, latentDim, getVerbosity());
  }
  else if(hasBack)
  {
    bK.resize(Y.getRows(), Y.getRows());
    backKern.compute(bK, Y);
    bK.setSymmetric(true);
    if (hasDyn) {
      pmodel = new CGplvm(&kern, &dynKern, &bK, &noise, latentDim, getVerbosity());
    } 
    else {
      pmodel = new CGplvm(&kern, &bK, &noise, latentDim, getVerbosity());
    }
  }
  else
  {
    exitError("Unsupported learning mode");
  }

  pmodel->setLatentRegularised(regulariseLatent);
  pmodel->setDynamicScaling(dynamicScaling);
  pmodel->setInputScaleLearnt(inputScaleLearnt);
  if(dynamicsRatio!=-1.0)
      pmodel->setDynamicKernelLearnt(false);
  if(initialisationType=="rand")
    pmodel->initXrand();
  else if(initialisationType=="pca")    
  {}// do nothing, this is normal.
  else
    exitError("Unknown initialisation type: " + initialisationType);

  cout << "Optimiser is " << optimiser;
  if(optimiser=="scg")
  {
    pmodel->setDefaultOptimiser(CGplvm::SCG);
  }
  else if(optimiser=="conjgrad")
  {
    pmodel->setDefaultOptimiser(CGplvm::CG);
  }
  else if(optimiser=="graddesc")
  {
    pmodel->setDefaultOptimiser(CGplvm::GD);
    pmodel->setLearnRate(1e-4);
    pmodel->setMomentum(0.9);
  }
  else if(optimiser=="quasinew")
  {
    pmodel->setDefaultOptimiser(CGplvm::BFGS);
  }
  else
  {
    exitError("Unrecognised model optimiser type.");
  }


  // Optimise the GP-LVM
  pmodel->optimise(iters);
  if(labelsProvided)
    // set labels after optimisation (to show they aren't being used!)
    pmodel->setLabels(labels);
  string comment="";
  switch(getFileFormat())
  {
  case 0: /// GPLVM file format.
    comment = "Run as:";
    for(int i=0; i<argc; i++)
	{
	  comment+=" ";
	  comment+=argv[i];
	}
    comment += " with seed " + ndlstrutil::itoa(getSeed()) + ".";
    writeGplvmToFile(*pmodel, modelFileName, comment);
    break;
  case 1: /// Matlab file format.
#ifdef _NDLMATLAB
    // Write matlab output.
    pmodel->writeMatlabFile(modelFileName, "gplvmInfo");
    pmodel->pkern->updateMatlabFile(modelFileName, "kern");
    X.updateMatlabFile(modelFileName, "X");
    Y.updateMatlabFile(modelFileName, "Y");
#else 
    exitError("Error MATLAB not incorporated at compile time.");
#endif
    break;
  default:
    exitError("Unrecognised file format number.");
      
  }
  exitNormal();
}

void CClgplvm::display()
{
  incrementArgument();
  setMode("display");
  while(isFlags())
  {
    if(isCurrentArgumentFlag())
    {
      if(getCurrentArgumentLength()!=2)
        unrecognisedFlag();
      if (isCurrentArg("-?", "--?") ||isCurrentArg("-h", "--help"))
      {
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
  if(getCurrentArgumentNo()>=argc)
    exitError("There are not enough input parameters.");
  string modelFileName=getCurrentArgument();
  CGplvm* pmodel=readGplvmFromFile(modelFileName, getVerbosity());
  pmodel->display(cout);
  exitNormal();
}

void CClgplvm::gnuplot()
{
  incrementArgument();
  setMode("gnuplot");
  double pointSize = 2;
  double lineWidth = 2;
  int resolution = 80;
  string name = "gplvm";
  string modelFileName="gplvm_model";
  string labelFileName="";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
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
  if(getCurrentArgumentNo()<argc) 
    modelFileName=argv[getCurrentArgumentNo()];
  if((getCurrentArgumentNo()+1)<argc) 
    name=argv[getCurrentArgumentNo()+1];
  string outputFileName=name+"_plot_data";
  if((getCurrentArgumentNo()+3)<argc) 
    outputFileName=argv[getCurrentArgumentNo()+2];
  CGplvm* pmodel=readGplvmFromFile(modelFileName, getVerbosity());
  if(pmodel->getLatentDim()!=2)
    {
      exitError("Plotting is only implemented for 2 dimensional latent spaces.");
    }

  // If there is a labels file prepare files separately.
  vector<int> labels;
  vector<bool> labelFileWritten;
  int maxVal=0;
  int minVal=0;
  if(labelFileName.length()>0)
    {
      ifstream in(labelFileName.c_str());
      string line;
      if(!in) throw ndlexceptions::FileReadError(labelFileName);
      while(getline(in, line))
	{
	  int index = atol(line.c_str());
	  if(index<0) throw ndlexceptions::FileFormatError(labelFileName);	  
	  labels.push_back(index);
	}
      in.close();
      if(labels.size() != pmodel->pX->getRows()) throw ndlexceptions::Error("Incorrect number of labels");
      pmodel->setLabels(labels);
    }
  if(pmodel->isLabels())
    {
      vector<int> storeIndex;
      int currentLabel = 0;
      maxVal = pmodel->getMaxLabelVal();
      minVal = pmodel->getMinLabelVal();
      for(int j=minVal; j<=maxVal; j++) 
	{
	  storeIndex.clear();
	  for(int i=0; i<pmodel->getNumData(); i++)
	    if(pmodel->getLabel(i)==j)
	      storeIndex.push_back(i);
	  if(storeIndex.size()>0)
	    {
	      // create a temporary storage matrix for writing.
	      CMatrix Xtemp(storeIndex.size(), pmodel->pX->getCols()+1, 0.1);
	      for(int i=0; i<storeIndex.size(); i++)
		for(int k=0; k<2; k++)
		  Xtemp.setVal(pmodel->pX->getVal(storeIndex[i], k), i, k);
	      Xtemp.toFile(name+"_latent_data" + ndlstrutil::itoa(j) + ".dat");
	      labelFileWritten.push_back(true);
	    }
	  else
	    labelFileWritten.push_back(false);
	  
	}
      
    }
  else
    {
      CMatrix Xext(pmodel->pX->getRows(), 3, 0.1);
      for(int i=0; i<pmodel->pX->getRows(); i++)
	{
	  for(int j=0; j<2; j++)
	    Xext.setVal(pmodel->pX->getVal(i, j), i, j);
	}
      Xext.toFile(name+"_latent_data.dat");
    }

  // Prepare the background variance plot.
  CMatrix minVals(1, pmodel->pX->getCols());
  CMatrix maxVals(1, pmodel->pX->getCols());
  pmodel->pX->maxRow(maxVals);
  pmodel->pX->minRow(minVals);
  int numx=resolution;
  int numy=resolution;
  double xspan=maxVals.getVal(0, 0)-minVals.getVal(0, 0);
  double xdiff=xspan/(numx-1);
  xdiff*=1.1;
  double yspan= maxVals.getVal(0, 1)-minVals.getVal(0, 1);
  double ydiff=yspan/(numy-1);
  ydiff*=1.1;
  CMatrix Xgrid(numx*numy, 2);
  vector<CMatrix*> matrixOut; 
  double x;
  double y;
  int i;
  int j;
  for(i=0, y=minVals.getVal(0, 1)-0.05*yspan; i<numy; y+=ydiff, i++)
    {
      matrixOut.push_back(new CMatrix(numx, 3));
      for(j=0, x=minVals.getVal(0, 0)-0.05*xspan; j<numx; x+=xdiff, j++)
	{
	  Xgrid.setVal(x, i*numy+j, 0);
	  matrixOut[i]->setVal(x, j, 0);
	  Xgrid.setVal(y, i*numy+j, 1);
	  matrixOut[i]->setVal(y, j, 1);
	}
    }
  CMatrix yTest(Xgrid.getRows(), pmodel->getNumProcesses());
  yTest.ones();
  CMatrix variances(yTest.getRows(), yTest.getCols());
  CMatrix means(yTest.getRows(), yTest.getCols());
  pmodel->posteriorMeanVar(means, variances, Xgrid);
  for(int i=0; i<numy; i++)
    for(int j=0; j<numx; j++)
      {
	matrixOut[i]->setVal(-log(variances.getVal(i*numy+j)), j, 2);
      }
  
  string matrixFile = name + "_variance_matrix.dat";
  ofstream out(matrixFile.c_str());
  if(!out) throw ndlexceptions::FileWriteError(matrixFile);
  out << setiosflags(ios::scientific);
  out << setprecision(17);
  out << "# Prepared plot of model file " << endl;
  for(int i=0; i<numy; i++)
    {
      matrixOut[i]->toStream(out);
      out << endl;
    }
  out.close();

  // prepare the plot file
  string plotFileName = name + "_plot.gp";
  ofstream outGnuplot(plotFileName.c_str());
  if(!outGnuplot) throw ndlexceptions::FileWriteError(plotFileName);
  outGnuplot << "set palette gray" << endl;
  outGnuplot << "set palette gamma 2.5" << endl;
  outGnuplot << "set pm3d map" << endl;
  outGnuplot << "set pm3d explicit" << endl;
  outGnuplot << "splot \"" << name << "_variance_matrix.dat\" with pm3d";
  if(pmodel->isLabels())
    {
      for(int i=minVal, j=0; i<=maxVal; i++, j++)
	if(labelFileWritten[j])
	  outGnuplot << ", \"" << name << "_latent_data" << i << ".dat\" with points ps " << pointSize;
    }
  else
    outGnuplot << ", \"" << name << "_latent_data.dat\"  with points ps " << pointSize;
  outGnuplot << endl;
  outGnuplot << "pause -1";
  outGnuplot.close();
  exitNormal();
}
void CClgplvm::helpInfo()
{
  string command = getMode();
  if(command=="gplvm")
    {
      helpHeader();
      helpUsage("gplvm [options] command [command-options]");
      cout << "Commands:" << endl;
      helpArgument("learn", "For learning the GPLVM model.");
      helpArgument("display", "For displaying the parameters of a learned GPLVM model.");
      helpArgument("gnuplot", "For plotting the results in gnuplot.");
      helpDescriptor("For help on a specific command type gplvm command --h");
      cout << endl;
      cout << "Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      helpArgument("-v, --verbosity [0..3]", "Verbosity level (default is set to 2).");      
      helpArgument("-s, --seed long", "Set random seed (default is to use current time in seconds).");      
    }
  else if(command=="learn")
    {
      helpHeader();
      helpUsage("gplvm [options] learn example_file [model_file]");
      helpDescriptor("This file is for learning a data set with an GPLVM. By default 1000 iterations of scaled conjugate gradient are used.");
      cout << "Arguments:" << endl;
      helpArgument("example_file", "File with the training data. The data file is in the format used by SVM light.");
      helpArgument("model_file", "File to store the resulting model in. By default this is set to gplvm_model.");
      cout << endl;
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      helpArgument("-l, --labelled-indices-file string", "File containing indices of labelled points. The index starts from 1.");
      helpArgument("-#, --#iterations int", "Number of iterations for optimisation by scaled conjugate gradient. Default is 1000.");
      helpArgument("-O, --optimiser string", "Optimiser to be used, valid options are scg, conjgrad, quasinew and graddesc. Default is scg.");
      helpArgument("-x, --latent-dim int", "Dimension for latent variable space. Deafult is 2.");
      helpArgument("-C, --Centre-data bool", "Centre the data by removing the mean value. Default value is 1 (true).");
      helpArgument("-I, --Initialise string", "How to initialise the latent positions. Options are pca (principal component analysis) or rand (small values drawn from a normal distribution). Default setting is pca.");
      helpArgument("-L, --Learn-scales bool", "Whether or not to learn the scaling on the input data. Default value is 0 (false).");
      helpArgument("-R, --Regularise bool", "Whether or not to regularise the latent space by using an L2 penalty (equivalent to the MAP solution for a Gaussian prior) over the latent space. The default value is 1 (true).");
      helpArgument("-S, --Scale-data bool", "Scale the data by setting the standard deviation in each direction to 1.0. Default value is 0 (false).");
      helpArgument("-c, --constrained string", "Use a back constraint kernel matrix with the given kernel type (see --kernel for options). Default is no constraint.");
      helpArgument("-D, --dynamics-kernel string", "Specify kernel for dynamics in the latent space. Default value is no dynamics kernel and therefore no learning of dynamics.");
      //      helpArgument("-dh, --dynamics-hyperparameter double", "Specify weighting of the prior on the dynamics hyperparameters.");
      helpArgument("-ds, --dynamics-scale true", "Specify standard deviations of the move in latent space associated with dynamics, default is 0.1.");
      helpArgument("-dr, --dynamics-ratio double", "Ratio between signal and noise in the dynamics kernel.");
      string m=getMode();
      setMode("kern");
      helpInfo();
      setMode(m);
    }
  else if(command=="kern")
    {
      cout << endl << "Kernel options:" << endl;
      helpArgument("-k, --kernel string", "Type of kernel function. Currently available options are lin (linear), poly (polynomial -- not recommended), rbf (radial basis function) and mlp (multi-layer perceptron otherwise known as arcsin). If the kernel is not specified it defaults to rbf.");
      helpArgument("-g, --gamma float", "Inverse width parameter in RBF kernel.");
      helpArgument("-v, --variance float", "Variance parameter for kernel.");
      helpArgument("-w, --weight float", "Weight parameter for polynomial and MLP kernel.");
      helpArgument("-b, --bias float", "Bias parameter for polynomial and MLP kernel.");
      helpArgument("-d, --degree int", "Degree parameter for polynomial kernel.");
    }    
  else if(command=="display")
    {
      helpHeader();
      helpUsage("gplvm [options] display [model_file]");
      helpDescriptor("Summarise the contents of a model file for quick viewing. The model is loaded in and the parameters of the GPLVM are given.");
      cout << "Arguments:" << endl;
      helpArgument("model_file", "File containing the model to be tested. By default this is set to gplvm_model.");
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
    }
  else if(command=="gnuplot")
    {
      helpHeader();
      helpUsage("gplvm [options] gnuplot [model_file] [name]");
      helpDescriptor("This command creates files for displaying the visualisation in gnuplot.");
      cout << "Arguments:" << endl;
      helpArgument("model_file", "The GPLVM model you want to use for generating the plot. The default is gplvm_model");
      helpArgument("name", "The gnuplot script will be output as \"name_plot.gp\". Data files needed etc. will also be preceded by \"name\". Default is \"gplvm\". Requires gnuplot vs 4.0 or later.");
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      helpArgument("-l, --labels string", "Label file name for visualisation of the data. If you wish to add labels to the data you can specify a file containing a vector of those labels here. By default any labels in the original data file are used.");
      helpArgument("-p, --point-size float", "Size of the point markers in gnuplot.");
      helpArgument("-r, --resolution int", "Resolution of the mesh grid in 3-d plots. The grid will have the given number of points in each direction. Default is 80.");
      
    }
}  

void CClgplvm::helpHeader()
{
  cout << endl << "GPLVM Code: Version " << GPLVMVERSION  << endl;
}

void addWangPrior(CKern* kern, double parameter)
{
  CWangDist* weightingPrior = new CWangDist();
  weightingPrior->setParam(parameter, 0);
  for(int j=0; j<kern->getNumParams(); j++)
    kern->addPrior(weightingPrior, j);
}
