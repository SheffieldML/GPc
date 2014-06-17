#include "ppa.h"

int main(int argc, char* argv[])
{
  CClppa command(argc, argv);
  command.setFlags(true);
  command.setVerbosity(2);
  command.setMode("ppa");
  try {    
    while(command.isFlags())
      {
	string argument = command.getCurrentArgument();
	if(argv[command.getCurrentArgumentNo()][0]=='-')
	  {
	    int j=1;
	    if(argv[command.getCurrentArgumentNo()][j]=='-')
	      j++;
	    else if(command.getCurrentArgumentLength()!=2)
	      command.unrecognisedFlag();	    
	    switch (argv[command.getCurrentArgumentNo()][j]) 
	      { 
	      case '?': 
		if(j==2)
		  command.confirmCurrentArg("--?");
		command.helpInfo(); 
		exit(0);
	      case 'h': 
		if(j==2)
		  command.confirmCurrentArg("--help");
		command.helpInfo(); 
		exit(0);
	      case 'v': 
		if(j==2)
		  command.confirmCurrentArg("--verbosity");
		command.incrementArgument(); 
		command.setVerbosity(command.getIntFromCurrentArgument()); 
		break;
	      default:
		command.unrecognisedFlag();
	      }
	  }
	else if(argument=="learn") // learning a model.
	  command.learn();	  
	else if(argument=="relearn") // initialise with an old model.
	  command.relearn();	  
	else if(argument=="test") // test with a model.
	  command.test();	  
	else if(argument=="log-likelihood") // compute model log likelihood.
	  command.logLikelihood();	  
	else if(argument=="predict") 
	  command.predict();
	else if(argument=="class-one-probabilities") 
	  command.classOneProbabilities();
	else if(argument=="display") 
	  command.display();
	else if(argument=="gnuplot") 
	  command.gnuplot();
	else  
	  command.exitError("Invalid ppa command provided.");
	command.incrementArgument();
      }
    command.exitError("No ppa command provided.");
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
CClppa::CClppa(int arc, char** arv) : CClctrl(arc, arv)
{
}

void CClppa::relearn()
{
  incrementArgument();
  setMode("relearn");
  string labelledIndicesFile="";
  bool labelledIndicesFlag = false;

  int kernIters=100;
  int noiseIters=20;
  int extIters=4;
  string modelFileName="ppa_model";
  string newModelFileName="ppa_model";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();

	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    case 'l':
	      if(j==2)
		confirmCurrentArg("--labelled-indices-file");
	      incrementArgument();
	      labelledIndicesFile=getCurrentArgument();
	      labelledIndicesFlag = true;
	      break;
	    case '#':
	      if(j==2)
		confirmCurrentArg("--#kernel-iterations");
	      incrementArgument();
	      kernIters=getIntFromCurrentArgument();
	      break;
	    case 'n':
	      if(j==2)
		confirmCurrentArg("--noise-iterations");
	      incrementArgument();
	      noiseIters=getIntFromCurrentArgument();
	      break;
	    case 'e':
	      if(j==2)
		confirmCurrentArg("--external-iterations");
	      incrementArgument();
	      extIters=getIntFromCurrentArgument();
	      break;
	    default: 
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
  if((getCurrentArgumentNo()+2)<argc) 
    newModelFileName=argv[getCurrentArgumentNo()+2];
  CMatrix X;
  CMatrix y;
  readData(X, y, trainDataFileName);
  vector<int> labelledIndices;
  if(labelledIndicesFlag)
    {
      ifstream in(labelledIndicesFile.c_str());
      string line;
      if(!in) throw ndlexceptions::FileReadError(labelledIndicesFile);
      while(getline(in, line))
	{
	  int index = atol(line.c_str())-1;
	  if(index<0 || index>y.getRows()) throw ndlexceptions::FileFormatError(labelledIndicesFile);	  
	  labelledIndices.push_back(index);
	}
      in.close();
    }
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  if(pmodel->getNumInputs()!=X.getCols())
    throw ndlexceptions::Error(trainDataFileName + ": input data is not of correct dimension");  
  
  if(labelledIndicesFlag)
    if(pmodel->noise.getType()!="ncnm")
      {
	CMatrix newy(labelledIndices.size(), y.getCols());
	CMatrix newX(labelledIndices.size(), X.getCols());
	for(int i=0; i<labelledIndices.size(); i++)
	  {
	    newy.copyRowRow(i, y, labelledIndices[i]);
	    newX.copyRowRow(i, X, labelledIndices[i]);
	  }
	y = newy;
	X = newX;
	if(getVerbosity()>0)
	  cout << "Reduced data set ... contains " << y.getRows() << " points." << endl;
	  
	
      }
    else
      {
	int count=0;
	for(int i=0; i<y.getRows(); i++)
	  {
	    vector<int>::iterator pos1 = find(labelledIndices.begin(), labelledIndices.end(), i);
	    if(pos1==labelledIndices.end())
	      {
		count++;
		for(int j=0; j<y.getCols(); j++)		  
		  y.setVal(0, i, j);
	      }
	  }
	if(getVerbosity()>0)
	  cout << "Removed labels from " << count << " points that weren't indexed." << endl;
      }  
  int numNoiseParams = pmodel->noise.getNumParams();
  CMatrix noiseParams(1, numNoiseParams);
  pmodel->noise.getParams(noiseParams);
  pmodel->noise.setTarget(y);
  pmodel->noise.setParams(noiseParams);
  CPpa model(X, y, pmodel->kern, pmodel->noise, getVerbosity());
  model.optimise(extIters, kernIters, noiseIters);
  string comment="";
  switch(getFileFormat())
    {
    case 0: /// PPA file format.
      comment = "Run as:";
      for(int i=0; i<argc; i++)
	{
	  comment+=" ";
	  comment+=argv[i];
	}
      comment += " with seed " + ndlstrutil::itoa(getSeed()) + ".";
      writePpaToFile(model, newModelFileName, comment);
      break;
    case 1: /// Matlab file format.
#ifdef _NDLMATLAB
      // Write matlab output.
      model.writeMatlabFile(newModelFileName, "ppaInfo");
      model.kern.updateMatlabFile(newModelFileName, "kern");
      model.noise.updateMatlabFile(newModelFileName, "noise");
      X.updateMatlabFile(newModelFileName, "X");
      y.updateMatlabFile(newModelFileName, "y");
#else 
      exitError("Error MATLAB not incorporated at compile time.");
#endif
      break;
    default:
      exitError("Unrecognised file format number.");
      
    }

  exit(0);
}
void CClppa::learn()
{
  incrementArgument();
  setMode("learn");
  string type="classification";
  string labelledIndicesFile="";
  bool labelledIndicesFlag = false;
  double tol=1e-6;
  vector<string> kernelTypes;
  vector<double> rbfInvWidths;
  vector<double> weightVariances;
  vector<double> biasVariances;
  vector<double> variances;
  vector<double> degrees;
  vector<bool> selectInputs;
  char noiseType;
  int kernIters=100;
  int noiseIters=20;
  int extIters=4;
  string modelFileName="ppa_model";

  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();
	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    case 'l':
	      if(j==2)
		confirmCurrentArg("--labelled-indices-file");
	      incrementArgument();
	      labelledIndicesFile=getCurrentArgument();
	      labelledIndicesFlag = true;
	      break;
	    case 'o': 
	      if(j==2)
		confirmCurrentArg("--output-type");
	      incrementArgument(); 
	      type = getCurrentArgument(); 
	      break;
	    case 'k': 
	      if(j==2)
		confirmCurrentArg("--kernel");
	      incrementArgument(); 
	      kernelTypes.push_back(getCurrentArgument()); 
	      rbfInvWidths.push_back(-1.0);
	      weightVariances.push_back(-1.0);
	      biasVariances.push_back(-1.0);
	      variances.push_back(-1.0);
	      degrees.push_back(-1.0);
	      selectInputs.push_back(false);
	      break;
	    case 'g': 
	      if(j==2)
		confirmCurrentArg("--gamma");
	      incrementArgument(); 
	      if(kernelTypes.size()==0)
		exitError("Inverse width specification must come after kernel type is specified.");
	      if(kernelTypes[kernelTypes.size()-1]!="rbf")
		exitError("Inverse width parameter only valid for RBF kernel.");
	      rbfInvWidths[rbfInvWidths.size()-1]=2*getDoubleFromCurrentArgument(); 
		      break;
	    case 'd':
	      if(j==2)
		confirmCurrentArg("--degree");
	      incrementArgument();
	      if(kernelTypes.size()==0)
		exitError("Polynomial degree specification must come after kernel type is specified.");
	      if(kernelTypes[kernelTypes.size()-1]!="poly")
		exitError("Polynomial degree parameter only valid for poly kernel.");
	      degrees[degrees.size()-1]=getDoubleFromCurrentArgument(); 
	      break;
	    case 'w':
	      if(j==2)
		confirmCurrentArg("--weight");
	      incrementArgument();
	      if(kernelTypes.size()==0)
		exitError("`Weight variance' parameter specification must come after kernel type is specified.");
	      if(kernelTypes[kernelTypes.size()-1]!="poly" 
		 && kernelTypes[kernelTypes.size()-1]!="mlp")
		exitError("`Weight variance' parameter only valid for polynomial and MLP kernel.");
	      weightVariances[weightVariances.size()-1]=getDoubleFromCurrentArgument(); 
	      break;
	    case 'b':
	      if(j==2)
		confirmCurrentArg("--bias");
	      incrementArgument();
	      if(kernelTypes.size()==0)
		exitError("`Bias variance' parameter specification must come after kernel type is specified.");
	      if(kernelTypes[kernelTypes.size()-1]!="poly" 
		 && kernelTypes[kernelTypes.size()-1]!="mlp")
		exitError("`Bias variance' parameter only valid for polynomial and MLP kernel.");
	      biasVariances[biasVariances.size()-1]=getDoubleFromCurrentArgument(); 
	      break;
	    case 'v':
	      if(j==2)
		confirmCurrentArg("--variance");
	      incrementArgument();
	      if(kernelTypes.size()==0)
		exitError("Variance parameter specification must come after kernel type is specified.");
	      variances[variances.size()-1]=getDoubleFromCurrentArgument(); 
	      break;
	    case 'i':
	      if(j==2)
		confirmCurrentArg("--input-select");
	      incrementArgument();
	      if(kernelTypes.size()==0)
		exitError("Input selection flag must come after kernel type is specified.");
	      try
		{
		  selectInputs[selectInputs.size()-1]=getBoolFromCurrentArgument();
		}
	      catch(ndlexceptions::CommandLineError err)
		{
		  exitError("Input selection flag must be either 1, 0, true or false.");
		}
		  
	      break;
	    case '#':
	      if(j==2)
		confirmCurrentArg("--#kernel-iterations");
	      incrementArgument();
	      kernIters=getIntFromCurrentArgument();
	      break;
	    case 'n':
	      if(j==2)
		confirmCurrentArg("--noise-iterations");
	      incrementArgument();
	      noiseIters=getIntFromCurrentArgument();
	      break;
	    case 'e':
	      if(j==2)
		confirmCurrentArg("--external-iterations");
	      incrementArgument();
	      extIters=getIntFromCurrentArgument();
	      break;
	    case 'f':
	      if(j==2)
		confirmCurrentArg("--file-format");
	      incrementArgument();
	      setFileFormat(getIntFromCurrentArgument());
	      break;
	    default: 
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
  if(type=="classification")
    noiseType = 'c';
  else if(type=="regression") 
    noiseType = 'r';
  else if(type=="ncnm")
    noiseType = 'n';
  else 
    exitError("Unknown output type, valid types are 'classification', 'regression' and 'ncnm' (null category noise model).");
  string dataFile="";
  CMatrix X;
  CMatrix y;
  readData(X, y, trainDataFileName);
  vector<int> labelledIndices;
  if(labelledIndicesFlag)
    {
      ifstream in(labelledIndicesFile.c_str());
      string line;
      if(!in) throw ndlexceptions::FileReadError(labelledIndicesFile);
      while(getline(in, line))
	{
	  int index = atol(line.c_str())-1;
	  if(index<0 || index>y.getRows()) throw ndlexceptions::FileFormatError(labelledIndicesFile);	  
	  labelledIndices.push_back(index);
	}
      in.close();
    }
  
  // prior for use with ncnm.
  CDist* prior = new CGammaDist();
  prior->setParam(1.0, 0);
  prior->setParam(1.0, 1);
  
  // create noise model.
  CNoise* noise;
  bool missingData=false;
  double yVal=0.0;
  switch(noiseType)
    {
    case 'n': /// set up null category noise model.
      missingData=true;
      for(int i=0; i<y.getRows(); i++)
	{
	  yVal=y.getVal(i);
	  if(yVal!=1.0 && yVal!=-1.0 && yVal!=0.0 && !isnan(yVal))
	    exitError("Input data is not a classification data set. Labels must either be -1.0, 1.0 or (for unlabelled) 0.0");
	}
      noise = new CNcnmNoise;
      break;
    case 'c': /// Set up a classification model.
      for(int i=0; i<y.getRows(); i++)
	{
	  yVal=y.getVal(i);
	  if(yVal!=1.0 && yVal!=-1.0)
	    {
	      if(yVal==0.0 || isnan(yVal))
		{
		  if(missingData)
		    continue;
		  else
		    {
		      if(getVerbosity()>0)
			cout << "Some data are missing labels, using null category noise model." << endl;
		      missingData=true;
		    }
		}
	      else
		exitError("Input data is not a classification data set. Labels must either be -1.0, 1.0 or (for unlabelled) 0.0");
	    }
	}
      if(missingData)
	{
	  // set up ncnm noise model.
	  noise = new CNcnmNoise;
	  noiseType='n';
	}
      else
	{
	  // set up probit noise model.
	  noise = new CProbitNoise;
	}
      break;
    case 'r':
      noise = new CGaussianNoise;      
      // set up a Gaussian noise model.
      break;
    otherwise:
      // we should never get here ... exit with error.
      exitError("Critical failure");
    }
  
  if(labelledIndicesFlag)
    if(noiseType=='c' || noiseType=='r')
      {
	CMatrix newy(labelledIndices.size(), y.getCols());
	CMatrix newX(labelledIndices.size(), X.getCols());
	for(int i=0; i<labelledIndices.size(); i++)
	  {
	    newy.copyRowRow(i, y, labelledIndices[i]);
	    newX.copyRowRow(i, X, labelledIndices[i]);
	  }
	y = newy;
	X = newX;
	if(getVerbosity()>0)
	  cout << "Reduced data set ... contains " << y.getRows() << " points." << endl;
	  
	
      }
    else if(noiseType=='n')
      {
	int count=0;
	for(int i=0; i<y.getRows(); i++)
	  {
	    vector<int>::iterator pos1 = find(labelledIndices.begin(), labelledIndices.end(), i);
	    if(pos1==labelledIndices.end())
	      {		
		count++;
		for(int j=0; j<y.getCols(); j++)
		  y.setVal(0, i, j);
	      }
	  }
	if(getVerbosity()>0)
	  cout << "Removed labels from " << count << " points that weren't indexed." << endl;	
      }
    else 
      // shouldn't be here
      exitError("Critical failure.");
  
  
	
  

  // create kernel.
  CCmpndKern kern(X);
  vector<CKern*> kernels;
  for(int i=0; i<kernelTypes.size(); i++)
    {
      if(kernelTypes[i]=="lin")
	{
	  if(selectInputs[i])
	    kernels.push_back(new CLinardKern(X));
	  else
	    kernels.push_back(new CLinKern(X));
	  if(variances[i]!=-1.0)
	    kernels[i]->setParam(variances[i], 0); // set variance parameter as specified.
	  if(missingData)
	    kernels[i]->addPrior(prior, 0); // place L1 regulariser on variance for ncnm model.
	}
      else if(kernelTypes[i]=="poly")
	{
	  if(selectInputs[i])
	    {
	      kernels.push_back(new CPolyardKern(X));
	      if(degrees[i]!=-1.0)
		((CPolyardKern*)kernels[i])->setDegree(degrees[i]);
	    }
	  else
	    {
	      kernels.push_back(new CPolyKern(X));	
	      if(degrees[i]!=-1.0)
		((CPolyKern*)kernels[i])->setDegree(degrees[i]);
	    }
	  if(weightVariances[i]!=-1.0)
	    kernels[i]->setParam(weightVariances[i], 0); // set `weight variance' parameter as specified.
	  if(biasVariances[i]!=-1.0)
	    kernels[i]->setParam(biasVariances[i], 1); // set `bias variance' parameter as specified.
	  if(variances[i]!=-1.0)
	    kernels[i]->setParam(variances[i], 2); // set variance parameter as specified.
	  if(missingData)	
	    kernels[i]->addPrior(prior, 2); // place L1 regulariser on variance for ncnm model.
	}
      else if(kernelTypes[i]=="rbf")
	{
	  if(selectInputs[i])
	    kernels.push_back(new CRbfardKern(X));
	  else
	    kernels.push_back(new CRbfKern(X));
	  if(rbfInvWidths[i]!=-1.0)
	    kernels[i]->setParam(rbfInvWidths[i], 0); /// set rbf inverse width as specified.
	  if(variances[i]!=-1.0)
	    kernels[i]->setParam(variances[i], 1); /// set variance parameter as specified.
	  if(missingData)	
	    kernels[i]->addPrior(prior, 1);// place L1 regulariser on variance for ncnm model.
	}
      else if(kernelTypes[i] == "mlp")
	{
	  if(selectInputs[i])
	    kernels.push_back(new CMlpardKern(X));
	  else
	    kernels.push_back(new CMlpKern(X));
	  if(weightVariances[i]!=-1.0)
	    kernels[i]->setParam(weightVariances[i], 0); // set `weight variance' parameter as specified.
	  if(biasVariances[i]!=-1.0)
	    kernels[i]->setParam(biasVariances[i], 1); // set `bias variance' parameter as specified.
	  if(variances[i]!=-1.0)
	    kernels[i]->setParam(variances[i], 2); // set variance parameter as specified.
	  if(missingData)	
	    kernels[i]->addPrior(prior, 2);// place L1 regulariser on variance for ncnm model.
	}
      else
	{
	  exitError("Unknown kernel type: " + kernelTypes[i]);
	}
      
      kern.addKern(kernels[i]);
    }
  // if no kernel was specified, add a linear one.
  if(kernels.size()==0)
    {
      CKern* defaultKern = new CLinKern(X);
      if(missingData)
	defaultKern->addPrior(prior, 0); // place L1 regulariser on variance for ncnm model.
      kern.addKern(defaultKern);
    }
  CKern* biasKern = new CBiasKern(X);
  if(missingData)
    biasKern->addPrior(prior, 0); // place L1 regulariser on variance for ncnm model.
  //biasKern->setParam(biasKernVariance, 0);

  CKern* whiteKern = new CWhiteKern(X);
  if(missingData)
    whiteKern->addPrior(prior, 0); // place L1 regulariser on variance for ncnm model.
  //biasKern->setParam(whiteKernVariance, 0);
  
  kern.addKern(biasKern);

  kern.addKern(whiteKern);
  
  noise->setTarget(y);
  noise->initParams();
  CPpa model(X, y, kern, *noise, getVerbosity());
  model.optimise(extIters, kernIters, noiseIters);
  string comment="";
  switch(getFileFormat())
    {
    case 0: /// PPA file format.
      comment = "Run as: ";
      for(int i=0; i<argc; i++)
	{
	  comment+=argv[i];
	  comment+=" ";
	}
      writePpaToFile(model, modelFileName, comment);
      break;
    case 1: /// Matlab file format.
#ifdef _NDLMATLAB
      // Write matlab output.
      model.writeMatlabFile(modelFileName, "ppaInfo");
      model.kern.updateMatlabFile(modelFileName, "kern");
      model.noise.updateMatlabFile(modelFileName, "noise");
      X.updateMatlabFile(modelFileName, "X");
      y.updateMatlabFile(modelFileName, "y");
#else 
      exitError("Error MATLAB not incorporated at compile time.");
#endif
      break;
    default:
      exitError("Unrecognised file format number.");
      
    }
  exit(0);
}

void CClppa::test()
{
  incrementArgument();
  setMode("test");
  string modelFileName="ppa_model";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();
	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    default: 
	      unrecognisedFlag();
	    }
	  incrementArgument();
	}
      else
	setFlags(false);
    }
  if(getCurrentArgumentNo()>=argc)
    exitError("There are not enough input parameters.");
  string testDataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc) 
    modelFileName=argv[getCurrentArgumentNo()+1];
  string dataFile="";
  CMatrix X;
  CMatrix y;
  readData(X, y, testDataFileName);
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  if(getVerbosity()>2)
    pmodel->display(cout);
  if(pmodel->getNumProcesses()!=y.getCols())
    throw ndlexceptions::Error(testDataFileName + ": input targets are not of correct dimension");
  if(pmodel->getNumInputs()!=X.getCols())
    throw ndlexceptions::Error(testDataFileName + ": input data is not of correct dimension");
  
  pmodel->test(y, X);
  exit(0);
}
void CClppa::trainLogLikelihood()
{
  incrementArgument();
  setMode("train-log-likelihood");
  string modelFileName="ppa_model";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();
	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    default: 
	      unrecognisedFlag();
	    }
	  incrementArgument();
	}
      else
	setFlags(false);
    }
  if(getCurrentArgumentNo()<argc) 
    modelFileName=getCurrentArgument();
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  double ll = pmodel->approxLogLikelihood();
  if(getVerbosity()>0)
    cout << "Model log likelihood: ";
  cout << ll << endl;
  exit(0);
}
void CClppa::logLikelihood()
{
  incrementArgument();
  setMode("log-likelihood");
  string modelFileName="ppa_model";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();
	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    default: 
	      unrecognisedFlag();
	    }
	  incrementArgument();
	}
      else
	setFlags(false);
    }
  if(getCurrentArgumentNo()>=argc)
    exitError("There are not enough input parameters.");
  string testDataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc) 
    modelFileName=argv[getCurrentArgumentNo()+1];
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  CMatrix X;
  CMatrix y;
  readData(X, y, testDataFileName);
  double ll = pmodel->logLikelihood(y, X);
  if(getVerbosity()>0)
    cout << "Model log likelihood: ";
  cout << ll << endl;
  exit(0);
}
void CClppa::predict()
{
  incrementArgument();
  setMode("predict");
  string modelFileName="ppa_model";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();

	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    default: 
	      unrecognisedFlag();
	    }
	  incrementArgument();
	}
      else
	setFlags(false);
    }
  if(getCurrentArgumentNo()>=argc) 
    exitError("There are not enough input parameters.");
  
  string testDataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc) 
    modelFileName=argv[getCurrentArgumentNo()+1];
  string predictionFile="ppa_predictions";
  if((getCurrentArgumentNo()+2)<argc) 
    predictionFile=argv[getCurrentArgumentNo()+2];
  CMatrix X;
  CMatrix yPred;
  readData(X, yPred, testDataFileName);
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  yPred.resize(X.getRows(), pmodel->getNumProcesses());
  if(pmodel->getNumInputs()!=X.getCols())
    throw ndlexceptions::Error(testDataFileName + ": input data is not of correct dimension");  
  pmodel->out(yPred, X);
  yPred.toFile(predictionFile);
  exit(0);
}
void CClppa::classOneProbabilities()
{
  incrementArgument();
  setMode("class-one-probabilities");
  string modelFileName="ppa_model";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();

	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    default: 
	      unrecognisedFlag();
	    }
	  incrementArgument();
	}
      else
	setFlags(false);
    }
  if(getCurrentArgumentNo()>=argc)
    exitError("There are not enough input parameters.");
  string testDataFileName=getCurrentArgument();
  if((getCurrentArgumentNo()+1)<argc) 
    modelFileName=argv[getCurrentArgumentNo()+1];
  string probabilityFile="ppa_probabilities";
  if((getCurrentArgumentNo()+2)<argc) 
    probabilityFile=argv[getCurrentArgumentNo()+2];
  CMatrix X;
  CMatrix yTest;
  readData(X, yTest, testDataFileName);
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  if(pmodel->noise.getType()!="probit" && pmodel->noise.getType()!="ncnm")
    {
      exitError("Model file is not a classification model.");
    }
  yTest.resize(X.getRows(), pmodel->getNumProcesses());
  yTest.ones();
  if(pmodel->getNumInputs()!=X.getCols())
    throw ndlexceptions::Error(testDataFileName + ": input data is not of correct dimension");  
  CMatrix probs(yTest.getRows(), yTest.getCols());
  pmodel->likelihoods(probs, yTest, X);
  probs.toFile(probabilityFile);
  exit(0);
}

void CClppa::display()
{
  incrementArgument();
  setMode("display");
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();
	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    default: 
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
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  pmodel->display(cout);
  exit(0);
}

void CClppa::gnuplot()
{
  incrementArgument();
  setMode("gnuplot");
  double pointSize = 2;
  double lineWidth = 2;
  int resolution = 30;
  string name = "ppa";
  string modelFileName="ppa_model";
  while(isFlags())
    {
      if(isCurrentArgumentFlag())
	{
	  int j=1;
	  if(argv[getCurrentArgumentNo()][j]=='-')
	    j++;
	  else if(getCurrentArgumentLength()!=2)
	    unrecognisedFlag();
	  switch(argv[getCurrentArgumentNo()][j])
	    {
	    case '?': 
	      if(j==2)
		confirmCurrentArg("--?");
	      helpInfo(); 
	      exit(0);
	    case 'h': 
	      if(j==2)
		confirmCurrentArg("--help");
	      helpInfo(); 
	      exit(0);
	    case 'r': 
	      if(j==2)
		confirmCurrentArg("--resolution");
	      incrementArgument();
	      resolution=getIntFromCurrentArgument();
	      break;
	    case 'l': 
	      if(j==2)
		confirmCurrentArg("--line-width");
	      incrementArgument();
	      lineWidth=getDoubleFromCurrentArgument(); 
	      break;
	    case 'p': 
	      if(j==2)
		confirmCurrentArg("--point-size");
	      incrementArgument();
	      pointSize=getDoubleFromCurrentArgument(); 
	      break;
	    default: 
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
  CPpa* pmodel=readPpaFromFile(modelFileName, getVerbosity());
  if(pmodel->noise.getType()!="gaussian" && pmodel->getNumInputs()!=2)
    {
      exitError("Incorrect number of model inputs.");
    }
  if(pmodel->noise.getType()=="gaussian" && pmodel->getNumInputs()>2)
    {
      exitError("Incorrect number of model inputs.");
    }

  if(X.getCols()!=pmodel->getNumInputs())
    {
      exitError("Incorrect dimension of input data.");
    }
  if(pmodel->noise.getType()=="probit" || pmodel->noise.getType()=="ncnm")
    {
      pmodel->activeX.toFile(name+"_active_set.dat");
      int numPos = 0;
      int numNeg = 0;
      int numUnlab = 0;
      for(int i=0; i<X.getRows(); i++)
	{
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
      for(int i=0; i<X.getRows(); i++)
	{
	  if(y.getVal(i)==1.0)
	    {
	      for(int j=0; j<X.getCols(); j++)
		XPos.setVal(X.getVal(i, j), posIndex, j);
	      XPos.setVal(0, posIndex, XPos.getCols()-1);
	      posIndex++;
	    }
	  else if(y.getVal(i)==-1.0)
	    {
	      for(int j=0; j<X.getCols(); j++)
		XNeg.setVal(X.getVal(i, j), negIndex, j);
	      XNeg.setVal(0, negIndex, XNeg.getCols()-1);
	      negIndex++;
	    }
	  else
	    {
	      for(int j=0; j<X.getCols(); j++)
		XUnlab.setVal(X.getVal(i, j), unlabIndex, j);
	      XUnlab.setVal(0, unlabIndex, XUnlab.getCols()-1);
	      unlabIndex++;
	    }
	}
      if(numPos>0)
	XPos.toFile(name+"_positive.dat");
      if(numNeg>0)
	XNeg.toFile(name+"_negative.dat");
      if(numUnlab>0)
	XUnlab.toFile(name+"_unlabelled.dat");
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
      double x;
      double y;
      int i;
      int j;
      for(i=0, y=minVals.getVal(0, 1); i<numy; y+=ydiff, i++)
	{
	  probOut.push_back(new CMatrix(numx, 3));
	  for(j=0,  x=minVals.getVal(0, 0); j<numx; x+=xdiff, j++)
	    {
	      Xgrid.setVal(x, i*numy+j, 0);
	      probOut[i]->setVal(x, j, 0);
	      Xgrid.setVal(y, i*numy+j, 1);
	      probOut[i]->setVal(y, j, 1);
	    }
	}
      CMatrix yTest(Xgrid.getRows(), pmodel->getNumProcesses());
      yTest.ones();
      CMatrix probs(yTest.getRows(), yTest.getCols());
      pmodel->likelihoods(probs, yTest, Xgrid);
      for(int i=0; i<numy; i++)
	for(int j=0; j<numx; j++)
	  {
	    probOut[i]->setVal(probs.getVal(i*numy+j), j, 2);
	  }
      string matrixFile = name + "_prob_matrix.dat";
      ofstream out(matrixFile.c_str());
      if(!out) throw ndlexceptions::FileWriteError(matrixFile);
      out << setiosflags(ios::scientific);
      out << setprecision(17);
      out << "# Prepared plot of model file " << endl;
      for(int i=0; i<numy; i++)
	{
	  probOut[i]->toStream(out);
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
  else if(pmodel->noise.getType()=="gaussian")
    {
      CMatrix scatterActive(pmodel->activeX.getRows(), pmodel->activeX.getCols()+1);
      for(int i=0; i<pmodel->activeX.getRows(); i++)
	{
	  for(int j=0; j<pmodel->activeX.getCols(); j++)
	    scatterActive.setVal(pmodel->activeX.getVal(i, j), i, j);
	  scatterActive.setVal(pmodel->trainY.getVal(i, 0), i, pmodel->activeX.getCols());
	}
      
      scatterActive.toFile(name+"_active_set.dat");
      CMatrix scatterData(X.getRows(), X.getCols()+1);
      for(int i=0; i<X.getRows(); i++)
	{
	  for(int j=0; j<X.getCols(); j++)
	    scatterData.setVal(X.getVal(i, j), i, j);
	  scatterData.setVal(y.getVal(i, 0), i, X.getCols());
	}
      scatterData.toFile(name+"_scatter_data.dat");
      CMatrix minVals(1, X.getCols());
      CMatrix maxVals(1, X.getCols());
      X.maxRow(maxVals);
      X.minRow(minVals);
      
      if(pmodel->activeX.getCols()==2) // two dimensional inputs.
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
	  for(i=0,  y=minVals.getVal(0, 1); i<numy; y+=ydiff, i++)
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
	  CMatrix outVals(Xgrid.getRows(), pmodel->getNumProcesses());
	  pmodel->out(outVals, Xgrid);
	  for(int i=0; i<numy; i++)
	    for(int j=0; j<numx; j++)
	      {
		regressOut[i]->setVal(outVals.getVal(i*numy+j), j, 2);
	      }
	  string matrixFile = name + "_output_matrix.dat";
	  ofstream out(matrixFile.c_str());
	  if(!out) throw ndlexceptions::FileWriteError(matrixFile);
	  out << setiosflags(ios::scientific);
	  out << setprecision(17);
	  out << "# Prepared plot of model file " << endl;
	  for(int i=0; i<numy; i++)
	    {
	      regressOut[i]->toStream(out);
	      out << endl;
	    }
	  out.close();
	  string plotFileName = name + "_plot.gp";
	  ofstream outGnuplot(plotFileName.c_str());
	  if(!outGnuplot) throw ndlexceptions::FileWriteError(plotFileName);
	  outGnuplot << "splot \"" << name << "_output_matrix.dat\"  with lines lw " << lineWidth;
	  outGnuplot << ", \"" << name << "_scatter_data.dat\" with points ps " << pointSize;
	  outGnuplot << ", \"" << name << "_active_set.dat\" with points ps " << pointSize << endl;
	  outGnuplot << "pause -1";
	  outGnuplot.close();
	}
      else if(pmodel->activeX.getCols()==1) // one dimensional input.
	{	  
	  int numx=resolution;
	  double xspan=maxVals.getVal(0, 0)-minVals.getVal(0, 0);
	  double xdiff=xspan/(numx-1);
	  CMatrix Xinvals(numx, 1);
	  CMatrix regressOut(numx, 2);
	  CMatrix errorBarPlus(numx, 2);
	  CMatrix errorBarMinus(numx, 2);
	  double x;
	  int j;
	  for(j=0,  x=minVals.getVal(0, 0); j<numx; x+=xdiff, j++)
	    {
	      Xinvals.setVal(x, j, 0);
	      regressOut.setVal(x, j, 0);
	      errorBarPlus.setVal(x, j, 0);
	      errorBarMinus.setVal(x, j, 0);
	    }
	  CMatrix outVals(Xinvals.getRows(), pmodel->getNumProcesses());
	  CMatrix stdVals(Xinvals.getRows(), pmodel->getNumProcesses());
	  pmodel->out(outVals, stdVals, Xinvals);
	  for(int j=0; j<numx; j++)
	    {
	      double val = outVals.getVal(j);
	      regressOut.setVal(val, j, 1);
	      errorBarPlus.setVal(val + stdVals.getVal(j), j, 1);
	      errorBarMinus.setVal(val - stdVals.getVal(j), j, 1);
	    }
	  string lineFile = name + "_line_data.dat";
	  regressOut.toFile(lineFile);
	  string errorFile = name + "_error_bar_data.dat";
	  ofstream out(errorFile.c_str());
	  if(!out) throw ndlexceptions::FileWriteError(errorFile);
	  out << setiosflags(ios::scientific);
	  out << setprecision(17);
	  out << "# Prepared plot of model file " << endl;
	  errorBarPlus.toStream(out);
	  out << endl;
	  errorBarMinus.toStream(out);
	  string plotFileName = name + "_plot.gp";
	  ofstream outGnuplot(plotFileName.c_str());
	  if(!outGnuplot) throw ndlexceptions::FileWriteError(plotFileName);
	  outGnuplot << "plot \"" << name << "_line_data.dat\" with lines lw " << lineWidth;
	  outGnuplot << ", \"" << name << "_scatter_data.dat\" with points ps " << pointSize;
	  outGnuplot << ", \"" << name << "_active_set.dat\" with points ps " << pointSize;
	  outGnuplot << ", \"" << name << "_error_bar_data.dat\" with lines lw " << lineWidth << endl;
	  outGnuplot << "pause -1";
	  outGnuplot.close();
	}      
  
    }
  else 
    {
      exitError("Unknown noise model for gnuplot output.");
    }
  exit(0);
}

void CClppa::helpInfo()
{
  string command = getMode();
  if(command=="ppa")
    {
      helpHeader();
      helpUsage("ppa [options] command [command-options]");
      cout << "Commands:" << endl;
      helpArgument("learn", "For learning the PPA model.");
      helpArgument("relearn", "For learning an PPA model using a file to initialise.");
      helpArgument("test", "For getting the performance of a learned model on a test set.");
      helpArgument("predict", "For making predictions on a test set.");
      helpArgument("display", "For displaying the parameters of a learned PPA model.");
      helpArgument("gnuplot", "For plotting the results in gnuplot.");
      helpArgument("class-one-probabilities", "Classification models only: for giving the probability of data coming from the positive class.");
      helpArgument("log-likelihood", "For computing the log-likelihood of a data set.");
      helpDescriptor("For help on a specific command type ppa command --h");
      cout << endl;
      cout << "Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      helpArgument("-v, --verbosity [0..3]", "Verbosity level (default is set to 2).");      
    }
  else if(command=="learn")
    {
      helpHeader();
      helpUsage("ppa [options] learn -a int [command-options] example_file [model_file]");
      helpDescriptor("This file is for learning a data set with an PPA. By default the kernel parameters are also optimised for 4 iterations.");
      cout << "Arguments:" << endl;
      helpArgument("example_file", "File with the training data. The data file is in the format used by SVM light.");
      helpArgument("model_file", "File to store the resulting model in. By default this is set to ppa_model.");
      cout << endl;
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      helpArgument("-o, --output-type {c,r,n}", "select between classification, regression and null category noise model.");
      helpArgument("-e, --external-iterations int", "Number of external iterations. Default is 4.");
      helpArgument("-l, --labelled-indices-file string", "File containing indices of labelled points. The index starts from 1.");
      helpArgument("-n, --noise-iterations int", "Number of iterations for optimising noise model. Default is 20.");
      helpArgument("-#, --#kernel-iterations int", "Number of iterations for optimising kernel. Default is 100.");
      string m=getMode();
      setMode("kern");
      helpInfo();
      setMode(m);
    }
  else if(command=="relearn")
    {
      helpHeader();
      helpUsage("ppa [options] relearn -a int [command-options] example_file [model_file] [new_model_file]");
      helpDescriptor("This file is for learning a data set with an PPA, using a model file to initialise the learning. By default the kernel parameters are also optimised for 4 iterations.");
      cout << "Arguments:" << endl;
      helpArgument("example_file", "File with the training data. The data file is in the format used by SVM light.");
      helpArgument("model_file", "File to load the initial model from. By default this is set to ppa_model.");
      helpArgument("new_model_file", "File to save the learnt model to. By default this is set to ppa_model.");
      cout << endl;
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      helpArgument("-e, --external-iterations int", "Number of external iterations. Default is 4.");
      helpArgument("-l, --labelled-indices-file string", "File containing indices of labelled points. The index starts from 1.");
      helpArgument("-n, --noise-iterations int", "Number of iterations for optimising noise model. Default is 20.");
      helpArgument("-#, --#kernel-iterations int", "Number of iterations for optimising kernel. Default is 100.");
    }
  else if(command=="kern")
    {
      cout << endl << "Kernel options:" << endl;
      helpArgument("-k, --kernel string", "Type of kernel function. Currently available options are lin (linear), poly (polynomial -- not recommended), rbf (radial basis function) and mlp (multi-layer perceptron otherwise known as arcsin).");
      helpArgument("-g, --gamma float", "Inverse width parameter in RBF kernel.");
      helpArgument("-v, --variance float", "Variance parameter for kernel.");
      helpArgument("-w, --weight float", "Weight parameter for polynomial and MLP kernel.");
      helpArgument("-b, --bias float", "Bias parameter for polynomial and MLP kernel.");
      helpArgument("-d, --degree int", "Degree parameter for polynomial kernel.");
    }    
  else if(command=="test")
    {
      helpHeader();
      helpUsage("ppa [options] test [command-options] test_data_file [model_file]");
      helpDescriptor("This file is for testing a data set with an PPA. It gives an output as a classification error (for classification) or a mean squared error (for regression).");
      cout << "Arguments:" << endl;
      helpArgument("test_data_file", "File with the test data (in SVM light format).");
      helpArgument("model_file", "File containing the model to be tested. By default this is set to ppa_model.");

      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
    }
  else if(command=="class-one-probabilities")
    {
      helpHeader();
      helpUsage("ppa [options] class-one-probabilities [command-options] test_data_file [model_file] [predictions_file]");
      helpDescriptor("This command is for predicting the probability of membership of class one (classification only).");
      cout << "Arguments:" << endl;
      helpArgument("test_data_file", "File containing the data to make predictions on (in SVM light format).");
      helpArgument("model_file", "File containing the model to be tested. By default this is set to ppa_model.");
      helpArgument("predictions_file", "File containing the predictions made by the PPA for each data point. By default this is set to ppa_probabilities.");
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
    }
  else if(command=="log-likelihood")
    {
      helpHeader();
      helpUsage("ppa [options] log-likelihood [command-options] test_data [model_file]");
      helpDescriptor("This command is for measuring the bound on the log likelihood of the training set.");
      cout << "Arguments:" << endl;
      helpArgument("test_data", "File containing the data to be evaluated.");
      helpArgument("model_file", "File containing the model to be tested. By default this is set to ppa_model.");
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      
    }
  else if(command=="predict")
    {
      helpHeader();
      helpUsage("ppa [options] predict [command-options] test_data_file [model_file] [predictions_file]");
      helpDescriptor("This command is for making predictions with the PPA.");
      cout << "Arguments:" << endl;
      helpArgument("test_data_file", "File containing the data to make predictions on (in SVM light format).");
      helpArgument("model_file", "File containing the model to be tested. By default this is set to ppa_model.");
      helpArgument("predictions_file", "File containing the predictions made by the PPA for each data point. By default this is set to ppa_predictions.");
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
    }
  else if(command=="display")
    {
      helpHeader();
      helpUsage("ppa [options] display [model_file]");
      helpDescriptor("Summarise the contents of a model file for quick viewing. The model is loaded in and the parameters of the PPA are given.");
      cout << "Arguments:" << endl;
      helpArgument("model_file", "File containing the model to be tested. By default this is set to ppa_model.");
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
    }
  else if(command=="gnuplot")
    {
      helpHeader();
      helpUsage("ppa [options] gnuplot [data_file] [model_file] [name]");
      helpDescriptor("This command creates files for some simple results in gnuplot. For classification the model must take two dimensions of input and the output is a classification decision boundary with contours at posterior probabilities of 0.25 and 0.75.");
      cout << "Arguments:" << endl;
      helpArgument("data_file", "The data to plot, it can be test data or training data, whatever you want to visualise. It should be in SVM light format.");
      helpArgument("model_file", "The PPA model you want to use for generating the plot. The default is ppa_model");
      helpArgument("name", "The gnuplot script will be output as \"name_plot.gp\". Data files needed etc. will also be preceded by \"name\". Default is \"ppa\".");
      cout << "Command Options:" << endl;
      helpArgument("-?, -h, --help", "This help.");
      helpArgument("-l, --line-width float", "Width of the lines in gnuplot.");
      helpArgument("-p, --point-size float", "Size of the point markers in gnuplot.");
      helpArgument("-r, --resolution int", "Resolution of the mesh grid in 3-d plots. The grid will have the given number of points in each direction.");
    }
}  

void CClppa::helpHeader()
{
  cout << endl << "PPA Code: Version 0.1" << endl;
}
