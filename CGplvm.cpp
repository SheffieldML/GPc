#include "CGplvm.h"

CGplvm::CGplvm()
  : CDataModel(), CProbabilisticOptimisable()
{ 
  pkern=NULL;
  pnoise=NULL;
  _init();
}
void CGplvm::_init()
{
  setBaseType("DataModel");
  setType("gplvm");
  setName("Gaussian process latent variable model");
}

CGplvm::CGplvm(CKern* kernel, CScaleNoise* nois, 
               const int latDim, const int verbos)
  :
  CDataModel(), CProbabilisticOptimisable(),
  pkern(kernel), dynKern(0), bK(0),
  pnoise(nois), latentDim(latDim), KupToDate(false)
{
  _init();
  setApproximationType("ftc");
  setInputScaleLearnt(false);
  setLatentRegularised(true);
  setDynamicModelLearnt(false);
  setBackConstrained(false);
  setVerbosity(verbos);
  dataDim=pnoise->getOutputDim();
  numData=pnoise->getNumData();
  pX = new CMatrix();
  initStoreage();
  initVals();
}

CGplvm::CGplvm(CKern* kernel, CKern* dynKernel, CScaleNoise* nois, 
               const int latDim, const int verbos)
  :
  CDataModel(), CProbabilisticOptimisable(), pkern(kernel), dynKern(dynKernel), bK(0),
  pnoise(nois), latentDim(latDim), KupToDate(false)
{
  _init();
  setApproximationType("ftc");
  setInputScaleLearnt(false);
  setLatentRegularised(true);
  setDynamicModelLearnt(true);
  setDynamicScaling(false);
  setDynamicKernelLearnt(true);
  setBackConstrained(false);
  setVerbosity(verbos);
  dataDim=pnoise->getOutputDim();
  numData=pnoise->getNumData();
  pX = new CMatrix();
  initStoreage();
  initVals();
}

CGplvm::CGplvm(CKern* kernel, CMatrix* backKernel, CScaleNoise* nois, 
               const int latDim, const int verbos)
  :
  CDataModel(), CProbabilisticOptimisable(), pkern(kernel), dynKern(0), bK(backKernel),
  pnoise(nois), latentDim(latDim), KupToDate(false)
{
  _init();
  setApproximationType("ftc");
  setInputScaleLearnt(false);
  setLatentRegularised(true);
  setDynamicModelLearnt(false);
  setBackConstrained(true);
  setVerbosity(verbos);
  dataDim=pnoise->getOutputDim();
  numData=pnoise->getNumData();
  pX = new CMatrix();
  initStoreage();
  initVals();
}

CGplvm::CGplvm(CKern* kernel, CKern* dynKernel, CMatrix* backKernel,
               CScaleNoise* nois, 
               const int latDim, const int verbos)
  :
  CDataModel(), CProbabilisticOptimisable(), pkern(kernel), dynKern(dynKernel), bK(backKernel),
  pnoise(nois), latentDim(latDim), KupToDate(false)
{
  _init();
  setApproximationType("ftc");
  setInputScaleLearnt(false);
  setLatentRegularised(true);
  setDynamicModelLearnt(true);
  setDynamicScaling(false);
  setDynamicKernelLearnt(true);
  setBackConstrained(true);
  setVerbosity(verbos);
  dataDim=pnoise->getOutputDim();
  numData=pnoise->getNumData();
  pX = new CMatrix();
  initStoreage();
  initVals();
}


void CGplvm::initStoreage()
{
  pX->resize(numData, latentDim);
  m.resize(numData, dataDim);
  beta.resize(numData, dataDim);
  nu.resize(numData, dataDim);
  g.resize(numData, dataDim);
  K.resize(numData, numData);
  invK.resize(numData, numData);
  LcholK.resize(numData, numData);
  for(int i=0; i<numData; i++)
    gX.push_back(new CMatrix(numData, latentDim));
  gDiagX.resize(numData, latentDim);
  covGrad.resize(numData, numData);
  if (isSparseApproximation())
  {
    X_u.resize(numActive, latentDim);
  }
  if (dynamicsLearnt) 
  {
    Xout.resize(numData, latentDim);
    dynBreakList.push_back(0);
    dynK.resize(numData,numData);
    invDynK.resize(numData,numData);
    LcholDynK.resize(numData, numData);
  }
  if (isBackConstrained())
  {
    A.resize(getNumData(), getLatentDim());
    bK->resize(getNumData(), getNumData());
  }
}

void CGplvm::initVals()
{
  for(int i=0; i<numData; i++)
    pnoise->updateSites(m, beta, i, nu, g, i);
  initX();
}

void CGplvm::initXrand()
{
  CMatrix& Xvals = isBackConstrained() ? A : *pX;
  Xvals.randn(0.001, 0.0);
  updateX();
}

void CGplvm::initX()
{
  initXpca();
  updateX();
}

void CGplvm::initXpca()  // pca style initialisation.
{
  if (!isBackConstrained()) 
  {
    // Regular PCA initialization

    // Compute mean-subtracted Y'Y/N covariance matrix
    CMatrix ymean = meanCol(m);
    CMatrix covm(m.getCols(), m.getCols());
    covm.setSymmetric(true);
    covm.syrk(m, 1.0/(double)m.getRows(), 0.0, "u", "t");  
    ymean.trans();
    covm.syr(ymean, -1.0, "u");

    // Compute eigen decomposition of Y'Y/N
    CMatrix eigVals(1, m.getCols());
    covm.syev(eigVals, "v", "u");

    // X = Y * U * lambda^1/2, where U are the first few eigenvectors of Y'Y/N
    CMatrix Winv(m.getCols(), latentDim);
    for(int i=0; i<latentDim; i++)
    {
      Winv.copyColCol(i, covm, m.getCols()-1-i);
      Winv.scaleCol(i, 1/sqrt(eigVals.getVal(m.getCols()-1-i)));
    }
    pX->gemm(m, Winv, 1.0, 0.0, "n", "n");  // X := m*Winv

    // Subtract off mean of X also
    CMatrix meanX = meanCol(*pX);
    for(int i=0; i<pX->getRows(); i++)
      pX->axpyRowRow(i, meanX, 0, -1.0);
  }
  else
  {
    // PCA initialization with Back-kernel bK initialized before
    // constructor as kernel of the type bK_ij(Y_i,Y_j) after that it
    // isn't modified.  With an RBF back-kernel bK contains the
    // similarity of Y poses computed through the distances in Y
    // space. The reverse mapping should constrain things such that
    // `similar Ys' lead to `similar Xs'.

    // WVB: The objective value still depends on X, but now X = bk *
    // A, and it's the A we manipulate in the optimization, so there
    // we're doing d/d(A) and that makes a factor of bK pop out in the
    // gradient calc.

    // Get eigenVectors of bK
    CMatrix eigVectors(*bK);
    CMatrix eigVals(1, m.getRows());
    eigVectors.syev(eigVals, "v", "u");

    // X initialized to first few eigenvectors of bK
    for(int i=0; i<getLatentDim(); i++)
    {
      pX->copyColCol(i, eigVectors, m.getRows()-1-i);
    }

    // A initialized to solution of bK * A = X
    eigVectors.deepCopy(*bK);
    eigVectors.setSymmetric(true);
    A.deepCopy(*pX);
    A.sysv(eigVectors, "u"); // solve bK * A = X

  }
  updateX();
}

void CGplvm::updateX()
{
  if (isBackConstrained()) {
    // WVB TODO:  IS THIS THE RIGHT ORDER FOR THIS??? OR SHOULD
    // THE DYNAMICS UPDATE USE A??
    pX->symm(*bK, A, 1.0, 0.0, "u", "l"); // X := bK*A
  }
  if (dynamicsLearnt) {
    // Make an up-shifted copy of X into Xout:  Xout = [X(2:N,:); zeros]
    DIMENSIONMATCH(Xout.getRows()==pX->getRows());
    int nr = pX->getRows();
    for (int i=0; i<latentDim; i++) {
      dcopy_(nr-1, pX->getVals()+i*nr+1, 1, Xout.getVals()+i*nr, 1);
      for (int j=0; j<dynBreakList.size(); j++) {
        int fr = dynBreakList[j];
        fr = (fr==0) ? nr-1 : fr-1;
        Xout.setVal(0.0, fr, i);
      }
    }
  }
  setKupToDate(false);
}

unsigned int CGplvm::getOptNumParams() const
{
  unsigned int tot = pkern->getNumParams() + getNumData()*getLatentDim();
  if(isDynamicModelLearnt() && isDynamicKernelLearnt())
    tot+=dynKern->getNumParams();
  if(isInputScaleLearnt())
    tot+=getNumProcesses();
  return tot;
}

void CGplvm::getOptParams(CMatrix& param) const
{
  unsigned int counter = 0;
  for(unsigned int i=0; i<pkern->getNumParams(); i++)
  {
    param.setVal(pkern->getTransParam(i), counter);
    counter++;
  }
  if (dynamicsLearnt && dynamicKernelLearnt) {
    int nKernParams = pkern->getNumParams();
    for(int unsigned i=0; i<dynKern->getNumParams(); i++)
    {
      param.setVal(dynKern->getTransParam(i), counter);
      counter++;
    }
  }
  const CMatrix& Xvals = isBackConstrained() ? A : *pX;
  for(unsigned int j=0; j<getLatentDim(); j++)
  {
    for(unsigned int i=0; i<getNumData(); i++)
    {
      param.setVal(Xvals.getVal(i, j), counter);
      counter++;
    }
  }
  if(isInputScaleLearnt())
  {
    for(unsigned int j=0; j<getNumProcesses(); j++)
    {
      param.setVal(pnoise->getScale(j), counter);
      counter++;
    }
  }
}

void CGplvm::setOptParams(const CMatrix& param)
{
  setKupToDate(false);
  unsigned int counter=0;
  for(unsigned int i=0; i<pkern->getNumParams(); i++)
  {	  
    pkern->setTransParam(param.getVal(counter), i);
    counter++;
  }
  if (dynamicsLearnt && dynamicKernelLearnt)
  {
    for(unsigned int i=0; i<dynKern->getNumParams(); i++)
    {	  
      dynKern->setTransParam(param.getVal(counter), i);
      counter++;
    }
  }
  CMatrix& Xvals = isBackConstrained() ? A : *pX;
  for(unsigned int j=0; j<getLatentDim(); j++)
  {
    for(unsigned int i=0; i<getNumData(); i++)
    {
      Xvals.setVal(param.getVal(counter), i, j);
      counter++;
    }
  }
  updateX();
  if(isInputScaleLearnt())
  {
    for(unsigned int j=0; j<getNumProcesses(); j++)
    {
      pnoise->setScale(param.getVal(counter), j);
      counter++;
    }      
    for(unsigned int i=0; i<numData; i++) {
      pnoise->updateSites(m, beta, i, nu, g, i);
    }
  }
}

void CGplvm::out(CMatrix& yPred, const CMatrix& inData) const
{

}

void CGplvm::out(CMatrix& yPred, CMatrix& probPred, const CMatrix& inData) const
{
}

void CGplvm::posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const
{
  DIMENSIONMATCH(mu.getCols()==dataDim);
  DIMENSIONMATCH(varSigma.getCols()==dataDim);
  CMatrix kX(numData, Xin.getRows());
  updateK();

  pkern->compute(kX, *pX, Xin);
  // solve LcholK * tmp = kX,  kX := tmp
  kX.trsm(LcholK, 1.0, "L", "L", "N", "N"); // now it is LcholKinv*K
  for(int i=0; i<Xin.getRows(); i++)
  {
    double vsVal = pkern->diagComputeElement(Xin, i) - kX.norm2Col(i);
    BOUNDCHECK(vsVal>=0);
    for(int j=0; j<dataDim; j++)
      varSigma.setVal(vsVal, i, j);	    
  }
  // solve LcholK' * tmp = kX, kX := tmp
  kX.trsm(LcholK, 1.0, "L", "L", "T", "N"); // now it is invK * K
  mu.gemm(kX, m, 1.0, 0.0, "T", "N");
}


// Gradient routines
void CGplvm::updateCovGradient(int j, CMatrix& invKm) const
{
  // covGrad := 0.5*(invK * Y(:,j) * Y^t(:,j) * invK - invK)

  // (The 'D' factor doesn't need to be on the last term because we add 
  //  D of these covGrad expressions together in a loop in the caller.  
  //  In other words, this is covGrad for a single process only.)

  invKm.resize(invK.getRows(), 1);
  invKm.symvColCol(0, invK, m, j, 1.0, 0.0, "u"); // invKm = invK*m(:,index)
  covGrad.deepCopy(invK);                             // covgrad = invK
  covGrad.syr(invKm, -1.0, "u");                 // covGrad -= invKm * invKm^t
  covGrad.scale(-0.5);                           // covGrad *= -0.5
}

void CGplvm::updateDynCovGradient(int j, CMatrix& invKX) const
{
  // covGrad := 0.5*(invDynK * Xout(:,j) * Xout(:,j)' * invDynK - invDynK)

  // (The 'd' factor doesn't need to be on the last term because we add 
  //  d of these covGrad expressions together in a loop in the caller.)

  invKX.resize(invDynK.getRows(), 1);
  invKX.symvColCol(0, invDynK, Xout, j, 1.0, 0.0, "u"); // invKX = invDynK*Xout(:,j)
  covGrad.deepCopy(invDynK);                      // covgrad = invDynK
  covGrad.syr(invKX, -1.0, "u");                 // covGrad -= invKX * invKX^t
  covGrad.scale(-0.5);                           // covGrad *= -0.5
  vector<int>::const_iterator brkFrame=dynBreakList.begin();
  vector<int>::const_iterator end=dynBreakList.end();
  for (; brkFrame!=end; ++brkFrame) {
    int f = *brkFrame;
    f = (f==0)? numData-1 : f-1;
    covGrad.setVal(0.0, f,f);
  }
}


void CGplvm::updateK() const
{
  if(!isKupToDate())
  {
    _updateK();
    _updateInvK();
    pkern->updateX(*pX);
    if (dynamicsLearnt) {
      _updateDynK();
      _updateInvDynK();
      dynKern->updateX(*pX);
    }
    setKupToDate(true);
  }
}

void CGplvm::_updateK() const
{
  double kVal=0.0;
  for(int i=0; i<numData; i++)
  {
    K.setVal(pkern->diagComputeElement(*pX, i), i, i);
    for(int j=0; j<i; j++)
    {
      kVal=pkern->computeElement(*pX, i, *pX, j);
      K.setVal(kVal, i, j);
      K.setVal(kVal, j, i);
    }
  }
  K.setSymmetric(true);
}

// update invK with the inverse of the kernel plus beta terms computed from the active points.
void CGplvm::_updateInvK(int dim) const
{
  LcholK.deepCopy(K);
  // 
  //for(int i=0; i<numData; i++)
  //  invK.setVal(invK.getVal(i, i) + 1/getBetaVal(i, dim), i, i);
  LcholK.chol(); /// this will initially be upper triangular.
  logDetK = logDet(LcholK);
  invK.setSymmetric(true);
  invK.pdinv(LcholK);
  LcholK.trans();
}

void CGplvm::_updateDynK() const
{
  double kVal=0.0;
  for(int i=0; i<numData; i++)
  {
    dynK.setVal(dynKern->diagComputeElement(*pX, i), i, i);
    for(int j=0; j<i; j++)
    {
      kVal=dynKern->computeElement(*pX, i, *pX, j);
      dynK.setVal(kVal, i, j);
      dynK.setVal(kVal, j, i);
    }
  }
  // Zero out rows & cols that are on break frames
  vector<int>::const_iterator brkFrame=dynBreakList.begin();
  vector<int>::const_iterator end=dynBreakList.end();
  for (; brkFrame!=end; ++brkFrame) {
    int f = *brkFrame;
    f = (f==0)? numData-1 : f-1;
    for (int i=0; i<numData; i++) {
      dynK.setVal(0.0, f,i);
      dynK.setVal(0.0, i,f);
    }
    dynK.setVal(1.0, f,f);
  }
  dynK.setSymmetric(true);
}

void CGplvm::_updateInvDynK(int dim) const
{
  LcholDynK.deepCopy(dynK);

  LcholDynK.chol(); /// this will initially be upper triangular.

  // Note that dynK has some row/cols knocked out.  These will have 1's
  // on the diag of the cholesky decomp, and 1 on the diag doesn't contribute
  // to the logDet.
  logDetDynK = logDet(LcholDynK);
  invDynK.pdinv(LcholDynK);
  invDynK.setSymmetric(true);
  LcholDynK.trans();
}


// compute the approximation to the log likelihood.
double CGplvm::logLikelihood() const
{
  updateK();
  double L=0.0;
  CMatrix invKm(invK.getRows(), 1);
  for(int j=0; j<dataDim; j++)
  {
    // This computes invK*Y*Y', column by column of Y
    // invKm := invK * m(:,j)
    invKm.symvColCol(0, invK, m, j, 1.0, 0.0, "u");
    // L += invKm' * m(:,j)
    L += invKm.dotColCol(0, m, j);
    L += logDetK; 
  }
  if (isDynamicModelLearnt())
  {
    for(int j=0; j<latentDim; j++)
    {
      // This computes invDynK*Xout*Xout', column by column of Xout
      // Xout and invDynK are set up so that the rows that represent 
      // sequence breaks don't contribute.

      // invKm := invDynK * Xout(:,j)
      invKm.symvColCol(0, invDynK, Xout, j, 1.0, 0.0, "u");

      // L += invKm' * Xout(:,j)
      double d = invKm.dotColCol(0, Xout, j);
      L += dynamicScalingVal*d;
      L += dynamicScalingVal*logDetDynK;
    }
    if(isLatentRegularised())
      {
	// Regularise the first point in the sequence with an L2 term.
	L+=pX->norm2Col(0);
      }
  }
  else
    {
      if(isLatentRegularised())
	{
	  // Regularise the latent space with a Gaussian distribution.
	  for(int j=0; j<getLatentDim(); j++)
	    {
	      L+=pX->norm2Col(j);
	    }
	}
    }
  
  if(isInputScaleLearnt())
    {
      // scales lead to terms of the form log w_j to be added.
      for(int j=0; j<dataDim; j++)
	L+=2*log(fabs(pnoise->getScale(j)));
    }
  L*=-0.5;
  L+=pkern->priorLogProb();

  if(isDynamicModelLearnt() && isDynamicKernelLearnt())
    L+=dynKern->priorLogProb();
  return L;
}
// compute the gradients wrt parameters and latent variables.
double CGplvm::logLikelihoodGradient(CMatrix& g) const
{
  int numKernParams = pkern->getNumParams();
  int numDynKernParams = (isDynamicModelLearnt() && isDynamicKernelLearnt()) ? dynKern->getNumParams() : 0;
  int numParams = numKernParams + numDynKernParams;
  int xoffset = numParams;
  g.zeros();
  if (isBackConstrained()) {
    tempgX.resize(getNumData(), getLatentDim());
    tempgX.zeros();
    xoffset = 0;
  }
  CMatrix& gref = isBackConstrained() ? tempgX : g;
  updateK();
  pkern->getGradX(gX, *pX, *pX); // gX[1...ndata].val(1:ndata,1:latentdim)
  pkern->getDiagGradX(gDiagX, *pX);// zero except for w/ linear kernel
  for(int i=0; i<numData; i++)
  {
    gX[i]->scale(2.0); // accounts for symmetric covariance
                       // (WVB: it just pops out of the math...)
    for(int j=0; j<latentDim; j++)
	{
	  // deal with diagonal -- d kern(X_row_i,X_row_i) / d_component_j
	  gX[i]->setVal(gDiagX.getVal(i, j), i, j);
	}
  }
  CMatrix tempG(1, numKernParams);
  CMatrix tmpV(dataDim, 1);
  for(int j=0; j<dataDim; j++)
  {
    updateCovGradient(j, tmpV); //covGrad = -(invK Y(:,j) Y(:,j)^t invK - invK)/2
    if(j==0)
      pkern->getGradTransParams(tempG, *pX, covGrad, true);
    else
      pkern->getGradTransParams(tempG, *pX, covGrad, false);
    for(int i=0; i<numKernParams; i++)
    {
      g.addVal(tempG.getVal(i), i);
    }
    
    // This gives us the main dL/dX bit here
    for(int i=0; i<numData; i++)
      {
	for(int k=0; k<latentDim; k++)
	  {
	    int ind = xoffset + i + numData*k;
	    gref.addVal(gX[i]->dotColCol(k, covGrad, i), ind);
	  }
      }
  }
  if(isDynamicModelLearnt())
    {
      // Xout and invDynK are set up so that the rows that represent 
      // sequence breaks don't contribute.

      // Reuse gX.  Slightly wasteful in that we don't need derivs of first row.
      dynKern->getGradX(gX, *pX, *pX); // gX[1...ndata].val(1:ndata,1:latentdim)
      dynKern->getDiagGradX(gDiagX, *pX);// zero except for w/ linear kernel
      for(int i=0; i<numData; i++)
	{
	  gX[i]->scale(2.0);
	  for(int j=0; j<latentDim; j++)
	    {
	      // deal with diagonal -- d kern(X_row_i,X_row_i) / d_component_j
	      gX[i]->setVal(gDiagX.getVal(i, j), i, j);
	    }
	}
      CMatrix& invKX = tmpV;
      tempG.resize(1, numDynKernParams);
      for(int j=0; j<latentDim; j++)
	{
	  // covGrad = -0.5*(invDynK Xout(:,j) Xout(:,j)^t invDynK - invDynK)
	  // invKX   = invDynk Xout(:,j)
	  updateDynCovGradient(j,invKX);
	  if(isDynamicKernelLearnt())
	    {
	      if(j==0)
		// add any priors on the kernel parameters in
		dynKern->getGradTransParams(tempG, *pX, covGrad, true);
	      else
		dynKern->getGradTransParams(tempG, *pX, covGrad, false);
	      for(int i=0; i<numDynKernParams; i++)
		{
		  g.addVal(tempG.getVal(i), i+numKernParams);
		}
	    }
	  
	  // This gives us the main dL/dX bit for the dynamics kernel
	  for(int i=0; i<numData; i++)
	    {
	      for(int k=0; k<latentDim; k++)
		{
		  int ind = numParams + numData*k + i;
		  double v = gX[i]->dotColCol(k, covGrad, i);
		  g.addVal(v, ind);
		}
	      // One more part of dL/dX is required here since Xout is
	      // dependent on X Works out to be - P^t * invDK * Xout g
	      // -= P^t * invKX, where P is the shifter matrix [ the
	      // (i+1)%numData is implementing the P^t ]
	      int ind = xoffset + numData*j + ((i+1)%numData);
	      double factor = -1.0*dynamicScalingVal;
	      gref.addVal(factor*invKX.getVal(i), ind);
	    }
	}    
      if(isLatentRegularised())
	{
	  for(int k=0; k<latentDim; k++)
	    {
	      // grad of -X magnitude squared for first point.
	      int ind = xoffset+numData*k;
	      gref.addVal(-pX->getVal(0, k), ind);
	    } 
	}
    }
  else
    {
      if(isLatentRegularised())
	{
	  for(int i=0; i<numData; i++)
	    for(int k=0; k<latentDim; k++)
	      {
		// grad of -X magnitudes squared, just -X
		int ind = xoffset+i+numData*k;
		gref.addVal(-pX->getVal(i, k), ind);
	      } 
	}
    }
  if(isBackConstrained())
    {
      CMatrix tempgX2(getNumData(), 1);
      for(int k=0; k<getLatentDim(); k++)
	{
	  // The objective used is actually identical, just the gradients differ.
	  //
	  //   tempgX2(:, 0) :=  bK * tempgX(:, k)
	  //         g(:, k) :=  texpgX2(:,0)
	  // i.e.
	  //   g := bK * tempgX
	  tempgX2.symvColCol(0, *bK, gref, k, 1.0, 0.0, "u");
	  for(int i=0; i<getNumData(); i++)
	    {
	      int ind = numParams + getNumData()*k + i;
	      g.setVal(tempgX2.getVal(i, 0), ind);
	    }
	}
    }
  if(isInputScaleLearnt())
    {
      CMatrix &invKm = tmpV;  tmpV.resize(m.getRows(), 1);
      for(int j=0; j<getNumProcesses(); j++)
	{
	  // recomputing this again is inefficient (already done in the likelihood).
	  invKm.symvColCol(0, invK, m, j, 1.0, 0.0, "u");      
	  int ind = numParams + numData*latentDim + j;
	  g.setVal(1/pnoise->getScale(j)*(invKm.dotColCol(0, m, j)-1), ind);
	}
    }
  // Need to check if there is more efficient ways of calling this.
  return logLikelihood();

}
void CGplvm::pointLogLikelihood(const CMatrix& y, const CMatrix& X) const
{
  
}
// Optimise the GPLVM with respect to latent positions and kernel parameters.
void CGplvm::optimise(const int iters)
{
  if(getVerbosity()>2)
    {
      cout << "Initial model:" << endl;
      display(cout);
    }
  if(getVerbosity()>2 && getOptNumParams()<40)
    checkGradients();
  setMaxIters(iters);
  runDefaultOptimiser();

  if(getVerbosity()>1)
    cout << "... done. " << endl;
  if(getVerbosity()>0)
    display(cout);
}

bool CGplvm::equals(const CGplvm& model, const double tol) const
{
  throw ndlexceptions::Error("equals not yet implemented for GPLVM.");
}

void CGplvm::display(ostream& os) const
{
  cout << "GPLVM Model: " << endl;
  cout << "Data Set Size: " << numData << endl;
  cout << "Kernel Type: " << endl;
  cout << "Latent space regularised: " << isLatentRegularised() << endl;
  cout << "Dynamics learnt: " << isDynamicModelLearnt() << endl;
  cout << "Scales learnt: " << isInputScaleLearnt() << endl;
  pnoise->display(os);
  pkern->display(os);
  if (isDynamicModelLearnt()) {
    cout << "Dynamics' ";
    dynKern->display(os);
  }
}

void CGplvm::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  out << "numData=" << getNumData() << endl;
  out << "outputDim=" << getNumProcesses() << endl;
  out << "inputDim=" << getLatentDim() << endl;
  out << "latentRegularised=" << isLatentRegularised() << endl;
  out << "backConstrained=" << isBackConstrained() << endl;
  out << "dynamicsLearnt=" << isDynamicModelLearnt() << endl;
  pkern->toStream(out);
  if(isBackConstrained()){
    // In future back constraint info goes here.
  }
  if(isDynamicModelLearnt())
    dynKern->toStream(out);
  pnoise->toStream(out);
  out << "Y:" << getNumProcesses() << ",X:" << getLatentDim();
  if(isLabels())
    out << ",labels:1" << endl;
  else
    out << endl;
  for(int i=0; i<getNumData(); i++)
    {
      for(int j=0; j<getNumProcesses(); j++)
	{
 	  out << pnoise->getTarget(i, j) << " ";	  
	}
      for(int j=0; j<getLatentDim(); j++)
 	{
 	  out << pX->getVal(i, j) << " ";
 	}
      if(isLabels())
	{
	  out << getLabel(i) << endl;
	}
      else
	out << endl;
    }

}
void CGplvm::readParamsFromStream(istream& in)
{
  string tbaseType = getBaseTypeStream(in);
  if(tbaseType != getBaseType())
    throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getBaseType() + ".");
  string ttype = getTypeStream(in);
  if(ttype != getType())
    throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");


  setNumData(readIntFromStream(in, "numData"));
  setNumProcesses(readIntFromStream(in, "outputDim"));
  setLatentDim(readIntFromStream(in, "inputDim"));

  setLatentRegularised(readBoolFromStream(in, "latentRegularised"));
  setBackConstrained(readBoolFromStream(in, "backConstrained"));
  setDynamicModelLearnt(readBoolFromStream(in, "dynamicsLearnt"));
  
  pX = new CMatrix(numData, latentDim);
  if(isBackConstrained())
  {
    bK = new CMatrix(numData, numData);
  }
  // initialise storage
  initStoreage();

  // read kernel from the stream.
  //delete pkern; --- want to destroy it if it exists ... but not sure how to.
  pkern = readKernFromStream(in);
  
  if(isBackConstrained()){
    // future back constraint read.
  }
  if(isDynamicModelLearnt())
    dynKern = readKernFromStream(in);
  
  pnoise = (CScaleNoise*)readNoiseFromStream(in);
  string line;
  vector<string> tokens;

  tokens.clear();
  ndlstrutil::getline(in, line);
  ndlstrutil::tokenise(tokens, line, ",");
  bool labelsPresent = false;
  vector<string> subTokens;
  for(int i=0; i<tokens.size(); i++)
    {
      subTokens.clear();
      ndlstrutil::tokenise(subTokens, tokens[i], ":");
      if(subTokens[0]=="Y")
	{
	  if(atoi(subTokens[1].c_str())!=dataDim)
	    throw ndlexceptions::FileFormatError();
	}
      else if(subTokens[0]=="X")
	{
	  if(atoi(subTokens[1].c_str())!=latentDim)
	    throw ndlexceptions::FileFormatError();
	}
      else if(subTokens[0]=="labels")
	{
	  labelsPresent=true;
	  if(atoi(subTokens[1].c_str())!=1)
	    throw ndlexceptions::FileFormatError();
	}
      else
	{
	  throw ndlexceptions::FileFormatError();
	}
    }
  
  vector<int> labels;
  pX = new CMatrix(numData, latentDim, 0.0);
  CMatrix* pY = new CMatrix(numData, dataDim);
  vector<int> activeSet;
  for(int i=0; i<numData; i++)
    {
      tokens.clear();
      ndlstrutil::getline(in, line);
      ndlstrutil::tokenise(tokens, line, " ");
      for(int j=0; j<dataDim; j++)
	{
	  pY->setVal(atof(tokens[j].c_str()), i, j);
	}
      for(int j=0; j<latentDim; j++)
	{
	  pX->setVal(atof(tokens[j+dataDim].c_str()), i, j);
	}      
      if(labelsPresent)
	labels.push_back(atoi(tokens[latentDim+dataDim].c_str()));
	  
    }
  pnoise->setTarget(pY);
  CGplvm* pmodel;
  setLatentVals(pX);
  if(labelsPresent)
    setLabels(labels);
}


// Functions which operate on the object
void writeGplvmToStream(const CGplvm& model, ostream& out)
{
  model.toStream(out);
}

void writeGplvmToFile(const CGplvm& model, const string modelFileName, const string comment)
{
  if(model.getVerbosity()>0)
    cout << "Saving model file." << endl;
  ofstream out(modelFileName.c_str());
  if(!out) throw ndlexceptions::FileWriteError(modelFileName);
  out << setiosflags(ios::scientific);
  out << setprecision(17);
  if(comment.size()>0)
    out << "# " << comment << endl;
  writeGplvmToStream(model, out);
  out.close();

}

CGplvm* readGplvmFromStream(istream& in)
{
  CGplvm* pmodel = new CGplvm();
  pmodel->fromStream(in);
  return pmodel;
}
    
CGplvm* readGplvmFromFile(const string modelFileName, const int verbosity)
{
  // File is m, beta, X
  if(verbosity>0)
    cout << "Loading model file." << endl;
  ifstream in(modelFileName.c_str());
  if(!in.is_open()) throw ndlexceptions::FileReadError(modelFileName);
  CGplvm* pmodel;
  try
    {
      pmodel = readGplvmFromStream(in);
    }
  catch(ndlexceptions::FileFormatError err)
    {
      throw ndlexceptions::FileFormatError(modelFileName);
    }
  if(verbosity>0)
    cout << "... done." << endl;
  in.close();
  return pmodel;

}
