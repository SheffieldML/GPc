#include "CMatrix.h"
using namespace std;

unsigned int CMatrix::totalAlloc = 0;

void CMatrix::copyRowRow(unsigned int i, const CMatrix& X, unsigned int k)
{
  DIMENSIONMATCH(X.ncols==ncols);
  BOUNDCHECK(i<nrows);
  BOUNDCHECK(k<X.nrows);
  dcopy_(ncols, X.vals+k, X.nrows, vals+i, nrows);
}
void CMatrix::copyColCol(unsigned int j, const CMatrix& X, unsigned int k)
{
  DIMENSIONMATCH(X.nrows==nrows);
  BOUNDCHECK(j<ncols);
  BOUNDCHECK(k<X.ncols);
  dcopy_(nrows, X.vals+k*X.nrows, 1, vals+j*X.nrows, 1);
}
void CMatrix::copy(const CMatrix& x)
{
  // Level 1 Blas operation y <- x
  DIMENSIONMATCH(x.ncols == ncols);
  DIMENSIONMATCH(x.nrows == nrows);
  dcopy_(ncols*nrows, x.vals, 1, vals, 1);
  symmetric = x.symmetric;
  triangular = x.triangular;

}
void CMatrix::gemv(const CMatrix& A, const CMatrix& x, double alpha, double beta, const char* trans)
{
  // Level 2 Blas operation y <- alpha op(A)*x + beta*y
  DIMENSIONMATCH(ncols==1);
  DIMENSIONMATCH(((trans[0]=='n' || trans[0]=='N') && A.ncols==x.nrows && A.nrows == nrows)
	 ||
	 ((trans[0]=='t' || trans[0]=='T') && A.nrows==x.nrows && A.ncols == nrows));
  dgemv_(trans, A.nrows, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals, 1, beta, vals, 1);
}
void CMatrix::gemvRowRow(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* trans)
{
  // Level 2 Blas operation y(i, :)' <- alpha op(A)*x(k, :)' + beta*y(i, :)';
  BOUNDCHECK(i<nrows);
  BOUNDCHECK(k<x.nrows);
  DIMENSIONMATCH(((trans[0]=='n' || trans[0]=='N') && A.ncols==x.ncols && A.nrows == ncols)
	 ||
	 ((trans[0]=='t' || trans[0]=='T') && A.nrows==x.ncols && A.ncols == ncols));
  dgemv_(trans, A.nrows, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+k, x.nrows, beta, vals+i, nrows);
}
void CMatrix::gemvRowCol(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int j, double alpha, double beta, const char* trans)
{
  // Level 2 Blas operation y(i, :)' <- alpha op(A)*x(:, j) + beta*y(i, :)';
  BOUNDCHECK(i<nrows);
  BOUNDCHECK(j<x.ncols);
  DIMENSIONMATCH(((trans[0]=='n' || trans[0]=='N') && A.ncols==x.nrows && A.nrows == ncols)
	 ||
	 ((trans[0]=='t' || trans[0]=='T') && A.nrows==x.nrows && A.ncols == ncols));
  dgemv_(trans, A.nrows, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+j*x.nrows, 1, beta, vals+i, nrows);
}
void CMatrix::gemvColCol(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* trans)
{
  // Level 2 Blas operation y(:, j) <- alpha op(A)*x(:, k) + beta*y(:, j);
  BOUNDCHECK(j<ncols);
  BOUNDCHECK(k<x.ncols);
  DIMENSIONMATCH(((trans[0]=='n' || trans[0]=='N') && A.ncols==x.nrows && A.nrows == nrows)
	 ||
	 ((trans[0]=='t' || trans[0]=='T') && A.nrows==x.nrows && A.ncols == nrows));
  dgemv_(trans, A.nrows, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+k*x.nrows, 1, beta, vals+j*nrows, 1);
}
void CMatrix::gemvColRow(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned  int i, double alpha, double beta, const char* trans)
{
  // Level 2 Blas operation y(:, j) <- alpha op(A)*x(i, :)' + beta*y(:, j);
  BOUNDCHECK(j<ncols);
  BOUNDCHECK(i<x.nrows);
  DIMENSIONMATCH(((trans[0]=='n' || trans[0]=='N') && A.ncols==x.ncols && A.nrows == nrows)
	 ||
	 ((trans[0]=='t' || trans[0]=='T') && A.nrows==x.ncols && A.ncols == nrows));
  dgemv_(trans, A.nrows, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+i, x.nrows, beta, vals+j*nrows, 1);
}  
// "l" is A*B and "r" is B*A;
void CMatrix::symm(const CMatrix& A, const CMatrix& B, double alpha, double beta, const char* type, const char* side)
{
  MATRIXPROPERTIES(A.isSymmetric());
  CHARARGUMENTS(side[0]=='L' || side[0]=='l' || side[0]=='R' || side[0]=='r');
  CHARARGUMENTS(type[0]=='L' || type[0]=='l' || type[0]=='U' || type[0]=='u');
  switch(side[0])
    {
    case 'L':
    case 'l':
      DIMENSIONMATCH(A.nrows==nrows);
      DIMENSIONMATCH(B.nrows==nrows);
      DIMENSIONMATCH(B.ncols==ncols);
      break;
    case 'R':
    case 'r':
      DIMENSIONMATCH(A.ncols==ncols);
      DIMENSIONMATCH(B.ncols==A.nrows);
      DIMENSIONMATCH(B.nrows==nrows);
      break;
    default:
      throw ndlexceptions::MatrixError("No such value for side.");
    }
  dsymm_(side, type, nrows, ncols, alpha, A.vals, A.nrows, B.vals, B.nrows, beta, vals, nrows);
}
int CMatrix::sysv(const CMatrix& A, const char* uplo, int lwork)
{
  MATRIXPROPERTIES(A.isSymmetric());
  CHARARGUMENTS(uplo[0]=='L' || uplo[0]=='l' || uplo[0]=='U' || uplo[0]=='u');
  DIMENSIONMATCH(nrows==A.nrows);
  if(lwork < 0)
    lwork = 3*nrows;
  int info;
  std::vector<int> ipivv(nrows);
  int *ipiv = &ipivv[0];
  CMatrix work(1, lwork);

  dsysv_(uplo, nrows, ncols, A.vals, A.nrows, ipiv, vals, nrows, work.vals, lwork, info);
  if(info>0) throw ndlexceptions::MatrixSingular();
  if(info<0) throw ndlexceptions::Error("Incorrect argument in SYSV.");
  return (int)work.getVal(0); // optimal value for lwork
}

void CMatrix::symv(const CMatrix& A, const CMatrix& x, double alpha, double beta, const char* upperLower)
{
  // Level 2 Blas operation, symmetric A,  y <- alpha A*x + beta*y
  MATRIXPROPERTIES(A.isSymmetric());
  DIMENSIONMATCH(ncols==1);
  DIMENSIONMATCH(x.ncols==1);
  CHARARGUMENTS(upperLower[0]=='u' || upperLower[0]=='U' || upperLower[0]=='l' || upperLower[0]=='L');
  DIMENSIONMATCH(nrows==A.nrows);
  DIMENSIONMATCH(nrows==x.nrows);

  
  dsymv_(upperLower, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals, 1, beta, vals, 1);
}
void CMatrix::symvRowRow(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* upperLower)
{
  // Level 2 Blas operation, symmetric A,  y(i, :)' <- alpha A*x(k, :)' + beta*y(i, :)';
  MATRIXPROPERTIES(A.isSymmetric());
  BOUNDCHECK(i<nrows);
  BOUNDCHECK(k<x.nrows);
  CHARARGUMENTS(upperLower[0]=='u' || upperLower[0]=='U' || upperLower[0]=='l' || upperLower[0]=='L');
  DIMENSIONMATCH(ncols==A.nrows);
  DIMENSIONMATCH(ncols==x.ncols);
  dsymv_(upperLower, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+k, x.nrows, beta, vals+i, nrows);
}
void CMatrix::symvRowCol(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int j, double alpha, double beta, const char* upperLower)
{
  // Level 2 Blas operation, symmetric A,  y(i, :)' <- alpha A*x(:, j) + beta*y(i, :)';
  MATRIXPROPERTIES(A.isSymmetric());
  BOUNDCHECK(i<nrows);
  BOUNDCHECK(j<x.ncols);
  CHARARGUMENTS(upperLower[0]=='u' || upperLower[0]=='U' || upperLower[0]=='l' || upperLower[0]=='L');
  DIMENSIONMATCH(ncols==A.nrows);
  DIMENSIONMATCH(ncols==x.nrows);
  dsymv_(upperLower, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+j*x.nrows, 1, beta, vals+i, nrows);
}
void CMatrix::symvColCol(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* upperLower)
{
  // Level 2 Blas operation, symmetric A,  y(:, j) <- alpha A*x(:, k) + beta*y(:, j);
  DIMENSIONMATCH(A.isSymmetric());
  BOUNDCHECK(j<ncols);
  BOUNDCHECK(k<x.ncols);
  CHARARGUMENTS(upperLower[0]=='u' || upperLower[0]=='U' || upperLower[0]=='l' || upperLower[0]=='L');
  DIMENSIONMATCH(nrows==A.nrows);
  DIMENSIONMATCH(A.ncols==A.nrows);
  DIMENSIONMATCH(nrows==x.nrows);
  dsymv_(upperLower, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+k*x.nrows, 1, beta, vals+j*nrows, 1);
}
void CMatrix::symvColColOff(unsigned int j, int yr1, const CMatrix& A, const CMatrix& x, unsigned int k, int xr1, double alpha, double beta, const char* upperLower)
{
  // Level 2 Blas operation, symmetric A of NxN,  y(yr1:yr1+N, j) <- alpha A*x(xr1:xr1+N, k) + beta*y(yr1:yr1+N, j);
  // This version allows offsets into the x and y vectors.
  MATRIXPROPERTIES(A.isSymmetric());
  DIMENSIONMATCH(A.ncols==A.nrows);
  BOUNDCHECK(j<ncols);
  BOUNDCHECK(k<x.ncols);
  BOUNDCHECK(yr1+A.nrows<=nrows);
  BOUNDCHECK(xr1+A.nrows<=x.nrows);
  CHARARGUMENTS(upperLower[0]=='u' || upperLower[0]=='U' || upperLower[0]=='l' || upperLower[0]=='L');
  dsymv_(upperLower, A.ncols, alpha, A.vals, A.nrows, 
    x.vals+k*x.nrows+xr1, 1, beta, vals+j*nrows+yr1, 1);
}
void CMatrix::symvColRow(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned int i, double alpha, double beta, const char* upperLower)
{
  // Level 2 Blas operation, symmetric A,  y(:, j) <- alpha A*x(i, :)' + beta*y(:, j);
  MATRIXPROPERTIES(A.isSymmetric());
  BOUNDCHECK(j<ncols);
  BOUNDCHECK(i<x.nrows);
  CHARARGUMENTS(upperLower[0]=='u' || upperLower[0]=='U' || upperLower[0]=='l' || upperLower[0]=='L');
  DIMENSIONMATCH(nrows==A.nrows);
  DIMENSIONMATCH(nrows==x.ncols);
  dsymv_(upperLower, A.ncols, alpha, A.vals, A.nrows, 
	 x.vals+i, x.nrows, beta, vals+j*nrows, 1);
}  
  
void CMatrix::gemm(const CMatrix& A, const CMatrix& B, double alpha, double beta, const char* transa, const char* transb)
{
  // Level 3 Blas operation C <- alpha op(A)*op(B) + beta*C
  setSymmetric(false);
  unsigned int m = 0;
  unsigned int n = 0;
  unsigned int k = 0;
  switch(transa[0])
    {
    case 'n':
    case 'N':
      m = A.nrows;
      k = A.ncols;
      break;
    case 't':
    case 'T':
      m=A.ncols;
      k=A.nrows;
      break;
    default:
      CHARARGUMENTS(0);
    }
  switch(transb[0])
    {
    case 'n':
    case 'N':
      n = B.ncols;
      DIMENSIONMATCH(k==B.nrows);
      break;
    case 't':
    case 'T':
      n = B.nrows;
      DIMENSIONMATCH(k==B.ncols);
      break;
    default:
      CHARARGUMENTS(0);
    }	
  DIMENSIONMATCH(n==ncols);
  DIMENSIONMATCH(m==nrows);
  dgemm_(transa, transb, m, n, k, alpha, A.vals, A.nrows, 
	 B.vals, B.nrows, beta, vals, nrows);
  
}
void CMatrix::trmm(const CMatrix& A, double alpha, const char* side, const char* type, const char* trans, const char* diag)
{
  CHARARGUMENTS(side[0]=='L' || side[0]=='l' || side[0]=='R' || side[0]=='r');
  CHARARGUMENTS(type[0]=='L' || type[0]=='l' || type[0]=='U' || type[0]=='u');
  CHARARGUMENTS(trans[0]=='N' || trans[0]=='n' || trans[0]=='T' || trans[0]=='t');
  CHARARGUMENTS(diag[0]=='N' || diag[0]=='n' || diag[0]=='U' || diag[0]=='u');
  
  MATRIXPROPERTIES(A.isTriangular());
  switch(side[0])
    {
    case 'L':
    case 'l':
      DIMENSIONMATCH(A.nrows==nrows);
      break;
    case 'R':
    case 'r':
      DIMENSIONMATCH(A.nrows==ncols);
      break;
    default:
      throw ndlexceptions::MatrixError("No such value for side.");
    }
  
  dtrmm_(side, type, trans, diag, nrows, ncols, alpha, A.vals, A.nrows, vals, nrows);
}
void CMatrix::trsm(const CMatrix& A, double alpha, const char* side, const char* type, const char* trans, const char* diag)
{
  CHARARGUMENTS(side[0]=='L' || side[0]=='l' || side[0]=='R' || side[0]=='r');
  CHARARGUMENTS(type[0]=='L' || type[0]=='l' || type[0]=='U' || type[0]=='u');
  CHARARGUMENTS(trans[0]=='N' || trans[0]=='n' || trans[0]=='T' || trans[0]=='t');
  CHARARGUMENTS(diag[0]=='N' || diag[0]=='n' || diag[0]=='U' || diag[0]=='u');
  
  MATRIXPROPERTIES(A.isTriangular());
  switch(side[0])
    {
    case 'L':
    case 'l':
      DIMENSIONMATCH(A.nrows==nrows);
      break;
    case 'R':
    case 'r':
      DIMENSIONMATCH(A.nrows==ncols);
      break;
    default:
      throw ndlexceptions::MatrixError("No such value for side.");
    }
  
  dtrsm_(side, type, trans, diag, nrows, ncols, alpha, A.vals, A.nrows, vals, nrows);
}

void CMatrix::syrk(const CMatrix& A, double alpha, double beta, const char* type, const char* trans)
{
  //  C:=alpha*op(A)*op(A)' + beta*C.
  MATRIXPROPERTIES(isSymmetric() || beta==0.0);  
  unsigned int n = 0;
  unsigned int k = 0;
  switch(trans[0])
  {
  case 'n':
  case 'N':
    n = ncols;
    k = A.ncols;
    DIMENSIONMATCH(n==A.nrows);
    break;
  case 't':
  case 'T':
    n = nrows;
    k = A.nrows;
    DIMENSIONMATCH(n==A.ncols);
    break;
  default:
    CHARARGUMENTS(0);
  }
  dsyrk_(type, trans, n, k, alpha, A.vals, A.nrows, beta, vals, nrows);
  copySymmetric(type);
}
void CMatrix::sumRow(const CMatrix& A, double alpha, double beta)
{
  DIMENSIONMATCH(rowsMatch(A) && ncols==1 || ncols==A.nrows && nrows==1);
  for(unsigned int i=0; i<A.getRows(); i++)
  {
    double s = 0.0;
    for(unsigned int j=0; j<A.getCols(); j++)
    {
      s += A.getVal(i, j);
    }
    setVal(beta*getVal(i)+alpha*s, i);
  }
} 
void CMatrix::sumCol(const CMatrix& A, double alpha, double beta)
{
  DIMENSIONMATCH(colsMatch(A) && nrows==1 || nrows==A.ncols && ncols==1);
  for(unsigned int j=0; j<A.getCols(); j++)
  {
    double s = 0.0;
    for(unsigned int i=0; i<A.getRows(); i++)
    {
      s += A.getVal(i, j);
    }
    setVal(beta*getVal(j)+alpha*s, j);
  }
}

void CMatrix::copySymmetric(const char* type)
{
  switch(type[0]) 
    {
    case 'U':
    case 'u':
      for(unsigned int i=0; i<nrows; i++)
	for(unsigned int j=0; j<i; j++)
	  vals[i + nrows*j] = vals[j + ncols*i];
      break;
    case 'L':
    case 'l':
      for(unsigned int j=0; j<ncols; j++)
	for(unsigned int i=0; i<j; i++)
	  vals[i + nrows*j] = vals[j + ncols*i];
      break;
    default:
      CHARARGUMENTS(0);
    }
}

void CMatrix::potrf(const char* type)
{
  MATRIXPROPERTIES(isSymmetric());
  int info;
  dpotrf_(type, nrows, vals, ncols, info);
  setSymmetric(false);
  setTriangular(true);
  if(info!=0) throw ndlexceptions::MatrixNonPosDef();
}
void CMatrix::chol(const char* type)
{
  // type is either U or L for upper or lower triangular.
  MATRIXPROPERTIES(isSymmetric()); // matrix should be symmetric.
  potrf(type);
  switch((int)type[0]) {
  case 'L':
      for(unsigned int j=0; j<ncols; j++)
	for(unsigned int i=0; i<j; i++)
	  vals[i + nrows*j] = 0.0;
    break;
  case 'U':
      for(unsigned int i=0; i<nrows; i++)
	for(unsigned int j=0; j<i; j++)
	  vals[i + nrows*j] = 0.0;
    break;
  }
  setTriangular(true);    
}
void CMatrix::chol()
{
  MATRIXPROPERTIES(isSymmetric()); // matrix should be symmetric.
  chol("U");
}
double logDet(const CMatrix& U)
{
  MATRIXPROPERTIES(U.isTriangular()); // should be chol decomp
  double logDet = 0.0;
  for(unsigned int i=0; i<U.getRows(); i++)
    logDet+=std::log(U.getVal(i, i));
  logDet *= 2;
  return logDet;
}

void CMatrix::potri(const char* type)
{
  MATRIXPROPERTIES(isSquare());
  int info;
  dpotri_(type, nrows, vals, ncols, info);
  if(info!=0) throw ndlexceptions::MatrixNonPosDef();
}
void CMatrix::pdinv(const CMatrix& U)
{
  MATRIXPROPERTIES(U.isTriangular()); // U should be cholesky decomposition.
  MATRIXPROPERTIES(isSymmetric()); // matrix should be symmetric.
  deepCopy(U);
  potri("U");
  // make matrix symmetric.
  for(unsigned int i=0; i<nrows; i++)
    for(unsigned int j=0; j<i; j++)
      vals[i + nrows*j] = vals[j + ncols*i];
  setSymmetric(true);
}
void CMatrix::pdinv()
{
  MATRIXPROPERTIES(isSymmetric()); // matrix should be symmetric.
  potrf("U");
  potri("U");
  for(unsigned int i=0; i<nrows; i++)
    for(unsigned int j=0; j<i; j++)
      vals[i + nrows*j] = vals[j + ncols*i];
  setSymmetric(true);
}
int CMatrix::syev(CMatrix& eigVals, const char* jobz, const char* uplo, int lwork)
{
  CHARARGUMENTS(jobz[0]=='v' || jobz[0]=='V' || jobz[0]=='n' || jobz[0]=='N');
  CHARARGUMENTS(uplo[0]=='l' || uplo[0]=='L' || uplo[0]=='u' || uplo[0]=='U');
  MATRIXPROPERTIES(isSymmetric());
  DIMENSIONMATCH(eigVals.getNumElements()==nrows);
  if(lwork<3*ncols-1)
    lwork = 3*ncols-1;
  CMatrix work(1, lwork);
  int info=0;
#ifndef _NOSYSEV
  dsyev_(jobz, uplo, ncols,  vals,  nrows, eigVals.vals, work.vals, lwork, info);
  if(jobz[0]=='V' || jobz[0]=='v')
    setSymmetric(false);
  if(info>0)
    throw ndlexceptions::MatrixError("Eigendecomposition failed to converge.");
  else if(info<0)
  {
      string msg = "Argument: ";
      msg+=-info;
      msg+=" error in eigendecomposition.";
      throw ndlexceptions::Error(msg);
  }
#else // _NOSYSEV is defined
  throw ndlexceptions::Error("Not able to access lapack DSYEV routine on this machine.");
#endif
  return (int)work.getVal(0);
}

void CMatrix::lu()
{
  // TODO this isn't really properly implemented yet ... need to return ipiv somehow.
  MATRIXPROPERTIES(isSquare());
  int info;
  std::vector<int> ipiv(nrows);
  // TODO should really check for errors here.
  dgetrf_(nrows, ncols, vals, ncols, &ipiv[0], info);
  if(info!=0) throw ndlexceptions::MatrixConditionError();
}

void CMatrix::inv()
{
  MATRIXPROPERTIES(isSquare());
  // create output matrix by lu decomposition of input
    
  int length = nrows;
  std::vector<int> ipiv(length);
  int info = 0;
    
  dgetrf_(nrows, ncols, 
	  vals, ncols, &ipiv[0], info);
  if(info!=0) throw ndlexceptions::MatrixConditionError();
  int order = nrows;
  int lwork = order*16;
  std::vector<double> work(lwork);
  info = 0;
  dgetri_(order, vals, ncols, 
	  &ipiv[0], &work[0], lwork, info);
  // check for successful inverse
  if(info!=0) throw ndlexceptions::MatrixConditionError();
}
void CMatrix::rand(double a, double b) {
  double val;
  double span = b-a;
  for(unsigned int i=0; i<nrows*ncols; i++) {
    val = ndlutil::rand();
    vals[i]=val*span + a;
  }
}
void CMatrix::rand() {
  for(unsigned int i=0; i<nrows*ncols; i++) {
    vals[i] = ndlutil::rand();
  }
}
void CMatrix::randn(double var, double mean)
{
  double sd = sqrt(var);
  double val;
  for(unsigned int i=0; i<nrows*ncols; i++)
    {
      val = ndlutil::randn();
      vals[i]=val*sd+mean;
    }
}
double CMatrix::sum() const
{
  // matrix should be square
  double sum = 0.0;
  for(unsigned int i=0; i<nrows*ncols; i++)
    sum += vals[i];
  return sum; 
}
double CMatrix::trace() const
{
  // matrix should be square
  MATRIXPROPERTIES(isSquare());
  double tr = 0;
  for(unsigned int i=0; i<nrows; i++)
    tr += vals[i*nrows + i];
  return tr;
}
bool CMatrix::equals(const CMatrix& A, double tol) const
{
  if(nrows != A.nrows)
    return false;
  if(ncols != A.ncols)
    return false;
  if(maxAbsDiff(A)>tol)
    return false;
  return true;
}
double CMatrix::maxAbsDiff(const CMatrix& X) const
{
  DIMENSIONMATCH(dimensionsMatch(X));
  double max=0.0;
  double diff=0.0;
  for(unsigned int i=0; i<nrows*ncols; i++)
    {
      diff = abs(vals[i]-X.vals[i]);
      if(diff>max)
	max = diff;
    }
  return max;
} 

double CMatrix::max() const
{
  double max = vals[0];
  double val = 0.0;
  for(unsigned int i=1; i<nrows*ncols; i++)
    val = vals[i];
    if(val > max)
      max = val;
  return max;
}
void CMatrix::randn()
{
  randn(1.0, 0.0);
}
void CMatrix::negate()
{
  scale(-1.0);
}
void CMatrix::ones()
{
  for(unsigned int i=0; i<nrows*ncols; i++)
    {
      vals[i]=1.0;
    }
}
void CMatrix::diag(const CMatrix& d) 
{
  MATRIXPROPERTIES(isSquare());
  for(unsigned int i=0; i<nrows; i++)
    {
      for(unsigned int j=0; j<i; j++)
	{
	  vals[i + nrows*j] = 0.0;
	  vals[j + nrows*i] = 0.0;
	}
      vals[i + nrows*i] = d.getVal(i);
    }
  setSymmetric(true);
}
void CMatrix::diag(double d) 
{
  MATRIXPROPERTIES(isSquare());
  for(unsigned int i=0; i<nrows; i++)
    {
      for(unsigned int j=0; j<i; j++)
	{
	  vals[i + nrows*j] = 0.0;
	  vals[j + nrows*i] = 0.0;
	}
      vals[i + nrows*i] = d;
    }
  setSymmetric(true);
}
void CMatrix::eye() 
{
  MATRIXPROPERTIES(isSquare());
  for(unsigned int i=0; i<nrows; i++)
    {
      for(unsigned int j=0; j<i; j++)
	{
	  vals[i + nrows*j] = 0.0;
	  vals[j + nrows*i] = 0.0;
	}
      vals[i + nrows*i] = 1.0;
    }
  setSymmetric(true);
}
      

void CMatrix::zeros()
{
  for(unsigned int i=0; i<nrows*ncols; i++)
    {
      vals[i]=0.0;
    }
}
void CMatrix::memAllocate()
{
  // totalAlloc += nrows*ncols;
//   if(nrows*ncols > 10)
//     cout << "Allocating " << nrows*ncols << " memory. Total " << totalAlloc << endl;
  BOUNDCHECK(nrows>0);
  BOUNDCHECK(ncols>0);
  switch(allocation)
  {
  case NEW:
    vals = new double[nrows*ncols];
    if(vals==NULL)
      throw std::bad_alloc();
    break;
  case MALLOC:
    vals = (double *)malloc(sizeof(double)*nrows*ncols);
    if(vals==NULL)
      throw std::bad_alloc();
    break;
  default:
    throw ndlexceptions::NotImplementedError("Memory allocation method not known.");
  }
}
void CMatrix::memDeAllocate()
{
  // totalAlloc -= nrows*ncols;
//   if (nrows*ncols > 10)
//     cout << "Deallocating " << nrows*ncols << " memory. Total " << totalAlloc << endl;

  if(vals==NULL)
  {
    throw ndlexceptions::Error("De-allocation attempted before allocation");
  }
  switch(allocation)
  {
  case NEW:
    delete[] vals;
    break;
  case MALLOC:
    free(vals);
    break;
  case MATLAB:
    // do nothing MATLAB should handle it.
    break;
  default:
    throw ndlexceptions::NotImplementedError("Memory allocation method not known.");
  }
}
void CMatrix::memReAllocate(int rowIncrease, int colIncrease)
{
  SANITYCHECK(-rowIncrease<nrows);
  SANITYCHECK(-colIncrease<ncols);
  unsigned int newRows = nrows+rowIncrease;
  unsigned int newCols = ncols+colIncrease;
  unsigned int minRows = newRows;
  unsigned int minCols = newCols;
  double* newVals;
  
  if(nrows<minRows) minRows = nrows;
  if(ncols<minCols) minCols = ncols;
  SANITYCHECK(newRows>0);
  SANITYCHECK(newCols>0);
  switch(allocation)
  {
  case NEW:
    newVals = new double[newRows*newCols];
    if(newVals==NULL)
      throw std::bad_alloc();
    for(unsigned int i=0; i<minRows; i++)
      for(unsigned int j=0; j<minCols; j++)
	newVals[i + newRows*j] = vals[i + nrows*j];
    // set remaining elements to zero.
    for(unsigned int i=minRows; i<newRows; i++)
      for(unsigned int j=0; j<newCols; j++)
	newVals[i+newRows*j] = 0.0;
    for(unsigned int j=minCols; j<newCols; j++)
      for(unsigned int i=0; i<newRows; i++)
	newVals[i+newRows*j] = 0.0;
    delete []vals;
    vals = newVals;
    nrows=newRows;
    ncols=newCols;
    break;
  case MALLOC:
    break;
  case MATLAB:
    break;
  default:
    throw ndlexceptions::NotImplementedError("Memory allocation method not known.");
  }
}
void CMatrix::maxRow(CMatrix& m) const
{
  DIMENSIONMATCH(m.getRows()==1);
  DIMENSIONMATCH(m.getCols()==getCols());
  for(unsigned int j=0; j<getCols(); j++)
    {
      m.setVal(getVal(0, j), j);
      for(unsigned int i=1; i<getRows(); i++)
	{
	  double val=getVal(i, j);
	  if(val<m.getVal(0, j))
	    m.setVal(val, 0, j);
	}
    }
} 
void CMatrix::minRow(CMatrix& m) const
{
  DIMENSIONMATCH(m.getRows()==1);
  DIMENSIONMATCH(m.getCols()==getCols());
  for(unsigned int j=0; j<getCols(); j++)
    {
      m.setVal(getVal(0, j), j); 
      for(unsigned int i=1; i<getRows(); i++)
	{
	  double val=getVal(i, j);
	  if(val>m.getVal(0, j))
	    m.setVal(val, 0, j);
	}
    }
} 


double CMatrix::jitChol(CMatrix& A, unsigned int maxTries) 
{
  // this is an upper cholesky

  // set the jitter to be 1e-6 times the mean of diagonal.
  MATRIXPROPERTIES(A.isSquare());
  MATRIXPROPERTIES(A.isSymmetric());
  
  double jitter = 1e-6*A.trace()/(double)A.getRows();
  bool success = false;
  unsigned int tries = 0;
  while(!success && tries<maxTries)
  {
    try{
      deepCopy(A);
      chol();
      success = true;
    }
    catch(ndlexceptions::MatrixNonPosDef& e)
    {
      A.addDiag(jitter);
      jitter*=10;
      tries++;
      if(jitter>10)
	throw ndlexceptions::MatrixNonPosDef();
    }
    catch(...)
    {
      throw;
    }
  }
  if(tries>=maxTries)
  {
    cout << "Adding jitter failed after " << tries << " tries." << endl;
    throw ndlexceptions::MatrixNonPosDef();
  }
  return jitter;
}




ostream& operator<<(ostream& out, const CMatrix& A)
{
  for(unsigned int i = 0; i < A.getRows(); i++){
    for(unsigned int j = 0; j < A.getCols(); j++){
      if (A.getVal(i, j) > ndlutil::DISPEPS || A.getVal(i, j) < -ndlutil::DISPEPS)
        out << A.getVal(i, j) << " ";
      else
        out << 0 << " ";
    }
    out << endl;
  }
  return out;
}

CMatrix lu(const CMatrix& inMatrix) 
{
  
  CMatrix outMatrix(inMatrix);
  outMatrix.lu();
  return outMatrix;
}

CMatrix chol(const CMatrix& inMatrix) 
{
  
  CMatrix outMatrix(inMatrix);
  chol(outMatrix, inMatrix);
  return outMatrix;
}
void chol(CMatrix& outMatrix, const CMatrix& inMatrix) 
{
  
  DIMENSIONMATCH(outMatrix.dimensionsMatch(inMatrix));
  outMatrix.copy(inMatrix);
  outMatrix.chol("U");
}
CMatrix jitChol(const CMatrix& inMatrix, unsigned int maxTries = 10) 
{
  CMatrix outMatrix(inMatrix);
  jitChol(outMatrix, inMatrix, maxTries);
  return outMatrix;

}

double jitChol(CMatrix& outMatrix, const CMatrix& inMatrix, unsigned int maxTries = 10) 
{

  DIMENSIONMATCH(outMatrix.dimensionsMatch(inMatrix));
  double jitter = 0;
  outMatrix.copy(inMatrix);
  for(unsigned int i = 0; i<maxTries; i++)
  {
    try
    {
      if(jitter==0)
      {
	jitter = 1e-6*abs(inMatrix.trace()/inMatrix.getRows());
	outMatrix.chol("U");
	return jitter;
      }
      else
      {
	cout << "Warning, matrix is not positive definite in jitChol, adding " << jitter << " jitter";
	outMatrix.copy(inMatrix);
	outMatrix.addDiag(jitter);
	outMatrix.chol("U");
	return jitter;
      }
    }
    catch(ndlexceptions::MatrixNonPosDef& e)
    {
      bool nonPosDef = true;
      jitter *= 10.0;
      if(i==maxTries)
	throw ndlexceptions::MatrixNonPosDef();
    }
    catch(...)
    {
      throw;
    }

  }
}


CMatrix inv(const CMatrix& inMatrix)
{
  CMatrix outMatrix(inMatrix);
  outMatrix.inv();
  return outMatrix;
}

CMatrix multiply(const CMatrix& A, const CMatrix& B)
{
  CMatrix C(A.getRows(), B.getCols());
  C.gemm(A, B, 1.0, 0.0, "n", "n");
  return C;
}

CMatrix multiply(const CMatrix& A, const char* transa, const CMatrix& B, const char* transb)
{
  int n=0;
  switch(transa[0])
    {
    case 'n':
    case 'N':
      n = A.getRows();
      break;
    case 't':
    case 'T':
      n = A.getCols();
      break;
    default:
      CHARARGUMENTS(0);
    }
  int m = 0;
  switch(transb[0])
    {
    case 'n':
    case 'N':
      m = B.getCols();
      break;
    case 't':
    case 'T':
      m = B.getRows();
      break;
    default:
      CHARARGUMENTS(0);
    }
  CMatrix C(n, m);
  C.gemm(A, B, 1.0, 0.0, transa, transb);
  return C;
}
double trace(const CMatrix& A)
{
  return A.trace();
}
double sum(const CMatrix& A)
{
  return A.sum();
}
double max(const CMatrix& A)
{
  // matrix should be square
  return A.max();
}
CMatrix pdinv(const CMatrix& A)
{
  CMatrix B(A);
  B.pdinv();
  return B;
}

CMatrix sumRow(const CMatrix& A)
{
  CMatrix S(A.getRows(), 1);
  for(unsigned int i=0; i<A.getRows(); i++)
  {
    double s = 0.0;
    for(unsigned int j=0; j<A.getCols(); j++)
    {
      s += A.getVal(i, j);
    }
    S.setVal(s, i);
  }
  return S;
} 
CMatrix meanRow(const CMatrix& A)
{
  CMatrix M = sumRow(A);
  double numColsInv = 1.0/(double)A.getCols();
  for(unsigned int i=0; i<A.getRows(); i++)
    M.setVal(M.getVal(i, 0)*numColsInv, i, 0);
  return M;
}
CMatrix varRow(const CMatrix& A)
{
  CMatrix M = meanRow(A);
  double numColsInv = 1.0/(double)A.getCols();
  for(unsigned int i=0; i<A.getRows(); i++)
    M.setVal(A.norm2Row(i)*numColsInv - M.getVal(i, 0)*M.getVal(i, 0), i, 0);   
  return M;
}
CMatrix stdRow(const CMatrix& A)
{
  CMatrix v = varRow(A);
  for(unsigned int i=0; i<A.getRows(); i++)
    v.setVal(sqrt(v.getVal(i)), i);
  return v;
}
CMatrix sumCol(const CMatrix& A)
{
  CMatrix S(1, A.getCols());
  for(unsigned int j=0; j<A.getCols(); j++)
  {
    double s = 0;
    for(unsigned int i=0; i<A.getRows(); i++)
    {
      s += A.getVal(i,j);
    }
    S.setVal(s, j);
  }
  return S;
} 
CMatrix meanCol(const CMatrix& A)
{
  CMatrix M = sumCol(A);
  double numRowsInv = 1.0/(double)A.getRows();
  for(unsigned int j=0; j<A.getCols(); j++)
    M.setVal(M.getVal(0, j)*numRowsInv, 0, j);
  return M;
}
CMatrix varCol(const CMatrix& A)
{
  CMatrix M = meanCol(A);
  double numRowsInv = 1.0/(double)A.getRows();
  for(unsigned int j=0; j<A.getCols(); j++)
    M.setVal(A.norm2Col(j)*numRowsInv - M.getVal(0, j)*M.getVal(0, j), 0, j);   
  return M;
}
CMatrix stdCol(const CMatrix& A)
{
  CMatrix v = varCol(A);
  for(unsigned int i=0; i<A.getCols(); i++)
    v.setVal(sqrt(v.getVal(i)), i);
  return v;
}
CMatrix zeros(const int rows, const int cols) {
  CMatrix z(rows, cols);
  z.zeros();
  return z;
}
double randn(double mean, double var) {
  double sd = sqrt(var);
  double val = ndlutil::randn();
  val*=sd;
  val+=mean;
  return val;
}
double randn() {
  return ndlutil::randn();
}
// void swap(CMatrix& x, CMatrix& y) {
//   // Level 1 Blas operation y <-> x
//   DIMENSIONMATCH(y.ncols==x.ncols);
//   DIMENSIONMATCH(y.nrows==x.nrows);   
//   dswap_(x.ncols*x.nrows, x.vals, 1, y.vals, 1);
// }
void CMatrix::readParamsFromStream(istream& in) 
{
  string line;
  
  if(getBaseTypeStream(in)!="matrix")
    throw ndlexceptions::StreamFormatError("matrix", "Unexpected base type");
  if(getTypeStream(in)!="doubleMatrix")
    throw ndlexceptions::StreamFormatError("matrix", "Unexpected matrix type");
  int nrs = readIntFromStream(in, "numRows");
  int ncls = readIntFromStream(in, "numCols");
  resize(nrs, ncls);
  vector<string> tokens;
  for(unsigned int i = 0; i<getRows(); i++)
  {
    if(!ndlstrutil::getline(in, line))
      throw ndlexceptions::StreamFormatError("matrix", "Incorrect number of rows in matrix.");
    if(line[line.size()-1]=='\r')
      line.erase(line.size()-1);
    tokens.clear();
    ndlstrutil::tokenise(tokens, line);
    if(getCols()!=tokens.size())
      throw ndlexceptions::StreamFormatError("matrix", "Incorrect number of columns in row " + ndlstrutil::itoa(i) + " of matrix.");
    for(unsigned int j=0; j<tokens.size(); j++) 
    {
      int ind = tokens[j].find('.');
      if(ind==std::string::npos||ind<0)
	       setVal((double)atoi(tokens[j].c_str()), i, j);
      else
	       setVal(atof(tokens[j].c_str()), i, j);
    }
  }
}
void CMatrix::writeParamsToStream(ostream& out) const
{

  writeToStream(out, "baseType", "matrix");
  writeToStream(out, "type", "doubleMatrix");
  writeToStream(out, "numRows", getRows());
  writeToStream(out, "numCols", getCols());
  toUnheadedStream(out);
}
void CMatrix::fromUnheadedFile(const string fileName) {
  ifstream in(fileName.c_str());
  if(!in) throw ndlexceptions::FileReadError(fileName);
  try 
  {
      fromUnheadedStream(in);
  }
  catch(ndlexceptions::StreamFormatError err) 
  {
    throw ndlexceptions::FileFormatError(fileName, err);
  }
  in.close();
}
void CMatrix::fromUnheadedStream(istream& in) 
{
  string line;
  vector<string> tokens;
  unsigned int rowNo = 0;
  unsigned int allocateRows = ndlstrutil::ALLOCATECHUNK/getCols();
  while(getline(in, line)) 
  {
    if(line[line.size()-1]=='\r')
      line.erase(line.size()-1);
    if(line[0]=='#')
      continue;
    tokens.clear();
    ndlstrutil::tokenise(tokens, line);
    
    if(rowNo==0) 
    {
      int cols = tokens.size();
      if(getRows()==0)
	resize(allocateRows, cols);	  
    }
    
    if(rowNo>=getRows())
      resize(getRows()+allocateRows, getCols());
    if(tokens.size()!=getCols())
      throw ndlexceptions::StreamFormatError("numCols");
    for(unsigned int j=0; j<tokens.size(); j++) 
    {
      unsigned int ind = tokens[j].find('.');
      if(ind==std::string::npos||ind<0)
	setVal((double)atoi(tokens[j].c_str()), rowNo, j);
      else
	setVal(atof(tokens[j].c_str()), rowNo, j);
    }
    rowNo++;
  }
  resize(rowNo+1, getCols());
}

void CMatrix::toUnheadedFile(const string fileName, const string comment) const {
  ofstream out(fileName.c_str());
  if(!out) throw ndlexceptions::FileWriteError(fileName);
  if(comment.length()>0)
    out << "#" << comment << endl;
  toUnheadedStream(out);
  out.close();
}
void CMatrix::toUnheadedStream(ostream& out) const 
{
  for(unsigned int i = 0; i < getRows(); i++) 
  {
    for(unsigned int j = 0; j < getCols(); j++) 
    {
      double val = getVal(i, j);
      if((val - (int)val)==0.0)
	out << (int)val << " ";
      else
	out << val << " ";
    }
    out << endl;
  }
}
#ifdef _NDLMATLAB
mxArray* CMatrix::toMxArray() const 
{
  int dims[2];
  dims[0] = nrows;
  dims[1] = ncols;
  mxArray* matlabArray = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  double* matlabVals = mxGetPr(matlabArray);
  dcopy_(ncols*nrows, vals, 1, matlabVals, 1);
  return matlabArray;
}
void CMatrix::fromMxArray(const mxArray* matlabArray) 
{
  // TODO implement true sparse matrices to handle this.
  if(mxIsSparse(matlabArray)) 
  {
    fromSparseMxArray(matlabArray);
  }
  else 
  {
    fromFullMxArray(matlabArray);
  }
}
void CMatrix::fromFullMxArray(const mxArray* matlabArray) 
{
  SANITYCHECK(!mxIsSparse(matlabArray));
  if(mxGetClassID(matlabArray) != mxDOUBLE_CLASS) 
  {
    throw ndlexceptions::Error("mxArray is not a double matrix.");
  }
  if(mxGetNumberOfDimensions(matlabArray) != 2) 
  {
    throw ndlexceptions::Error("mxArray does not have 2 dimensions.");
  }
  const int* dims = mxGetDimensions(matlabArray);
  resize(dims[0], dims[1]);
  double* matlabVals = mxGetPr(matlabArray);
  dcopy_(ncols*nrows, matlabVals, 1, vals, 1);
}
void CMatrix::fromSparseMxArray(const mxArray* matlabArray) 
{
  SANITYCHECK(mxIsSparse(matlabArray));
  if(mxGetClassID(matlabArray) != mxDOUBLE_CLASS) 
  {
    throw ndlexceptions::Error("mxArray is not a double matrix.");
  }
  if(mxGetNumberOfDimensions(matlabArray) != 2) 
  {
    throw ndlexceptions::Error("mxArray does not have 2 dimensions.");
  }
  const int* dims = mxGetDimensions(matlabArray);
  resize(dims[0], dims[1]);
  double* matlabVals = mxGetPr(matlabArray);
  setVals(0.0);
  int* matlabIr = mxGetIr(matlabArray);
  int* matlabJc = mxGetJc(matlabArray);
  int nnz = matlabJc[getCols()];
  for(int j=0; j<getCols(); j++) 
  {
    for(int i=matlabJc[j]; i<matlabJc[j+1]; i++) 
    {
      setVal(matlabVals[i], matlabIr[i], j);
    } 
  }
} 
#endif   

#ifdef _HDF5
    void CMatrix::writeToHdf5( const std::string& filename, const std::string& path_to_dataset ) const
    {
        hid_t file_id = open_hdf5_file( filename );
        try
        {
            hid_t group_id = get_parent_h5_obj( file_id, path_to_dataset );
            std::vector<std::string> path_comp = split_path( path_to_dataset );
            
            std::vector<size_t> shape(2);
            shape[0] = nrows; shape[1] = ncols;

            array_to_hdf5( group_id, path_comp.back().c_str(), vals, shape, "CMatrix", COLUMN_MAJOR );
        }
        catch( ... )
        {
            close_hdf5_file( file_id );
        }        
    }

    void CMatrix::readFromHdf5( const std::string& filename, const std::string& path_to_dataset, bool transpose_ )
    {
        hid_t file_id = open_hdf5_file( filename );
        try
        {
            hid_t group_id = get_parent_h5_obj( file_id, path_to_dataset );
            std::vector<std::string> path_comp = split_path( path_to_dataset );
            
            std::vector<size_t> shape = get_dataset_shape( group_id, path_comp.back().c_str() );
            resize( shape[1], shape[0] );
            array_from_hdf5( group_id, path_comp.back().c_str(), vals );
            
            //required when reading dataset stored from a row-major matrix
            if( transpose_ )
                trans();
        }
        catch( ... )
        {
            close_hdf5_file( file_id );
        }       
    }
#endif

