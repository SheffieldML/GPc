#ifndef CMATRIX_H
#define CMATRIX_H
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include "ndlassert.h"
#include "CNdlInterfaces.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include "ndlstrutil.h"
#include "lapack.h"

#ifdef _HDF5
    #include <hdf5/hdf5_io.hpp> //in mlprojects/branches
#endif


//using namespace std;


// Base matrix class that acts as an interface to LAPACK and BLAS.
class CMatrix : public CMatInterface, public CStreamInterface
{
public:
  // The default constructor.
  CMatrix()
  {
    _init();
    nrows = 1;
    ncols = 1;
    symmetric = false;
    triangular = false;
    memAllocate();
    vals[0] = 0;
      
  }
  // A constructor for creating a 1x1 CMatrix from a double.
  CMatrix(double val)
  {
    _init();
    nrows = 1;
    ncols = 1;
    symmetric = false;
    triangular = false;
    memAllocate();
    vals[0] = val;
  }
  // The standard memory allocating constructor for creating a matrix o f size numRows*numCols.
  CMatrix(unsigned int numRows, unsigned int numCols) : nrows(numRows), ncols(numCols)
  {
    _init();
    symmetric = false;
    triangular = false;
    memAllocate();
  }
  // Constructor which allocates memory and then fills the CMatrix with constant values.
  CMatrix(unsigned int numRows, unsigned int numCols, double val) : nrows(numRows), ncols(numCols)
  {
    _init();
    symmetric = false;
    triangular = false;
    memAllocate();
    setVals(val);
  }
  // Constructor for initialising a CMatrix from a double* array.
  CMatrix(unsigned int numRows, unsigned int numCols, double* inVals) : nrows(numRows), ncols(numCols)
  {
    _init();
    symmetric = false;
    triangular = false;
    memAllocate();
    dcopy_(nrows*ncols, inVals, 1, vals, 1);
  }
  // Constructor for initialising a CMatrix from a double* array for use with SWIG.
 CMatrix(double* pythonInVals, int numRows, int numCols) : nrows(numCols), ncols(numRows)
  {
    BOUNDCHECK(numRows>0);
    BOUNDCHECK(numCols>0);
    // for numpy storeage --- first copy memory then transpose. 
    _init();
    symmetric = false;
    triangular = false;
    memAllocate();
    dcopy_(nrows*ncols, pythonInVals, 1, vals, 1);
    trans();
  }
  
  // This was designed for interfacing with SWIG/Python, but for some reason it seems to cause a memory leak.
  // Constructor for initialising a CMatrix from rows of double* array.
  /*  CMatrix(double* inVals, int numRows, int numCols, int* indexVals, int numIndices) : nrows(numIndices), ncols(numCols) */
/*   { */
/*     _init(); */
/*     symmetric = false; */
/*     triangular = false; */
/*     memAllocate(); */
/*     for(unsigned int i=0; i<numIndices; i++) */
/*       for(unsigned int j=0; j<ncols; j++) */
/*         vals[i+nrows*j] = inVals[indexVals[i]+nrows*j]; */
/*   } */
  
  // Constructor for initialising a CMatrix from a double** array.
/*  CMatrix(unsigned int numRows, unsigned int numCols, double** inVals) : nrows(numRows), ncols(numCols) */
/*   { */
/*     _init(); */
/*     symmetric = false; */
/*     triangular = false; */
/*     memAllocate(); */
/*     for(unsigned int i=0; i<nrows; i++) */
/*       for(unsigned int j=0; j<ncols; j++) */
/*         vals[i+nrows*j] = inVals[i][j]; */
/*   } */
 CMatrix(unsigned int numRows, unsigned int numCols, vector<double>& inVals) : nrows(numRows), ncols(numCols)
  {
    _init();
    DIMENSIONMATCH(numRows*numCols==inVals.size());
    symmetric = false;
    triangular = false;
    memAllocate();
    for(unsigned int i=0; i<inVals.size(); i++)
      vals[i] = inVals[i];
  }
	
  // Constructor for special initialisations such as identity or random matrices.
  CMatrix(unsigned int numRows, unsigned int numCols, char type) : nrows(numRows), ncols(numCols)
  {
    _init();
    setSymmetric(false);
    setSymmetric(false);
    memAllocate();
    switch(type) {
    case 'I':
      // the identity
      DIMENSIONMATCH(numRows==numCols);
      for(unsigned int i=0; i<nrows; i++)
        for(unsigned int j=0; j<ncols; j++)
          if(i==j)
            vals[i+nrows*j] = 1.0;
          else
            vals[i+nrows*j] = 0.0;
      setSymmetric(true);
      break;
    default:
      SANITYCHECK(0);
    }
  }
  // Constructor for special initialisations where a value is also passed.
  CMatrix(unsigned int numRows, unsigned int numCols, char type, double val) : nrows(numRows), ncols(numCols)
  {
    _init();
    setSymmetric(false);
    setSymmetric(false);
    memAllocate();
    switch(type) {
    case 'S':
      // a spherical covariance matrix
      for(unsigned int i=0; i<nrows; i++)
        for(unsigned int j=0; j<ncols; j++)
          if(i==j)
            vals[i+nrows*j] = val;
          else
            vals[i+nrows*j] = 0.0;
      setSymmetric(true);
      break;
    default:
      SANITYCHECK(0);
    }
  }
 CMatrix(const CMatrix& A, vector<unsigned int> indices) : nrows(indices.size()), ncols(1), symmetric(false), triangular(false)
  {
    _init();
    memAllocate();
    for(unsigned int i =0; i<indices.size(); i++)
    {
      setVal(A.getVal(indices[i]), i);
    }
  }
 CMatrix(const CMatrix& A, 
	 vector<unsigned int> rowIndices, 
	 vector<unsigned int> colIndices) : 
  nrows(rowIndices.size()), 
  ncols(colIndices.size()), 
  symmetric(false), triangular(false)
  {
    _init();
    memAllocate();
    for(unsigned int i =0; i<rowIndices.size(); i++)
    {
      for(unsigned int j =0; j<colIndices.size(); j++)
      {
	setVal(A.getVal(rowIndices[i], colIndices[j]), i, j);
      }
    }
  }
  // The copy constructor, it performs a deep copy.
  CMatrix(const CMatrix& A) : nrows(A.nrows), ncols(A.ncols), symmetric(A.symmetric), triangular(A.triangular)
  {
    _init();
    memAllocate();
    copy(A);
      
  }
  // The class destructor, it deallocates the memory.
  virtual ~CMatrix()
  {
    memDeAllocate();
  }
  // Perform a deep copy of the matrix A, resizing if necessary.
  void deepCopy(const CMatrix& A)
  {
    resize(A.nrows, A.ncols);
    copy(A);
  }
  // Get the number of rows in the matrix.
  inline unsigned int getRows() const
  {
    return nrows;
  }
  // Get the number of columns in the matrix.
  inline unsigned int getCols() const
  {
    return ncols;
  }
  // Get the number of elements in the matrix (rows*cols).
  inline unsigned int getNumElements() const
  {
    return nrows*ncols;
  }
  // Return pointer to the raw column-major storage for the matrix
  inline double* getVals() 
  {
    return vals;
  }
  inline const double* getVals() const
  {
    return vals;
  }
  vector<double> getVector() const
  {
    vector<double> valVector;
    for(unsigned int i=0; i<getNumElements(); i++)
    {
      valVector.push_back(vals[i]);
    }
    return valVector;
  }
  // Get the matrix element in the ith row and jth column (indexing from 0).
  inline double getVal(unsigned int i, unsigned int j) const
  {
    #ifdef _NDLPYTHON
    if(!(i<nrows))
      throw std::range_error("getVal: column index out of range, maximum is " + ndlstrutil::itoa(nrows) + " value was " + ndlstrutil::itoa(i));
    if(!(j<ncols))
    {
      throw std::range_error("getVal: column index out of range, maximum is " + ndlstrutil::itoa(ncols) + " value was " + ndlstrutil::itoa(j));
    }
    #else
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(j<ncols);
    #endif
    return vals[i + nrows*j];
  } 
  // Get the ith element from the matrix.
  inline double getVal(unsigned int i) const
  {
    #ifdef _NDLPYTHON
    if(!(i<nrows*ncols))
    {
      throw std::range_error("getVal: matrix index out of range, maximum is " + ndlstrutil::itoa(nrows*ncols) + " value was " + ndlstrutil::itoa(i));
    } 
    #else
    BOUNDCHECK(i<nrows*ncols);
    #endif
    return vals[i];
  }
  // Set all elements of the matrix to val.
  inline void setVals(double val)
  {
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] = val;
  }
  // Set the ith element of the matrix to val.
  inline void setVal(double val, unsigned int i)
  {
    #ifdef _NDLPYTHON
    if(i>=nrows*ncols)
      throw std::range_error("setVal: matrix index out of range");
    #else
    BOUNDCHECK(i<nrows*ncols);
    #endif
    vals[i] = val;
  }
  // Add val to the ith element of the matrix.
  inline void addVal(double val, unsigned int i)
  {
    #ifdef _NDLPYTHON
    if(i>=nrows*ncols)
      throw std::range_error("addVal: matrix index out of range");
    #else
    BOUNDCHECK(i<nrows*ncols);
    #endif
    vals[i] += val;
  }
  // Set the matrix element from the ith row and jth column to val.
  inline void setVal(double val, unsigned int i, unsigned int j)
  {
    #ifdef _NDLPYTHON
    if(i>=nrows)
      throw std::range_error("setVal: row index out of range");
    if(j>=ncols)
      throw std::range_error("setVal: column index out of range");
    #else
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(j<ncols);
    #endif
    vals[i + nrows*j] = val;
  }
  // Add val to the matrix element from the ith row and jth column.
  inline void addVal(double val, unsigned int i, unsigned int j)
  {
    #ifdef _NDLPYTHON
    if(i>=nrows)
      throw std::range_error("addVal: row index out of range");
    if(j>=ncols)
      throw std::range_error("addVal: column index out of range");
    #else
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(j<ncols);
    #endif
    vals[i + nrows*j] += val;
  }
  bool isAnyNan() const
  {
    for(unsigned int i=0; i<nrows*ncols; i++)
	{
	  if(isnan(vals[i]))
		  return true;
	}
	return false;
  }
  bool isAnyInf() const
  {
    for(unsigned int i=0; i<nrows*ncols; i++)
	{
		if(isinf(vals[i]))
	    return true;
	}
	return false;
  }
  // Returns true if the matrix has the same number of rows as columns.
  inline const bool isSquare() const
  {
    return nrows==ncols;
  }
  // Returns true if matrix is triangular.
  inline const bool isTriangular() const
  {
    return triangular;
  }
  // Sets whether or not matrix is triangular.
  inline void setTriangular(const bool val) 
  {
    MATRIXPROPERTIES((val && isSquare()) || !val);	
    triangular=val;
  }
  // Returns true if the matrix is symmetric.
  inline const bool isSymmetric() const
  {
    return symmetric;
  }
  // Sets whether or not matrix is symmetric.
  inline void setSymmetric(const bool val) 
  {
    MATRIXPROPERTIES((val && isSquare()) || !val);	
    symmetric=val;
  }
  // Returns true if the matrix A has the same dimensions as the matrix.
  inline const bool dimensionsMatch(const CMatrix& A) const
  {
    return (nrows==A.nrows && ncols==A.ncols);
  }
  // Returns true if A has the same number of rows as the matrix.
  inline const bool rowsMatch(const CMatrix& A) const
  {
    return (nrows==A.nrows);
  }
  // Returns true if A has the same number of columns as the matrix.
  inline const bool colsMatch(const CMatrix& A) const
  {
    return (ncols==A.ncols);
  }
  // Add the columns of the matrix to this matrix (should have same number of columns).
  void sumCol(const CMatrix& A, double alpha, double beta);
  // Add the rows of the matrix to this matrix (should have same number of rows).
  void sumRow(const CMatrix& A, double alpha, double beta);
  // copy the upper part to the lower or vice versa.
  void copySymmetric(const char* type);
  void copyRowRow(unsigned int i, const CMatrix& X, unsigned int k);
  void copyColCol(unsigned int j, const CMatrix& X, unsigned int k);
  // Scale the matrix by a constant alpha.
  void scale(double alpha)
  {
    dscal_(nrows*ncols, alpha, vals, 1); 
  }
  // Scale the jth column of the matrix.
  void scaleCol(unsigned int j, double alpha)
  {
    BOUNDCHECK(j<ncols);
    dscal_(nrows, alpha, vals+j*nrows, 1);
  }
  // Scale the ith row of the matrix.
  void scaleRow(unsigned int i, double alpha)
  {
    BOUNDCHECK(i<nrows);
    dscal_(ncols, alpha, vals+i, nrows);
  }
  // Level 1 BLAS axpy  y c:= alpha x + y
  void axpy(const CMatrix& x, double alpha)
  {
    DIMENSIONMATCH(x.ncols==ncols);
    DIMENSIONMATCH(x.nrows==nrows);
    daxpy_(ncols*nrows, alpha, x.vals, 1, vals, 1);
  }
  // Level 1 BLAS axpy y(i, :) := alpha*x(k, :) + y(i, :);
  void axpyRowRow(unsigned int i, const CMatrix& x, unsigned int k, double alpha)
  {
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(k<x.nrows);
    DIMENSIONMATCH(x.ncols==ncols);
    daxpy_(ncols, alpha, x.vals+k, x.nrows, vals+i, nrows);
  }
  // Level 1 BLAS axpy (i, :) := alpha*x(:, j)' + y(i, :);
  void axpyRowCol(unsigned int i, const CMatrix& x, unsigned int j, double alpha)
  {
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(j<x.ncols);
    DIMENSIONMATCH(x.nrows==ncols);
    daxpy_(ncols, alpha, x.vals+j*x.nrows, 1, vals+i, nrows);
  }
  // Level 1 BLAS axpy (:, j) := alpha*x(:, k) + y(:, j);
  void axpyColCol(unsigned int j, const CMatrix& x, unsigned int k, double alpha)
  {
    BOUNDCHECK(j<ncols);
    BOUNDCHECK(k<x.ncols);
    DIMENSIONMATCH(x.nrows==nrows);
    daxpy_(nrows, alpha, x.vals+k*x.nrows, 1, vals+j*nrows, 1);
  }
  // Level 1 BLAS axpy (:, j) = alpha*x(i, :)' + y(:, j);
  void axpyColRow(unsigned int j, const CMatrix& x, unsigned int i, double alpha)
  {
    BOUNDCHECK(j<ncols);
    BOUNDCHECK(i<x.nrows);
    DIMENSIONMATCH(x.ncols==nrows);
    daxpy_(nrows, alpha, x.vals+i, x.nrows, vals+j*nrows, 1);
  }
  // Level 1 BLAS axpy diag(Y) = diag(Y) + alpha*x(i, :)'
  void axpyDiagRow(const CMatrix& x, unsigned int i, double alpha)
  {
    MATRIXPROPERTIES(isSquare());
    BOUNDCHECK(i<x.nrows);
    DIMENSIONMATCH(x.ncols==nrows);
    daxpy_(nrows, alpha, x.vals+i, x.nrows, vals, nrows+1);
  }
  // Level 1 BLAS axpy diag(Y) = diag(Y) + alpha*x(i, :)'
  void axpyDiagCol(const CMatrix& x, unsigned int j, double alpha)
  {
    MATRIXPROPERTIES(isSquare());
    BOUNDCHECK(j<x.ncols);
    DIMENSIONMATCH(x.nrows==nrows);
    daxpy_(nrows, alpha, x.vals+j*x.nrows, 1, vals, nrows+1);
  }
  // Level 2 BLAS Rank 1 update: ger, A = alpha*x*y' + A; 
  void ger(const CMatrix& x, const CMatrix& y, double alpha)
  {
    DIMENSIONMATCH(x.ncols==1);
    DIMENSIONMATCH(y.ncols==1);
    DIMENSIONMATCH(x.nrows==nrows);
    DIMENSIONMATCH(y.nrows==ncols);
    dger_(nrows, ncols, alpha, x.vals, 1, y.vals, 1, vals, nrows);
  }
  // Level 2 BLAS Rank 1 update: A := alpha*x(k, :)'*y(i, :) + A;
  void gerRowRow(const CMatrix& x, unsigned int i, const CMatrix& y, unsigned int k, double alpha)
  {
    BOUNDCHECK(i<x.nrows);
    BOUNDCHECK(k<y.nrows);
    DIMENSIONMATCH(x.ncols==nrows);
    DIMENSIONMATCH(y.ncols==ncols);
    dger_(nrows, ncols, alpha, x.vals+i, x.nrows, y.vals+k, y.nrows, vals, nrows);
  }
  // Level 2 BLAS Rank 1 update: A := alpha*x(:, j)*y(i, :) + A;
  void gerRowCol(const CMatrix& x, unsigned int i, const CMatrix& y, unsigned int j, double alpha)
  {
    BOUNDCHECK(i<x.nrows);
    BOUNDCHECK(j<y.ncols);
    DIMENSIONMATCH(x.ncols==nrows);
    DIMENSIONMATCH(y.nrows==ncols);
    dger_(nrows, ncols, alpha, x.vals+i, x.nrows, y.vals+j*y.nrows, 1, vals, nrows);
  }
  // Level 2 BLAS Rank 1 update: A := alpha*x(:, k)*y(:, j)' + A;
  void gerColCol(const CMatrix& x, unsigned int j, const CMatrix& y, unsigned int k, double alpha)
  {
    BOUNDCHECK(j<x.ncols);
    BOUNDCHECK(k<y.ncols);
    DIMENSIONMATCH(x.nrows==nrows);
    DIMENSIONMATCH(y.nrows==ncols);
    dger_(nrows, ncols, alpha, x.vals+j*x.nrows, 1, y.vals+k*y.nrows, 1, vals, nrows);
  }
  // Level 2 BLAS Rank 1 update: A := alpha*x(i, :)'x(:, j)' + A;
  void gerColRow(const CMatrix& x, unsigned int j, const CMatrix& y, unsigned int i, double alpha)
  {
    BOUNDCHECK(j<x.ncols);
    BOUNDCHECK(i<y.nrows);
    DIMENSIONMATCH(x.nrows==nrows);
    DIMENSIONMATCH(y.ncols==ncols);
    dger_(nrows, ncols, alpha, x.vals+j*x.nrows, 1, y.vals+i, y.nrows, vals, nrows);
  }

  // Level 2 BLAS Rank 1 update: syr, A = alpha*x*x' + A; 
  void syr(const CMatrix& x, double alpha, const char* type)
  {
    MATRIXPROPERTIES(isSymmetric());
    DIMENSIONMATCH(x.ncols==1);
    DIMENSIONMATCH(x.nrows==nrows);
    dsyr_(type, nrows, alpha, x.vals, 1, vals, nrows);
    copySymmetric(type);
  }
  // Level 2 BLAS Rank 1 update: A := alpha*x(i, :)'*x(i, :) + A;
  void syrRow(const CMatrix& x, unsigned int i, double alpha, const char* type)
  {
    MATRIXPROPERTIES(isSymmetric());
    BOUNDCHECK(i<x.nrows);
    DIMENSIONMATCH(x.ncols==nrows);
    dsyr_(type, nrows, alpha, x.vals+i, x.nrows, vals, nrows);
    copySymmetric(type);
  }
  // Level 2 BLAS Rank 1 update: A := alpha*x(:, j)x(:, j)' + A;
  void syrCol(const CMatrix& x, unsigned int j, double alpha, const char* type)
  {
    MATRIXPROPERTIES(isSymmetric());
    BOUNDCHECK(j<x.ncols);
    DIMENSIONMATCH(x.ncols==nrows);
    dsyr_(type, nrows, alpha, x.vals+j*x.nrows, 1, vals, nrows);
    copySymmetric(type);
  }

  // Return the euclidean distance squared between two row vectors.
  double dist2Row(unsigned int i, const CMatrix& A, unsigned int k) const
  {
    DIMENSIONMATCH(ncols==A.ncols);
    BOUNDCHECK(k<A.nrows);
    BOUNDCHECK(i<nrows);
    // |X-Y|^2 == |X|^2 + |Y|^2 - 2 X dot Y
    return norm2Row(i) + A.norm2Row(k) - 2.0*dotRowRow(i, A, k);
    // WVB's approach
    // double val = 0;
    // double *v1 = vals+i;
    // double *v2 = A.vals+k;
    // for (int i=0; i<ncols; i++,v1+=nrows,v2+=A.nrows) {
    //  double d = *v1 - *v2;
    //  val += d*d;
    //}
    // return val;  
  }
  // Return the euclidean distance squared between two column vectors.
  double dist2Col(unsigned int j, const CMatrix& A, unsigned int k) const
  {
    DIMENSIONMATCH(nrows==A.nrows);
    BOUNDCHECK(k<A.ncols);
    BOUNDCHECK(j<ncols);
    return norm2Col(j) + A.norm2Col(k) - 2.0*dotColCol(j, A, k);
  }
  // Return the norm of the ith row of the matrix.
  double normRow(unsigned int i) const
  {
    BOUNDCHECK(i<nrows);
    return dnrm2_(ncols, vals+i, nrows);
  }
  // Return the squared norm of the ith row of the matrix.
  double norm2Row(unsigned int i) const
  {
    BOUNDCHECK(i<nrows);
    double val=dnrm2_(ncols, vals+i, nrows);
    return val*val;
    // WVB's approach
    // return ddot_(ncols,vals+i,nrows,vals+i,nrows);
  }
  // Return the norm of the jth column of the matrix.
  double normCol(unsigned int j) const
  {
    BOUNDCHECK(j<ncols);
    return dnrm2_(nrows, vals+j*nrows, 1);
  }
  // Return the squared norm of the jth column of the matrix.
  double norm2Col(unsigned int j) const
  {
    BOUNDCHECK(j<ncols);
    double val=dnrm2_(nrows, vals+j*nrows, 1);
    return val*val;
    // WVB's approach
    // return ddot_(nrows,vals+j*nrows,1,vals+j*nrows,1);
  }
  // Return the inner product between the ith row of the matrix and the kth row of A.
  double dotRowRow(unsigned int i, const CMatrix& A, unsigned int k) const
  {
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(k<A.nrows);
    return ddot_(ncols, A.vals+k, A.nrows, vals+i, nrows);
  
  }
  // Return the inner product between the ith row of the matrix and the jth column of A.
  double dotRowCol(unsigned int i, const CMatrix& A, unsigned int j) const
  {
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(j<A.ncols);
    return ddot_(ncols, A.vals+j*A.nrows, 1, vals+i, nrows);
  
  }
  // Return the inner product between the jth column of the matrix and the kth column of A.
  double dotColCol(unsigned int j, const CMatrix& A, unsigned int k) const
  {
    BOUNDCHECK(j<ncols);
    BOUNDCHECK(k<A.ncols);
    return ddot_(nrows, A.vals+k*A.nrows, 1, vals+j*nrows, 1);
  }
  // Return the inner product between the jth column of the matrix and the ith row of A.
  double dotColRow(unsigned int j, const CMatrix& A, unsigned int i) const
  {
    BOUNDCHECK(j<ncols);
    BOUNDCHECK(i<A.nrows);
    return ddot_(nrows, A.vals+i, A.nrows, vals+j*nrows, 1);
  }
  // Swap the jth and the kth columns of the matrix.
  void swapCols(unsigned int j, unsigned int k)
  {
    BOUNDCHECK(j<ncols);
    BOUNDCHECK(k<ncols);
    if(j!=k)
      dswap_(nrows, vals+j*nrows, 1, vals+k*nrows, 1);
  }
  // Swap the ith and the kth rows of the matrix.
  void swapRows(unsigned int i, unsigned int k)
  {
    BOUNDCHECK(i<nrows);
    BOUNDCHECK(k<nrows);
    if(i!=k)
      dswap_(ncols, vals+i, nrows, vals+k, nrows);
  }
  // Add columns from A to the end of the matrix.
  void appendCols(const CMatrix& A)
  {
    DIMENSIONMATCH(rowsMatch(A));
    int origNcols = ncols;
    memReAllocate(0, A.ncols);
    setMatrix(0, origNcols, A);
  }
  // Add rows from A to the end of the matrix.
  void appendRows(const CMatrix& A)
  {
    DIMENSIONMATCH(colsMatch(A));
    int origNrows = nrows;
    memReAllocate(A.nrows, 0);
    setMatrix(origNrows, 0, A);
  }
  // Get the rows firstRow:lastRow and columns firstCol:lastCol and place in a  matrix C.
  void getMatrix(CMatrix& matrixOut, unsigned int firstRow, unsigned int lastRow, unsigned int firstCol, unsigned int lastCol) const
  {
    BOUNDCHECK(firstRow<=lastRow && lastRow<nrows);
    BOUNDCHECK(firstCol<=lastCol && lastCol<ncols);
    DIMENSIONMATCH(matrixOut.nrows==lastRow-firstRow+1 && matrixOut.ncols==lastCol-firstCol+1);
    for(unsigned int j=0; j<matrixOut.ncols; j++)
      for(unsigned int i=0; i<matrixOut.nrows; i++)
        matrixOut.vals[i+matrixOut.nrows*j] = vals[i+firstRow + nrows*(j+firstCol)];
  }
  // Get the rows in rows and columns firstCol:lastCol and place in a matrix matrixOut.
  void getMatrix(CMatrix& matrixOut, vector<unsigned int> rows, unsigned int firstCol, unsigned int lastCol)
  {
    BOUNDCHECK(firstCol<=lastCol && lastCol<ncols);
    DIMENSIONMATCH(matrixOut.nrows==rows.size() && matrixOut.ncols==lastCol-firstCol+1);
    for(unsigned int i=0; i<matrixOut.nrows; i++)
	{
	  BOUNDCHECK(rows[i]<nrows);
	  for(unsigned int j=0; j<matrixOut.ncols; j++)
	    matrixOut.vals[i+matrixOut.nrows*j] = vals[rows[i] + nrows*(j+firstCol)];
	}
  }
  // Get the rows firstRow:lastRow and columns in cols and place in matrix matrixOut.
  void getMatrix(CMatrix& matrixOut, unsigned int firstRow, unsigned int lastRow, vector<unsigned int> cols)
  {
    BOUNDCHECK(firstRow<=lastRow && lastRow<nrows);
    DIMENSIONMATCH(matrixOut.nrows==lastRow-firstRow+1 && matrixOut.ncols==cols.size());
    for(unsigned int j=0; j<matrixOut.ncols; j++)
	{
	  BOUNDCHECK(cols[j]<ncols);
	  for(unsigned int i=0; i<matrixOut.nrows; i++)
	    matrixOut.vals[i+matrixOut.nrows*j] = vals[i+firstRow + nrows*(cols[j])];
	}
  }
  // Get the rows from rows and columns from cols and place in matrix C.
  void getMatrix(CMatrix& matrixOut, vector<unsigned int> rows, vector<unsigned int> cols)
  {
    DIMENSIONMATCH(matrixOut.nrows==rows.size() && matrixOut.ncols==cols.size());
    for(unsigned int i=0; i<matrixOut.nrows; i++)
	{
	  BOUNDCHECK(rows[i]<matrixOut.nrows);
	  for(unsigned int j=0; j<matrixOut.ncols; j++)
      {
        BOUNDCHECK(cols[j]<ncols);
        matrixOut.vals[i+matrixOut.nrows*j] = vals[rows[i] + nrows*(cols[j])];
      }
	}
  }
  // Place A's first row and column at row, col and the rest of the matrix follows.
  void setMatrix(unsigned int row, unsigned int col, const CMatrix& A)
  {
    BOUNDCHECK(row+A.nrows <= nrows);
    BOUNDCHECK(col+A.ncols <= ncols);
    for(unsigned int i=0; i<A.nrows; i++)
      for(unsigned int j=0; j<A.ncols; j++)
        vals[i+row+nrows*(j+col)] = A.vals[i+A.nrows*j];
  }
  // Place the rows of A at the locations given by rows starting at column col.
  void setMatrix(const vector<unsigned int> rows, unsigned int col, const CMatrix& A)
  {
    DIMENSIONMATCH(rows.size()==A.nrows);
    BOUNDCHECK(col+A.ncols<=ncols);
    for(unsigned int i = 0; i<A.nrows; i++)
      for(unsigned int j = col; j<A.ncols+col; j++)
	  {
	    BOUNDCHECK(rows[i]<nrows);
	    vals[rows[i]+nrows*j]=A.vals[i+A.nrows*j];
	  }
  }
  // Place the columns of A at the locations given by cols starting at row row.
  void setMatrix(unsigned int row, const vector<unsigned int> cols, const CMatrix& A)
  {
    DIMENSIONMATCH(cols.size()==A.ncols);
    BOUNDCHECK(row+A.nrows<=nrows);
    for(unsigned int i=row; i<A.nrows+row; i++)
      for(unsigned int j=0; j<A.ncols; j++)
	  {
	    BOUNDCHECK(cols[i]<ncols);	    
	    vals[i+nrows*cols[j]]=A.vals[i+A.nrows*j];
	  }
  }
  // Place the rows and columns of A at rows and cols.
  void setMatrix(const vector<unsigned int> rows, const vector<unsigned int> cols, const CMatrix& A)
  {
    DIMENSIONMATCH(cols.size()==A.ncols);
    DIMENSIONMATCH(rows.size()==A.nrows);
    for(unsigned int i=0; i<A.nrows; i++)
      for(unsigned int j=0; j<A.ncols; j++)
	  {
	    BOUNDCHECK(rows[i]<nrows);
	    BOUNDCHECK(cols[j]<ncols);
	    vals[rows[i]+nrows*cols[j]]=A.vals[i+A.nrows*j];
	  }
  }
  // In place transpose of the matrix using Algorithm 380.
  void dtrans(int lwork)
  {
    int res;
    vector<int> work(lwork);
    dtrans_(vals, nrows, ncols, nrows*ncols, &work[0], lwork, res);
    SANITYCHECK(res==0);
  }
  // In place transpose of the matrix using Algorithm 513.
  void dtransr(int lwork)
  {
    int res;
    vector<int> work(lwork);
    dtransr_(vals, nrows, ncols, nrows*ncols, &work[0], lwork, res);
    SANITYCHECK(res==0);
  }
  // In place transpose of the matrix using Algorithm 467 (currently not working).
  void dxpose(int lwork)
  {
    // this is algorithm 467 (it is supposed to be more efficient - but doesn't work!)
    std::vector<int> work(lwork);
    dxpose_(vals, nrows, ncols, nrows*ncols, &work[0], lwork);
  }
  // Perform matrix transpose.
  void trans()
  {
    // if rows or columns are 1 dimensional then you don't need to move elements.
    if (nrows!=1 && ncols!=1)
      dtransr((nrows+ncols)/2); // this is algorithm 513 (467 doesn't seem to work)
    // if the matrix is square then you don't need to swap rows and columns.
    if (!isSquare())
    {
      int temp = nrows;
      nrows = ncols;
      ncols = temp;
    }
  }
  // Multiply the elements of the matrix by the elements of A.
  void multiply(const CMatrix& A)
  {
    // if A is a row or column vector it is `replicated' before the operation.
    if(A.nrows==1) {
      DIMENSIONMATCH(A.ncols==ncols);
      for(unsigned int i=0; i<nrows; i++)
	for(unsigned int j=0; j<ncols; j++)
	  vals[i+nrows*j] *= A.vals[j];
    }
    else if(A.ncols==1) {
      DIMENSIONMATCH(A.nrows==nrows);
      for(unsigned int j=0; j<ncols; j++)
	for(unsigned int i=0; i<nrows; i++)
	  vals[i+nrows*j] *= A.vals[i];
    }
    else  {
      DIMENSIONMATCH(A.nrows==nrows && A.ncols == ncols);
      for(unsigned int i=0; i<nrows*ncols; i++)
	vals[i] *= A.vals[i];
    }    
  }
  // Add a scalar to the elements of the matrix.
  void add(double c) {
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] += c;
  }
  // Add a scalar to a column of the matrix.
  void addCol(unsigned int j, double c) {
    for(unsigned int i=0; i<nrows; i++)
      vals[i+nrows*j] += c;
  }
  // Add a scalar to a row of the matrix.
  void addRow(unsigned int i, double c)
  {
    for(unsigned int j=0; j<ncols; j++)
      vals[i+nrows*j] += c;
  }
  // Add a scalar to diagonal of the matrix.
  void addDiag(double c)
  {
    MATRIXPROPERTIES(isSquare());
    for(unsigned int j=0; j<ncols; j++)
      vals[j+nrows*j] += c;
  }
  // Add a matrix to the matrix.
  void add(const CMatrix& A)
  {
    // if A is a row or column vector it is `replicated' before the operation.
    if(A.nrows==1)
    {
      DIMENSIONMATCH(A.ncols==ncols);
      for(unsigned int i=0; i<nrows; i++)
	for(unsigned int j=0; j<ncols; j++)
	  vals[i+nrows*j] += A.vals[j];
    }
    else if(A.ncols==1)
    {
      DIMENSIONMATCH(A.nrows==nrows);
      for(unsigned int j=0; j<ncols; j++)
	for(unsigned int i=0; i<nrows; i++)
	  vals[i+nrows*j] += A.vals[i];
    }
    else  
    {
      DIMENSIONMATCH(A.nrows==nrows && A.ncols == ncols);
      for(unsigned int i=0; i<nrows*ncols; i++)
	vals[i] += A.vals[i];
    }    
  }
  // Subtract a scalar from the elements of the matrix.
  void subtract(double c)
  {
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] -= c;
  }
  // Subtract a matrix from the matrix.
  void subtract(const CMatrix& A)
  {
    // if A is a row or column vector it is `replicated' before the operation.
    if(A.nrows==1)
    {
      DIMENSIONMATCH(A.ncols==ncols);
      for(unsigned int i=0; i<nrows; i++)
	for(unsigned int j=0; j<ncols; j++)
	  vals[i+nrows*j] -= A.vals[j];
    }
    else if(A.ncols==1)
    {
      DIMENSIONMATCH(A.nrows==nrows);
      for(unsigned int j=0; j<ncols; j++)
	for(unsigned int i=0; i<nrows; i++)
	  vals[i+nrows*j] -= A.vals[i];
    }
    else  
    {
      DIMENSIONMATCH(A.nrows==nrows && A.ncols == ncols);
      for(unsigned int i=0; i<nrows*ncols; i++)
	vals[i] -= A.vals[i];
    }    
  }
  void operator+=(double c)
  {
    add(c);
  }
  void operator+=(const CMatrix& A)
  {
    add(A);
  }
  void operator-=(double c)
  {
    subtract(c);
  }
  void operator-=(const CMatrix& A)
  {
    subtract(A);
  }
  void operator*=(double c)
  {
    multiply(c);
  }
  void operator*=(const CMatrix& A)
  {
    multiply(A);
  }

  void operator-()
  {
    negate();
  }
  // element by element operations
  // the MATLAB .* (element by element multiply)
  void dotMultiply(const CMatrix& A)
  {
    DIMENSIONMATCH(dimensionsMatch(A));
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] = vals[i]*A.getVal(i);
  }
  void dotMultiplyRowRow(unsigned int i, const CMatrix& A, unsigned int k)
  {
    DIMENSIONMATCH(ncols==A.ncols);
    for(unsigned int j=0; j<ncols; j++)
      vals[i + nrows*j] = vals[i + nrows*j]*A.getVal(k, j);
  }
  void dotMultiplyRowCol(unsigned int i, const CMatrix& A, unsigned int j)
  {
    DIMENSIONMATCH(ncols==A.nrows);
    for(unsigned int k=0; k<ncols; k++)
      vals[i + nrows*k] = vals[i + nrows*k]*A.getVal(k, j);
  }
  void dotMultiplyColRow(unsigned int j, const CMatrix& A, unsigned int i)
  {
    DIMENSIONMATCH(nrows==A.ncols);
    for(unsigned int k=0; k<nrows; k++)
      vals[k + nrows*j] = vals[k + nrows*j]*A.getVal(i, k);
  }
  void dotMultiplyColCol(unsigned int j, const CMatrix& A, unsigned int k)
  {
    DIMENSIONMATCH(nrows==A.nrows);
    for(unsigned int i=0; i<nrows; i++)
      vals[i + nrows*j] = vals[i + nrows*j]*A.getVal(i, k);
  }

  // the MATLAB ./ (element by element divide)
  void dotDivide(const CMatrix& A)
  {
    DIMENSIONMATCH(dimensionsMatch(A));
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] = vals[i]/A.getVal(i);
  }
  void dotDivideRowRow(unsigned int i, const CMatrix& A, unsigned int k)
  {
    DIMENSIONMATCH(ncols==A.ncols);
    for(unsigned int j=0; j<ncols; j++)
      vals[i + nrows*j] = vals[i + nrows*j]/A.getVal(k, j);
  }
  void dotDivideRowCol(unsigned int i, const CMatrix& A, unsigned int j)
  {
    DIMENSIONMATCH(ncols==A.nrows);
    for(unsigned int k=0; k<ncols; k++)
      vals[i + nrows*k] = vals[i + nrows*k]/A.getVal(k, j);
  }
  void dotDivideColRow(unsigned int j, const CMatrix& A, unsigned int i)
  {
    DIMENSIONMATCH(nrows==A.ncols);
    for(unsigned int k=0; k<nrows; k++)
      vals[k + nrows*j] = vals[k + nrows*j]/A.getVal(i, k);
  }
  void dotDivideColCol(unsigned int j, const CMatrix& A, unsigned int k)
  {
    if(nrows!=A.nrows)
      
    for(unsigned int i=0; i<nrows; i++)
      vals[i + nrows*j] = vals[i + nrows*j]/A.getVal(i, k);
  }
  // invert each element of the matrix.
  void invElements()
  {
    for(unsigned int i = 0; i<nrows*ncols; i++)
      vals[i] = 1.0/vals[i];
  }
  // exponentiate each element of the matrix.
  void exp()
  {
    for(unsigned int i = 0; i<nrows*ncols; i++)
      vals[i] = std::exp(vals[i]);
  }
  // take hyperbolic tangent of each element.
  void tanh()
  {
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] = std::tanh(vals[i]);
  }
  // take sigmoid of each element.
  void sigmoid()
  {
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] = 1.0/(1.0+std::exp(-vals[i]));
  }
  // take cumulative Gaussian of each element.
  void cumGaussian()
  {
    for(unsigned int i=0; i<nrows*ncols; i++)
      vals[i] = ndlutil::cumGaussian(vals[i]);
  }
  // replace each element of the matrix with its sign.
  void sign()
  {
    for(unsigned int i = 0; i<nrows*ncols; i++)
    {
      if(vals[i]>0)
	vals[i]=1;
      else
	vals[i]=-1;
    }
  }
  // take the logarithm of each element of the matrix.
  void log()
  {
    for(unsigned int i = 0; i<nrows*ncols; i++)
      vals[i] = std::log(vals[i]);
  }

  // Lapack operations
  // Solve A*X=B placing overwriting B with X.  *this is B on input, X on output
  int sysv(const CMatrix& A, const char* uplo, int lwork=0);
  // Perform a symmetric eigenvalue decomposition.
  int syev(CMatrix& eigVals, const char* jobz="n", const char* uplo="u", int lwork=0);
  // Perform an LU decomposition
  void lu();
  // Perform an inverse of the matrix.
  void inv();
  // Perform a cholesky decomposition of the matrix.
  void chol();

  // Perform Cholesky decomposition with specification of whether upper or lower triagular form is returned.
  void chol(const char* uplo);
  // Perform Cholesky decomposition, adding jitter if necessary --- if jitter is added, A is modified.
  double jitChol(CMatrix& A, unsigned int maxTries=20); 
  // invert a positive definite matrix given the Cholesky decomposition.
  void pdinv(const CMatrix& U);
  // invert a positive definite matrix.
  void pdinv();

  // Lapack Cholesky factorisation.
  void potrf(const char* type);
  // inverse based on Cholesky.
  void potri(const char* type);

  // BLAS operations
  // Level 1 BLAS operations.

  // Level 2 BLAS operations.
  // y:= alpha*op(A)*x + beta*y
  void gemv(const CMatrix& A, const CMatrix& x, double alpha, double beta, const char* trans);

  // y(i, :)' := alpha op(A)*x(k, :)' + beta*y(i, :)';
  void gemvRowRow(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* trans);
  // y(i, :)' := alpha op(A)*x(:, j) + beta*y(i, :)';
  void gemvRowCol(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int j, double alpha, double beta, const char* trans);
  // y(:, j) := alpha op(A)*x(:, k) + beta*y(:, j);
  void gemvColCol(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* trans);
  // y(:, j) := alpha op(A)*x(i, :)' + beta*y(:, j);
  void gemvColRow(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned int i, double alpha, double beta, const char* trans);

  // C:= alpha*A*B + beta*C or alpha*B*A + beta*C
  void symm(const CMatrix& A, const CMatrix& B, double alpha, double beta, const char* upperLower, const char* side);

  // y:= alpha*A*x + beta*y
  void symv(const CMatrix& A, const CMatrix& x, double alpha, double beta, const char* upperOrLower);

  // y(i, :)' := alpha A*x(k, :)' + beta*y(i, :)';
  void symvRowRow(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* upperOrLower);
  // y(i, :)' := alpha A*x(:, j) + beta*y(i, :)';
  void symvRowCol(unsigned int i, const CMatrix& A, const CMatrix& x, unsigned int j, double alpha, double beta, const char* upperOrLower);
  // y(:, j) := alpha A*x(:, k) + beta*y(:, j);
  void symvColCol(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned int k, double alpha, double beta, const char* upperOrLower);
  // y(:, j) := alpha A*x(i, :)' + beta*y(:, j);
  void symvColRow(unsigned int j, const CMatrix& A, const CMatrix& x, unsigned int i, double alpha, double beta, const char* upperOrLower);
  // y(yr1:_, j) <- alpha A*x(xr1:_, k) + beta*y(yr1:_, j);
  void symvColColOff(unsigned int j, int yr1, const CMatrix& A, const CMatrix& x, unsigned int k, int xr1, double alpha, double beta, const char* upperLower);


  // Level 3 BLAS operations.
  // C:= alpha*op(A)*op(B) + beta*C
  void gemm(const CMatrix& A, const CMatrix& B, double alpha, double beta, const char* transa, const char* transb);
  //  C:=alpha*op(A)*op(A)' + beta*C.
  void syrk(const CMatrix& A, double alpha, double beta, const char* type, const char* trans);
  // Triangular matrix multiply.
  void trmm(const CMatrix& B, double alpha, const char* side, const char* type, const char* trans, const char* diag);
  // Triangular solution of matrix equation.
  void trsm(const CMatrix& B, double alpha, const char* side, const char* type, const char* trans, const char* diag);

  // String commands.
  // Write matrix to file.
  void toUnheadedFile(const string fileName, const string comment="") const;
  // Write matrix to stream.
  void toUnheadedStream(ostream& out) const;
  // Obtain a matrix from a file.
  void fromUnheadedFile(const string fileName);
  // Obtain a matrix from a stream.
  void fromUnheadedStream(istream& in);
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& out);

  void toArray(double* outVals) const
  {
    dcopy_(nrows*ncols, vals, 1, outVals, 1);
  }
  void toArray(double* outVals, int numRows, int numCols) const
  {
    DIMENSIONMATCH(nrows==numRows);
    DIMENSIONMATCH(ncols==numCols);
    dcopy_(nrows*ncols, vals, 1, outVals, 1);
  }
  void toSingleArray(double* outVals, int numElements) const
  {
    DIMENSIONMATCH(nrows*ncols==numElements);
    dcopy_(nrows*ncols, vals, 1, outVals, 1);
  }
  void fromArray(double* inVals) 
  {
    dcopy_(nrows*ncols, inVals, 1, vals, 1);
  }
#ifdef _NDLMATLAB
  // MATLAB interaction commands
  mxArray* toMxArray() const;
  void fromMxArray(const mxArray* matlabArray);
  void fromSparseMxArray(const mxArray* matlabArray);
  void fromFullMxArray(const mxArray* matlabArray);
#endif /* _NDLMATLAB*/

#ifdef _HDF5
    void writeToHdf5(  const std::string& filename, const std::string& path_to_dataset ) const;

    void readFromHdf5( const std::string& filename, const std::string& path_to_dataset ) { throw std::runtime_error("Not implemented in CMatrix."); };
    void readFromHdf5( const std::string& filename, const std::string& path_to_dataset, bool transpose );
#endif

  // sample all elements from a uniform distribution between a and b.
  void rand(double a, double b);
  // sample all elements from a uniform distribution. between 0 and 1.
  void rand();
  // sample all elements from a Gaussian with mean and variance.
  void randn(double var, double mean);
  // sample all elements from a standard normal.
  void randn();
  // set all elements of the matrix to zero.
  void zeros();
  // set all elements of the matrix to one.
  void ones();
  // create a diagonal matrix from a vector.
  void diag(const CMatrix &d);
  // create a diagonal matrix from a double.
  void diag(double d);
  // create the identity matrix.
  void eye();
  // set all elements to their negative value.
  void negate();
  // sum all elements of the matrix.
  double sum() const;
  // return trace of the matrix.
  double trace() const;
  // check if the two matrices are identical to within a tolerance.
  bool equals(const CMatrix& A, double tol=ndlutil::MATCHTOL) const;
  // find the maximum absolute difference between matrices.
  double maxAbsDiff(const CMatrix& X) const;
  // find the maximum element of the matrix.
  double max() const; 
  // Find the minimum of each row. 
  void minRow(CMatrix& m) const;
  // Find the maximum of each row.
  void maxRow(CMatrix& m) const;

  string getType() const
  {
    return "doubleMatrix";
  }
  string getBaseType() const
  {
    return "matrix";
  }
  // resize a matrix.
  void resize(unsigned int rows, unsigned int cols)
  {
    if(rows!=nrows || cols!=ncols)
    {
      memDeAllocate();	    
      nrows = rows;
      ncols = cols;
      memAllocate();
    }
  }
  void copy(const CMatrix& x);

private:

  void _init()
  {
  }

  static unsigned int totalAlloc;
  // Helper memory functions.
  void memAllocate();
  void memDeAllocate();
  void memReAllocate(int rowIncrease, int colIncrease);

  double* vals;
  
  unsigned int nrows;
  unsigned int ncols;
  enum{NEW, MATLAB, MALLOC};
  static const int allocation = NEW;
  bool symmetric;
  bool triangular;
};

CMatrix zeros(const int rows, const int cols);

double randn(double mean, double var);
double randn();
double max(const CMatrix& A);
  
CMatrix lu(const CMatrix& inMatrix);
CMatrix chol(const CMatrix& inMatrix);
void chol(CMatrix& outMatrix, const CMatrix& inMatrix);
CMatrix jitChol(const CMatrix& inMatrix, unsigned int maxTries);
double jitChol(CMatrix& outMatrix, const CMatrix& inMatrix, unsigned int maxTries);
CMatrix inv(const CMatrix& inMatrix);
//  Normal matrix multiply.
CMatrix multiply(const CMatrix& A, const CMatrix& B);
//  Matrix multiply but allowing transpose operations on the matrices.
CMatrix multiply(const CMatrix& A, const char* transa, const CMatrix& B, const char* transb);
//  give the trace of the matrix.
double trace(const CMatrix& A);
//  sum all elements in the matrix.
double sum(const CMatrix& A);
// double dist(const CMatrix& A, const CMatrix& B);
CMatrix pdinv(const CMatrix& A);
//  Overload output operator for console display.
ostream& operator<<(ostream& os, const CMatrix& A);
// ostream& operator<<(CMatrix A);
  
CMatrix sumRow(const CMatrix&);
CMatrix meanRow(const CMatrix&);
CMatrix varRow(const CMatrix&);
CMatrix stdRow(const CMatrix&);
CMatrix sumCol(const CMatrix&);
CMatrix meanCol(const CMatrix&);
CMatrix varCol(const CMatrix&);
CMatrix stdCol(const CMatrix&);
//  Log determinant of a positive definite matrix where U is the Cholesky decomposition of the matrix.
double logDet(const CMatrix& U);
  
    
// inline void swap(CMatrix& x, CMatrix& y);


#endif
