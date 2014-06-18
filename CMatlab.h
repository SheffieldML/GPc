/*This file contains an abstract base class CMatinterface which when implemented allows an object to load itself from MATLAB and save itselve to MATLAB. This is only compiled if _NDLMATLAB is defined. It requires MATLAB libraries to compile.

  20/10/2005 Included William V. Baxter's modifications to allow compilation under MSVC.*/

#ifndef CMATLAB_H
#define CMATLAB_H

#ifdef _NDLMATLAB
#include "ndlexceptions.h"
#include "mex.h"
#include "mat.h"
using namespace std;

// An abstract base class which enables loading and saving of a class to MATLAB.
class CMatInterface 
{
 public:
  virtual mxArray* toMxArray() const=0;
  virtual void fromMxArray(const mxArray* matlabArray)=0;
  void readMatlabFile(const string fileName, const string variableName) 
  {
    MATFile* matFile = matOpen(fileName.c_str(), "r");
    if(matFile==NULL)
      throw ndlexceptions::FileReadError(fileName);
    
    mxArray* matlabArray = matGetVariable(matFile, variableName.c_str());
    if(matlabArray==NULL)
      throw ndlexceptions::FileFormatError(fileName);
    fromMxArray(matlabArray);
    mxDestroyArray(matlabArray);
    if(matClose(matFile) !=0 )
      throw ndlexceptions::FileReadError(fileName);
    
  }
  void updateMatlabFile(string fileName, const string variableName) const
  {
    MATFile* matFile = matOpen(fileName.c_str(), "u");
    if(matFile==NULL)
      throw ndlexceptions::FileWriteError(fileName);
    mxArray* matlabArray = toMxArray();
    matPutVariable(matFile, variableName.c_str(), matlabArray);
    mxDestroyArray(matlabArray);
    if(matClose(matFile) !=0 )
      throw ndlexceptions::FileWriteError(fileName);
  }
  
  void writeMatlabFile(const string fileName, const string variableName) const
  {
    MATFile* matFile = matOpen(fileName.c_str(), "w");
    if(matFile==NULL)
      throw ndlexceptions::FileWriteError(fileName);
    mxArray* matlabArray = toMxArray();
    matPutVariable(matFile, variableName.c_str(), matlabArray);
    mxDestroyArray(matlabArray);
    if(matClose(matFile) !=0 )
      throw ndlexceptions::FileWriteError(fileName);
  }
  mxArray* convertMxArray(const bool val) const
  {
    int dims[1];
    dims[0] = 1;
    mxArray* matlabArray = mxCreateNumericArray(1, dims, mxDOUBLE_CLASS, mxREAL);
    double* matlabVals = mxGetPr(matlabArray);
    if(val)
      matlabVals[0] = 1.0;
    else
      matlabVals[0] = 0.0;
    return matlabArray;
  }
  mxArray* convertMxArray(const double val) const
  {
    int dims[1];
    dims[0] = 1;
    mxArray* matlabArray = mxCreateNumericArray(1, dims, mxDOUBLE_CLASS, mxREAL);
    double* matlabVals = mxGetPr(matlabArray);
    matlabVals[0] = val;
      return matlabArray;
  }
  mxArray* convertMxArray(const string val) const
  {
    return mxCreateString(val.c_str());
  }
  mxArray* convertMxArray(const vector<int> vals) const
  {
    int dims[2];
    dims[0]=vals.size();
    dims[1] = 1;
    mxArray* matlabArray = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    double* matlabVals = mxGetPr(matlabArray);
    for(int i=0; i<vals.size(); i++)
    {
      matlabVals[i]=(double)vals[i];
    }
      return matlabArray;
  }
  int mxArrayToInt(const mxArray* matlabArray) const
  {
    int val = 0;
    mxClassID classID = mxGetClassID(matlabArray);
    if(classID==mxDOUBLE_CLASS)
    {
      assert(mxGetN(matlabArray)==1);
      assert(mxGetM(matlabArray)==1);
      double* valD = mxGetPr(matlabArray);
      val = (int)valD[0];
    }
    else
      throw ndlexceptions::NotImplementedError("Conversion of this type to int not yet supported.");
    return val;
  }
  double mxArrayToDouble(const mxArray* matlabArray) const
  {
    double val = 0;
    mxClassID classID = mxGetClassID(matlabArray);
    if(classID==mxDOUBLE_CLASS)
    {
      assert(mxGetN(matlabArray)==1);
      assert(mxGetM(matlabArray)==1);
      double* valD = mxGetPr(matlabArray);
      val = valD[0];
    }
    else
      throw ndlexceptions::NotImplementedError("Conversion of this type to double not yet supported.");
    return val;
  }
  string mxArrayToString(const mxArray* matlabArray) const
  {
    vector<char> charVal;
    int buflen;
    mxClassID classID = mxGetClassID(matlabArray);
    if(classID ==  mxCHAR_CLASS)
    {
      buflen=mxGetNumberOfElements(matlabArray)*sizeof(char) + 1;
      charVal.resize(buflen);
      int ret = mxGetString(matlabArray, &charVal[0], buflen);
    }
    else
      throw ndlexceptions::NotImplementedError("Conversion of this type to string not yet supported.");
    string val(&charVal[0]);
    return val;
  }
  vector<int> mxArrayToVectorInt(const mxArray* matlabArray) const
  {
    vector<int> val;
    mxClassID classID = mxGetClassID(matlabArray);
    if(classID == mxDOUBLE_CLASS)
    {
      int length=mxGetNumberOfElements(matlabArray);
      double* valD = mxGetPr(matlabArray);
      for(int i=0; i<length; i++)
	val.push_back((int)valD[i]);
    }
    else
      throw ndlexceptions::NotImplementedError("Conversion of this type to vector<int> not yet supported.");
    return val;
  }
  bool mxArrayToBool(const mxArray* matlabArray) const
  {
    bool val;
    mxClassID classID = mxGetClassID(matlabArray);
    if(classID==mxDOUBLE_CLASS)
    {
      assert(mxGetN(matlabArray)==1);
      assert(mxGetM(matlabArray)==1);
      double* valD = mxGetPr(matlabArray);
      val = valD[0];
      if(val!=0)
	val=true;
      else
	val=false;
    }
    else
      throw ndlexceptions::NotImplementedError("Conversion of this type to bool not yet supported.");
    return false;
  }	  
  
  
  mxArray* mxArrayExtractMxArrayField(const mxArray* matlabArray, const string fieldName, const int index=0) const
  {
    const char* fName = fieldName.c_str();
    if(mxGetClassID(matlabArray) != mxSTRUCT_CLASS)
      throw ndlexceptions::Error("mxArray is not a structure.");
    mxArray* fieldPtr = mxGetField(matlabArray, index, fName);
    return fieldPtr;
  }
  
  int mxArrayExtractIntField(const mxArray* matlabArray, const string fieldName, const int index=0) const
  {
    mxArray* fieldPtr = mxArrayExtractMxArrayField(matlabArray, fieldName, index);
    if(fieldPtr==NULL)
      throw ndlexceptions::Error("Requested field does not exist");
    return mxArrayToInt(fieldPtr);
  }
  double mxArrayExtractDoubleField(const mxArray* matlabArray, const string fieldName, const int index=0) const
  {
    mxArray* fieldPtr = mxArrayExtractMxArrayField(matlabArray, fieldName, index);
    if(fieldPtr==NULL)
      throw ndlexceptions::Error("Requested field does not exist");
    return mxArrayToDouble(fieldPtr);
  }
  string mxArrayExtractStringField(const mxArray* matlabArray, const string fieldName, const int index=0) const
  {
    mxArray* fieldPtr = mxArrayExtractMxArrayField(matlabArray, fieldName, index);
    if(fieldPtr==NULL)
      throw ndlexceptions::Error("Requested field does not exist");
    return mxArrayToString(fieldPtr);
  }
  
  vector<int> mxArrayExtractVectorIntField(const mxArray* matlabArray, const string fieldName, const int index=0) const
  {
    mxArray* fieldPtr = mxArrayExtractMxArrayField(matlabArray, fieldName, index);
    if(fieldPtr==NULL)
      throw ndlexceptions::Error("Requested field does not exist");
    return mxArrayToVectorInt(fieldPtr);
  }
  bool mxArrayExtractBoolField(const mxArray* matlabArray, const string fieldName, const int index=0)
  {
    mxArray* fieldPtr = mxArrayExtractMxArrayField(matlabArray, fieldName, index);
    if(fieldPtr==NULL)
      throw ndlexceptions::Error("Requested field does not exist");
    return mxArrayToBool(fieldPtr);
  }
  
};

#else /* not _NDLMATLAB */

class CMatInterface {
 public:
 private:
};
#endif
#endif /* not CMATLAB_H*/
