  #include "CTransform.h"

  CParamTransforms* ptransforms;
  int testTransform(string transType);
  int main()
  { 
    ptransforms = new CParamTransforms();
    int fail = 0;
    fail+=testTransform("exp");
    fail+=testTransform("negLogLogit");
    fail+=testTransform("sigmoid");

    ptransforms->writeMatlabFile("crap.mat", "writtenTransforms");
    CParamTransforms* preadTrans = new CParamTransforms();
    preadTrans->readMatlabFile("crap.mat", "writtenTransforms");
   if(ptransforms->equals(*preadTrans))
      cout << "MATLAB Read and write transforms to stream passed"<<endl;
    else
    {
      cout << "FAILURE: Read and write to MATLAB" << endl;
      cout << "Written transforms:" << endl;
      ptransforms->display(cout);
      cout << "Read in transforms: " << endl;
      preadTrans->display(cout);
      fail++;
    }
 
    ptransforms->toFile("crap_paramtransform");
    preadTrans->fromFile("crap_paramtransform");
    if(ptransforms->equals(*preadTrans))
      cout << "Read and write transforms to stream passed"<<endl;
    else
    {
      cout << "FAILURE: Read and write to stream" << endl;
      cout << "Written transforms:" << endl;
      ptransforms->display(cout);
      cout << "Read in transforms: " << endl;
      preadTrans->display(cout);
      fail++;
    }
    cout << "Number of failures: " << fail << endl;
  }
   
  int testTransform(string transType)
  {
    int fail = 0;
    CTransform* trans = CTransform::getNewTransformPointer(transType);
    int ncols = 9;
    double a[19] = {-1e16, -1e8, -1e4, -100, -1, -0.1, -1e-5, -1e-9, -1e-13, 0, 1e-13, 1e-9, 1e-5, 0.1, 1, 100, 1e4, 1e8, 1e16};
    double x[19];
    double newa[19];
    double g[19];
    cout << "Transform " << transType << endl;
    for(int i=0; i<9; i++)
    {
      x[i] = trans->atox(a[i]);
      newa[i] = trans->xtoa(x[i]);
      if(newa[i]!=a[i])
      {
        cout << "FAIL: " << endl;
        cout << "a: " << a[i] << " x: " << x[i] << " newa: "  << newa[i] << endl;
        fail++;
      }
      else
        cout << "atox -> xtoa check passed: a: " << a[i] << endl;
      g[i] = trans->gradfact(x[i]);
      double diff = 1e-6;
      double aplus = a[i]+diff;
      double xplus = trans->atox(aplus);
      double aminus = a[i]-diff;
      double xminus = trans->atox(aminus);
      double gdiff = (xplus - xminus)/(2*diff);
      if(abs(gdiff-g[i])>2*diff)
      {
        cout << "FAIL: Gradient: " << "g: " << g[i] << " gdiff: " << gdiff << endl;
      }
      else
        cout << "Gradient check passed: a: " << a[i] << endl;
    }
    ptransforms->addTransform(trans, 1);
    ptransforms->addTransform(trans, 2);
    return fail;
  }
