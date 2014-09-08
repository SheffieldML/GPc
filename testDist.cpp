#include "CDist.h"

using namespace std;

int testType(const string distType);
int testDist(CDist* dist, CDist* dist2, const string fileName);

int main()
{
  int fail=0;
  try
  {
    fail += testType("gaussian");
    fail += testType("gamma");
    fail += testType("wang");
    cout << "Number of failures: " << fail << "." << endl;
  }
  catch(ndlexceptions::FileFormatError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileReadError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileWriteError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::Error err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(std::bad_alloc err)
  {
    cerr << "Out of memory.";
    exit(1);
  }
  catch(std::exception err)
  {
    cerr << "Unhandled exception.";
    exit(1);
  }

}

int testType(const string distType)
{
  string fileName = "matfiles" + ndlstrutil::dirSep() + distType + "DistTest.mat";
  CMatrix X;
  X.readMatlabFile(fileName, "X");

  CDist* dist;
  CDist* dist2;
 
  if(distType=="gaussian")
  {
    dist = new CGaussianDist();
    dist2 = new CGaussianDist();
  }
  if(distType=="gamma")
  {
    dist = new CGammaDist();
    dist2 = new CGammaDist();
  }
  else if(distType=="wang")
  {
    dist = new CWangDist();
    dist2 = new CWangDist();
  }
  int fail = testDist(dist, dist2, fileName);
  delete dist;
  delete dist2;
  return fail;
}
int testDist(CDist* dist, CDist* dist2, const string fileName)
{
  
  int fail = 0;
  CMatrix params;
  params.readMatlabFile(fileName, "params");
  CMatrix X;
  X.readMatlabFile(fileName, "X");
  CMatrix g;
  g.readMatlabFile(fileName, "g");
  CMatrix ll;
  ll.readMatlabFile(fileName, "ll");
  dist->setTransParams(params);
  dist2->readMatlabFile(fileName, "dist");
  if(dist2->equals(*dist))
    cout << dist->getName() << " Initial Dist matches." << endl;
  else
  {
    cout << "FAILURE: " << dist->getName() << " Initial Dist." << endl;
    fail++;
  }
  CMatrix g1(X.getRows(), X.getCols());
  dist->getGradInputs(g1, X);
  if(g1.equals(g))
    cout << dist->getName() << " parameter gradient matches." << endl;
  else
  {
    cout << "FAILURE: " << dist->getName() << " parameter gradient." << endl;
    cout << "Matlab gradient: " << endl;
    cout << g << endl;
    cout << "C++ gradient: " << endl;
    cout << g1 << endl;
    fail++;
  }
  CMatrix ll2(1, 1);
  ll2.setVal(dist->logProb(X), 0);
  if(ll2.equals(ll))
    cout << dist->getName() << " log likelihood matches." << endl;
  else
  {
    cout << "FAILURE: " << dist->getName() << " log likelihood." << endl;
    cout << "Matlab log likelihood: " << endl;
    cout << ll << endl;
    cout << "C++ log likelihood: " << endl;
    cout << ll2 << endl;
    fail++;
  }

  // Matlab read and Read and write test 
  dist->writeMatlabFile("crap.mat", "writtenDist");
  dist2->readMatlabFile("crap.mat", "writtenDist");
  if(dist->equals(*dist2))
    cout << "MATLAB written " << dist->getName() << " matches read in dist. Read and write to matlab passes." << endl;
  else
  {
    cout << "FAILURE: MATLAB read in " << dist->getName() << " does not match written out dist." << endl;
    fail++;
  }

  // Matlab read and Read and write test 
  dist->toFile("crap_dist");
  dist2->fromFile("crap_dist");
  if(dist->equals(*dist2))
    cout << "Text written " << dist->getName() << " matches read in dist. Read and write to text passes." << endl;
  else
  {
    cout << "FAILURE: Text read in " << dist->getName() << " does not match written out dist." << endl;
    fail++;
  }
  return fail;
}
  
