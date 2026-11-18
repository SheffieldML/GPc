/* CIP-0022: CSV vs SVML data load parity and CSV error paths. */
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include "CClctrl.h"
#include "CMatrix.h"
#include "ndlexceptions.h"

using namespace std;

namespace {
class TestCtrl : public CClctrl {
 public:
  TestCtrl(int c, char** v) : CClctrl(c, v) {}
  void helpInfo() {}
  void helpHeader() {}
};

int fail = 0;

void check(bool ok, const string& msg)
{
  if(!ok)
  {
    cout << "FAILURE: " << msg << endl;
    ++fail;
  }
  else
    cout << "passes: " << msg << endl;
}
}  // namespace

int main(int argc, char** argv)
{
  TestCtrl ctrl(argc, argv);
  ctrl.setVerbosity(0);

  try
  {
    CMatrix Xs(1, 1), ys(1, 1), Xc(1, 1), yc(1, 1);
    ctrl.readSvmlDataFile(Xs, ys, "examples/sinc.svml");
    ctrl.readCsvDataFile(Xc, yc, "examples/sinc.csv");
    check(Xs.getRows() == Xc.getRows() && Xs.getCols() == Xc.getCols(),
          "sinc SVML/CSV X shape match");
    check(ys.getRows() == yc.getRows() && ys.getCols() == yc.getCols(),
          "sinc SVML/CSV y shape match");
    check(Xs.equals(Xc) && ys.equals(yc),
          "sinc SVML/CSV values match within MATCHTOL");

    CMatrix Xa(1, 1), ya(1, 1);
    ctrl.setFileFormat(0);  // model-style default; extension must win
    ctrl.readData(Xa, ya, "examples/sinc.csv");
    check(Xa.equals(Xc) && ya.equals(yc),
          "readData auto-detects .csv despite fileFormat 0");

    const string badPath = "examples/_cip0022_bad.csv";
    {
      ofstream out(badPath.c_str());
      out << "1.0,2.0,3.0\n1.0,2.0\n";
    }
    bool threw = false;
    try
    {
      CMatrix Xb(1, 1), yb(1, 1);
      ctrl.readCsvDataFile(Xb, yb, badPath);
    }
    catch(ndlexceptions::FileFormatError&)
    {
      threw = true;
    }
    check(threw, "inconsistent CSV column count throws FileFormatError");
    remove(badPath.c_str());
  }
  catch(ndlexceptions::Error& err)
  {
    cout << "FAILURE: unexpected " << err.getMessage() << endl;
    ++fail;
  }
  catch(...)
  {
    cout << "FAILURE: unexpected non-ndlexception" << endl;
    ++fail;
  }

  if(fail)
    cout << fail << " failure(s)." << endl;
  else
    cout << "All tests passed." << endl;
  return fail ? 1 : 0;
}
