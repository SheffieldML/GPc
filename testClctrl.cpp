/* CIP-0022: CSV vs SVML data load parity and CSV error / format paths. */
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

bool throwsFileFormat(TestCtrl& ctrl, const string& path)
{
  try
  {
    CMatrix X(1, 1), y(1, 1);
    ctrl.readCsvDataFile(X, y, path);
  }
  catch(ndlexceptions::FileFormatError&)
  {
    return true;
  }
  return false;
}

void writeText(const string& path, const string& body)
{
  ofstream out(path.c_str(), ios::binary);
  out << body;
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

    // Extension routing: .svml, unknown→fileFormat, .mat without MATLAB.
    CMatrix Xsv(1, 1), ysv(1, 1);
    ctrl.readData(Xsv, ysv, "examples/sinc.svml");
    check(Xsv.equals(Xs) && ysv.equals(ys), "readData loads .svml by extension");

    const string tmpCsvAsDat = "examples/_cip0022_fallback.dat";
    writeText(tmpCsvAsDat, "1.0,2.0\n3.0,4.0\n");
    ctrl.setFileFormat(2);
    CMatrix Xfb(1, 1), yfb(1, 1);
    ctrl.readData(Xfb, yfb, tmpCsvAsDat);
    check(Xfb.getRows() == 2 && Xfb.getCols() == 1 && yfb.getRows() == 2,
          "unknown extension falls back to fileFormat CSV");
    remove(tmpCsvAsDat.c_str());

    bool matThrew = false;
    try
    {
      CMatrix Xm(1, 1), ym(1, 1);
      ctrl.readData(Xm, ym, "examples/_missing.mat");
    }
    catch(ndlexceptions::MatlabInterfaceError&)
    {
      matThrew = true;
    }
    catch(ndlexceptions::FileReadError&)
    {
      // If MATLAB were linked, missing file would fail earlier; without it,
      // extension 1 hits the no-MATLAB throw before open.
      matThrew = false;
    }
#ifndef _NDLMATLAB
    check(matThrew, "readData .mat without MATLAB throws MatlabInterfaceError");
#else
    (void)matThrew;
#endif

    // Missing file.
    bool missingThrew = false;
    try
    {
      CMatrix Xm(1, 1), ym(1, 1);
      ctrl.readCsvDataFile(Xm, ym, "examples/_cip0022_does_not_exist.csv");
    }
    catch(ndlexceptions::FileReadError&)
    {
      missingThrew = true;
    }
    check(missingThrew, "missing CSV throws FileReadError");

    // Header skip, # comments, blank lines, CR stripping, verbosity.
    const string hdrPath = "examples/_cip0022_header.csv";
    writeText(hdrPath,
              "# comment\r\n"
              "\r\n"
              "x,y\r\n"
              "1.5,2.5\r\n"
              "3.5,4.5\r\n");
    ctrl.setVerbosity(2);
    CMatrix Xh(1, 1), yh(1, 1);
    ctrl.readCsvDataFile(Xh, yh, hdrPath);
    check(Xh.getRows() == 2 && Xh.getVal(0, 0) == 1.5 && yh.getVal(1, 0) == 4.5,
          "header/comments/CRLF CSV loads");
    ctrl.setVerbosity(0);
    remove(hdrPath.c_str());

    // TSV / whitespace (no commas) and .tsv extension via readData.
    const string tsvPath = "examples/_cip0022.tsv";
    writeText(tsvPath, "1.0\t2.0\n3.0 4.0\n");
    CMatrix Xt(1, 1), yt(1, 1);
    ctrl.readData(Xt, yt, tsvPath);
    check(Xt.getRows() == 2 && Xt.getVal(1, 0) == 3.0 && yt.getVal(0, 0) == 2.0,
          "TSV/whitespace via .tsv extension");
    remove(tsvPath.c_str());

    // .txt extension + spaces-only blank field line (empty fields → continue).
    const string txtPath = "examples/_cip0022.txt";
    writeText(txtPath, "   \n1.0 9.0\n");
    CMatrix Xx(1, 1), yx(1, 1);
    ctrl.readCsvDataFile(Xx, yx, txtPath);
    check(Xx.getRows() == 1 && yx.getVal(0, 0) == 9.0,
          "space-separated .txt skips blank separator line");
    remove(txtPath.c_str());

    // Error paths: inconsistent columns, single column, non-numeric, empty,
    // consecutive commas / trailing comma (empty token).
    const string badCols = "examples/_cip0022_bad.csv";
    writeText(badCols, "1.0,2.0,3.0\n1.0,2.0\n");
    check(throwsFileFormat(ctrl, badCols),
          "inconsistent CSV column count throws FileFormatError");
    remove(badCols.c_str());

    const string oneCol = "examples/_cip0022_onecol.csv";
    writeText(oneCol, "1.0\n2.0\n");
    check(throwsFileFormat(ctrl, oneCol),
          "single-column CSV throws FileFormatError");
    remove(oneCol.c_str());

    const string nonNum = "examples/_cip0022_nonnum.csv";
    writeText(nonNum, "1.0,2.0\n3.0,abc\n");
    check(throwsFileFormat(ctrl, nonNum),
          "non-numeric CSV token throws FileFormatError");
    remove(nonNum.c_str());

    const string emptyCsv = "examples/_cip0022_empty.csv";
    writeText(emptyCsv, "# only comments\n\n");
    check(throwsFileFormat(ctrl, emptyCsv),
          "empty CSV throws FileFormatError");
    remove(emptyCsv.c_str());

    const string emptyTok = "examples/_cip0022_emptytok.csv";
    writeText(emptyTok, "1.0,,2.0\n");
    check(throwsFileFormat(ctrl, emptyTok),
          "empty CSV field (consecutive commas) throws FileFormatError");
    remove(emptyTok.c_str());

    const string trailComma = "examples/_cip0022_trail.csv";
    writeText(trailComma, "1.0,2.0,\n");
    check(throwsFileFormat(ctrl, trailComma),
          "trailing comma empty field throws FileFormatError");
    remove(trailComma.c_str());

    // Verbosity path on readData success.
    ctrl.setVerbosity(2);
    CMatrix Xv(1, 1), yv(1, 1);
    ctrl.readData(Xv, yv, "examples/sinc.csv");
    check(Xv.equals(Xc), "readData with verbosity>1 still loads");
    ctrl.setVerbosity(0);
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
