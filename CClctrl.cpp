#include "CClctrl.h"
#include <cstdlib>
#include <vector>

CClctrl::CClctrl(int arc, char** arv) : argc(arc), argv(arv)
{
  argNo=1;
  fileFormat = 0; // svmlight format
  verbosity = 2; // second highest level (highest is 3)
  time_t seconds;
  time(&seconds);
  setSeed((unsigned long) seconds);

}

bool CClctrl::isCurrentArg(string shortName, string commandName)
{
  return (getCurrentArgument()==shortName || getCurrentArgument()==commandName);
}
void CClctrl::confirmCurrentArg(string commandName)
{
  if(getCurrentArgument()!=commandName)
    exitError("Unrecognised argument " + getCurrentArgument() + " for `" + getMode() + "' command.");
}
void CClctrl::unrecognisedFlag()
{
  exitError("Unrecognised flag: " + getCurrentArgument() + " under " + getMode() +  " command.");
}
void CClctrl::helpUsage(const string description, const int width, const int padding)
{
  ndlstrutil::wrapOutputText(cout, "Usage: " + description, width, padding);
  cout << endl;
  cout << endl;
}
void CClctrl::helpDescriptor(const string description, const int width, const int padding)
{
  ndlstrutil::wrapOutputText(cout, description, width, padding);
  cout << endl;
  cout << endl;
}
void CClctrl::helpArgument(const string flags, const string explanation, const int width, const int padding)
{
  string padStr="";
  for(int i=0; i<padding; i++)
    padStr += " ";
  cout << padStr << flags << endl;
  ndlstrutil::wrapOutputText(cout, explanation, width, padding+3);
  cout << endl;
  cout << endl;
}
void CClctrl::waitForSpace()
{
  cout << "Press enter for more." << endl;
  char ch;
  cin.get(ch);    
}
int CClctrl::readSvmlDataFile(CMatrix& X, CMatrix& y, const string fileName)
{
  ifstream in(fileName.c_str());
  if(!in.is_open()) throw ndlexceptions::FileReadError(fileName);
  
  string line;
  string token;
  bool featureRead=false;
  int numData=0;
  int maxFeat=0;
  while(getline(in, line))
  {
    featureRead=false;
    
    if(line[line.size()-1]=='\r')
      line.erase(line.size()-1);
    if(line[0]=='#')
      continue;
    numData++;
    int pos=0;
    while(pos<line.size())
    {
      token.erase();
      while(pos<line.size() && line[pos]!=' ')
      {
	token+=line[pos];
	pos++;
      }
      pos++;
      if(token.size()>0)
      {
	// deal with token.
	if(featureRead)
	{
	  int ind = token.find(':');
	  if(ind==std::string::npos || ind < 0)
	  {
	    ndlexceptions::StreamFormatError err("");
	    throw ndlexceptions::FileFormatError(fileName, err);
	  }
	  string featStr=token.substr(0, ind);
	  int featNum = atoi(featStr.c_str());
	  if(featNum>maxFeat)
	    maxFeat=featNum;
	}
	else
	{
	  featureRead=true;
	}
      }
    }
    
  }
  if(verbosity>1)
  {
    cout << "Data number of features: " << maxFeat << endl;
    cout << "Number of data: " << numData << endl;
  }
  X.resize(numData, maxFeat);
  X.zeros();
  y.resize(numData, 1);
  y.zeros();
  in.close();
  ifstream inToo(fileName.c_str());
  int pointNo=0;
  while(getline(inToo, line))
  {
    if(line[line.size()-1]=='\r')
      line.erase(line.size()-1);
    featureRead=false;
    if(line[0]=='#')
      continue;
    else
    {
      int pos=0;
      while(pos<line.size())
      {
	token.erase();
	while(pos<line.size() && line[pos]!=' ')
	{
	  token+=line[pos];
	  pos++;
	}
	pos++;
	if(token.size()>0)
	{
	  // deal with token.
	  if(featureRead)
	  {
	    int ind = token.find(':');		      
	    // TODO Check that : is in the string.
	    string featStr=token.substr(0, ind);
	    string featValStr=token.substr(ind+1, token.size()-ind);
	    int featNum = atoi(featStr.c_str());
	    if(featNum<1 || featNum>maxFeat || pointNo<0 || pointNo>=numData)
	    {
	      ndlexceptions::StreamFormatError err("");
	      throw ndlexceptions::FileFormatError(fileName, err);
	    }
		      
	    double featVal = atof(featValStr.c_str());
	    X.setVal(featVal, pointNo, featNum-1);
	  }
	  else
	  {
	    y.setVal(atof(token.c_str()), pointNo);
	    featureRead=true;
	  }
	}
      }
      
    }
    pointNo++;
  }
  // WVB ADDED
  return -1;
}

namespace {

string dataFileExtension(const string& fileName)
{
  string::size_type slash = fileName.find_last_of("/\\");
  string::size_type base = (slash == string::npos) ? 0 : slash + 1;
  string::size_type dot = fileName.find_last_of('.');
  if(dot == string::npos || dot < base)
    return "";
  string ext = fileName.substr(dot + 1);
  for(string::size_type i = 0; i < ext.size(); ++i)
  {
    if(ext[i] >= 'A' && ext[i] <= 'Z')
      ext[i] = static_cast<char>(ext[i] - 'A' + 'a');
  }
  return ext;
}

// -1 = unknown (fall back to fileFormat); 0 svml; 1 mat; 2 csv
int dataFormatFromExtension(const string& fileName)
{
  const string ext = dataFileExtension(fileName);
  if(ext == "svml")
    return 0;
  if(ext == "mat")
    return 1;
  if(ext == "csv" || ext == "tsv" || ext == "txt")
    return 2;
  return -1;
}

bool csvTokenIsNumeric(const string& tok)
{
  if(tok.empty())
    return false;
  char* end = 0;
  strtod(tok.c_str(), &end);
  return end != tok.c_str() && *end == '\0';
}

void splitCsvFields(const string& line, vector<string>& fields)
{
  fields.clear();
  const bool useComma = (line.find(',') != string::npos);
  string cur;
  for(string::size_type i = 0; i < line.size(); ++i)
  {
    const char c = line[i];
    const bool sep = useComma ? (c == ',')
                              : (c == ' ' || c == '\t');
    if(sep)
    {
      if(!cur.empty())
      {
        fields.push_back(cur);
        cur.clear();
      }
      else if(useComma)
        fields.push_back("");
    }
    else if(c != '\r')
      cur += c;
  }
  if(!cur.empty())
    fields.push_back(cur);
  else if(useComma && !line.empty() && line[line.size() - 1] == ',')
    fields.push_back("");
}

}  // namespace

int CClctrl::readCsvDataFile(CMatrix& X, CMatrix& y, const string fileName)
{
  ifstream in(fileName.c_str());
  if(!in.is_open())
    throw ndlexceptions::FileReadError(fileName);

  vector<vector<double> > rows;
  int nCols = -1;
  string line;
  vector<string> fields;
  bool sawHeader = false;

  while(getline(in, line))
  {
    if(!line.empty() && line[line.size() - 1] == '\r')
      line.erase(line.size() - 1);
    if(line.empty())
      continue;
    if(line[0] == '#')
      continue;

    splitCsvFields(line, fields);
    if(fields.empty())
      continue;

    if(!sawHeader)
    {
      bool allNumeric = true;
      for(size_t i = 0; i < fields.size(); ++i)
      {
        if(!csvTokenIsNumeric(fields[i]))
        {
          allNumeric = false;
          break;
        }
      }
      if(!allNumeric)
      {
        sawHeader = true;
        continue;
      }
      sawHeader = true;
    }

    if(static_cast<int>(fields.size()) < 2)
    {
      ndlexceptions::StreamFormatError err("CSV needs at least two columns (X ..., y)");
      throw ndlexceptions::FileFormatError(fileName, err);
    }
    if(nCols < 0)
      nCols = static_cast<int>(fields.size());
    else if(static_cast<int>(fields.size()) != nCols)
    {
      ndlexceptions::StreamFormatError err("inconsistent CSV column count");
      throw ndlexceptions::FileFormatError(fileName, err);
    }

    vector<double> row(static_cast<size_t>(nCols));
    for(int j = 0; j < nCols; ++j)
    {
      if(!csvTokenIsNumeric(fields[static_cast<size_t>(j)]))
      {
        ndlexceptions::StreamFormatError err("non-numeric CSV token");
        throw ndlexceptions::FileFormatError(fileName, err);
      }
      row[static_cast<size_t>(j)] = atof(fields[static_cast<size_t>(j)].c_str());
    }
    rows.push_back(row);
  }
  in.close();

  if(rows.empty() || nCols < 2)
  {
    ndlexceptions::StreamFormatError err("empty or incomplete CSV");
    throw ndlexceptions::FileFormatError(fileName, err);
  }

  const int numData = static_cast<int>(rows.size());
  const int inputDim = nCols - 1;
  if(verbosity > 1)
  {
    cout << "Data number of features: " << inputDim << endl;
    cout << "Number of data: " << numData << endl;
  }

  X.resize(numData, inputDim);
  y.resize(numData, 1);
  for(int i = 0; i < numData; ++i)
  {
    for(int j = 0; j < inputDim; ++j)
      X.setVal(rows[static_cast<size_t>(i)][static_cast<size_t>(j)], i, j);
    y.setVal(rows[static_cast<size_t>(i)][static_cast<size_t>(inputDim)], i, 0);
  }
  DIMENSIONMATCH(X.getRows() == y.getRows());
  return 0;
}

void CClctrl::readData(CMatrix& X, CMatrix& y, const string fileName)
{
  string m = getMode();
  setMode("file");
  if(verbosity>1)
    cout << "Loading data." << endl;

  // Prefer extension so CLI -f (model format) does not force SVML for .csv paths.
  int dataFmt = dataFormatFromExtension(fileName);
  if(dataFmt < 0)
    dataFmt = fileFormat;

  switch(dataFmt)
  {
  case 0: /// svmlight file format.
    readSvmlDataFile(X, y, fileName);
    break;
  case 1: /// Matlab file format.
#ifdef _NDLMATLAB
    X.readMatlabFile(fileName, "X");
    y.readMatlabFile(fileName, "y");
#else
    throw ndlexceptions::MatlabInterfaceError("MATLAB not incorporated at compile time");
#endif
    break;
  case 2: /// dense CSV / TSV (last column = y).
    readCsvDataFile(X, y, fileName);
    break;
  default:
    exitError("Unrecognised file format number.");
    
  }
  if(verbosity>1)
    cout << "Data set loaded." << endl;
  setMode(m);
}
void CClctrl::exitNormal()
{
#ifdef _DEBUG
#ifdef _MSC_VER
  // For debugging under visual studio, to prevent window disappearing.
  waitForSpace();
#endif
#endif
  exit(0);
}
void CClctrl::exitError(const string error)
{
  cerr << error << endl << endl;
  waitForSpace();
  helpInfo();
  exit(1);
}

void CClctrl::writeGnuplotStyleHeader(ostream& out)
{
  // Colours match GPy/plotting/Tango.py (matplotlib defaults in matplot_dep/defaults.py).
  out << "# GPc gnuplot style (GPy Tango palette)" << endl;
  out << "# mean/CI: mediumBlue #3465a4, lightBlue #729fcf, darkBlue #204a87" << endl;
  out << "# data: Aluminium6 #2e3436; inducing: darkRed #a40000" << endl;
  out << "# pos/neg/unlab: mediumGreen #73d216 / mediumOrange #f57900 / mediumPurple #75507b" << endl;
  out << "set style line 1 lc rgb \"#3465a4\" lw 2  # mean" << endl;
  out << "set style line 2 lc rgb \"#204a87\" lw 1  # CI edge" << endl;
  out << "set style line 3 lc rgb \"#2e3436\" pt 2   # data" << endl;
  out << "set style line 4 lc rgb \"#a40000\" pt 7   # inducing / active" << endl;
  out << "set style line 5 lc rgb \"#73d216\" pt 7   # positive" << endl;
  out << "set style line 6 lc rgb \"#f57900\" pt 7   # negative" << endl;
  out << "set style line 7 lc rgb \"#75507b\" pt 5   # unlabelled" << endl;
  out << "set style line 8 lc rgb \"#3465a4\" lw 2  # decision" << endl;
  out << "set style line 9 lc rgb \"#729fcf\" lw 1  # contours" << endl;
  out << "set style fill transparent solid 0.25 noborder" << endl;
  out << "set key top left Left reverse samplen 1 spacing 1" << endl;
  out << "set border 3" << endl;
  out << "set tics nomirror out" << endl;
  out << "set grid lc rgb \"#d3d7cf\" lw 1" << endl;
}

void CClctrl::writeGnuplotTermSelect(ostream& out)
{
  // Do not honour the GNUTERM environment variable: many shells still export
  // GNUTERM=x11, which Homebrew gnuplot no longer provides.
  out << "# Terminal: interactive qt by default (Homebrew gnuplot)." << endl;
  out << "# File output (any one of):" << endl;
  out << "#   gnuplot -e \"GPLOT_OUT='name.pdf'\" name_plot.gp   # .pdf/.svg/.png from extension" << endl;
  out << "#   gnuplot -e \"GPLOT_PDF='name.pdf'\" name_plot.gp" << endl;
  out << "#   gnuplot -e \"GPLOT_SVG='name.svg'\" name_plot.gp" << endl;
  out << "#   gnuplot -e \"GPLOT_PNG='name.png'\" name_plot.gp" << endl;
  out << "# Override interactive term: gnuplot -e \"GPC_GNUTERM='aqua'\" name_plot.gp" << endl;
  out << "gpc_file = 0" << endl;
  out << "if (exists(\"GPLOT_OUT\")) {" << endl;
  out << "  gpc_out = GPLOT_OUT" << endl;
  out << "  gpc_file = 1" << endl;
  out << "  if (strstrt(gpc_out, \".pdf\") > 0 || strstrt(gpc_out, \".PDF\") > 0) {" << endl;
  out << "    set terminal pdfcairo size 9in,6in enhanced font \",11\"" << endl;
  out << "  } else if (strstrt(gpc_out, \".svg\") > 0 || strstrt(gpc_out, \".SVG\") > 0) {" << endl;
  out << "    set terminal svg size 900,600 enhanced dynamic font \",11\"" << endl;
  out << "  } else {" << endl;
  out << "    set terminal pngcairo size 900,600 enhanced font \",11\"" << endl;
  out << "  }" << endl;
  out << "  set output gpc_out" << endl;
  out << "} else if (exists(\"GPLOT_PDF\")) {" << endl;
  out << "  gpc_file = 1" << endl;
  out << "  set terminal pdfcairo size 9in,6in enhanced font \",11\"" << endl;
  out << "  set output GPLOT_PDF" << endl;
  out << "} else if (exists(\"GPLOT_SVG\")) {" << endl;
  out << "  gpc_file = 1" << endl;
  out << "  set terminal svg size 900,600 enhanced dynamic font \",11\"" << endl;
  out << "  set output GPLOT_SVG" << endl;
  out << "} else if (exists(\"GPLOT_PNG\")) {" << endl;
  out << "  gpc_file = 1" << endl;
  out << "  set terminal pngcairo size 900,600 enhanced font \",11\"" << endl;
  out << "  set output GPLOT_PNG" << endl;
  out << "} else {" << endl;
  out << "  if (exists(\"GPC_GNUTERM\")) {" << endl;
  out << "    set terminal GPC_GNUTERM size 900,600 enhanced" << endl;
  out << "  } else {" << endl;
  out << "    set terminal qt size 900,600 enhanced" << endl;
  out << "  }" << endl;
  out << "}" << endl;
}

void CClctrl::writeGnuplotFinish(ostream& out)
{
  out << "if (gpc_file != 0) {" << endl;
  out << "  unset output" << endl;
  out << "} else {" << endl;
  out << "  pause mouse close" << endl;
  out << "}" << endl;
}

