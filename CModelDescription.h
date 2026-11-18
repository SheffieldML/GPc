/* Structured model / parameter description for display (CIP-001E / REQ-000F).
   Persistence (.model streams) is separate; this is for human and façade renderers. */
#ifndef CMODELDESCRIPTION_H
#define CMODELDESCRIPTION_H

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

struct ParamRow
{
  std::string path;
  double value;
  std::string transform;
  bool fixed;
  std::string role;

  ParamRow()
    : path(), value(0.0), transform(), fixed(false), role()
  {
  }
  ParamRow(const std::string& p, double v, const std::string& r = std::string())
    : path(p), value(v), transform(), fixed(false), role(r)
  {
  }
};

struct ModelDescription
{
  std::string title;
  std::vector<std::string> headerLines;
  std::vector<ParamRow> params;
  bool hasLogLikelihood;
  double logLikelihood;

  ModelDescription()
    : title(), headerLines(), params(), hasLogLikelihood(false), logLikelihood(0.0)
  {
  }
};

inline std::string constraintLabel(const ParamRow& row)
{
  if(row.fixed)
    return "fixed";
  if(row.transform == "exp" || row.transform == "negLogLogit")
    return "+ve";
  if(row.transform == "sigmoid")
    return "[0, 1]";
  if(row.transform == "linear" || row.transform.empty())
    return "";
  return row.transform;
}

inline std::string formatParamValue(double val)
{
  std::ostringstream oss;
  const double aval = std::fabs(val);
  if(aval >= 1e4 || (aval < 1e-3 && val != 0.0))
    oss << std::setprecision(4) << val;
  else
    oss << std::setprecision(6) << val;
  return oss.str();
}

/** Paramz-inspired Name / Objective / Parameters table (CIP-001E). */
inline std::ostream& formatModelDescription(const ModelDescription& desc, std::ostream& os,
                                           const double* objectiveOverride = 0)
{
  const std::string title = desc.title.empty() ? std::string("model") : desc.title;
  const size_t nParams = desc.params.size();
  size_t nFree = 0;
  for(size_t i = 0; i < nParams; ++i)
  {
    if(!desc.params[i].fixed)
      ++nFree;
  }

  bool haveObj = false;
  double obj = 0.0;
  if(objectiveOverride)
  {
    obj = *objectiveOverride;
    haveObj = std::isfinite(obj);
  }
  else if(desc.hasLogLikelihood)
  {
    // GPy prints the optimisation objective (negative log likelihood).
    obj = -desc.logLikelihood;
    haveObj = true;
  }

  os << "Name : " << title << std::endl;
  if(haveObj)
    os << "Objective : " << formatParamValue(obj) << std::endl;
  os << "Number of Parameters : " << nParams << std::endl;
  os << "Number of Optimization Parameters : " << nFree << std::endl;
  os << "Updates : True" << std::endl;
  os << "Parameters:" << std::endl;

  size_t nameW = std::string("GP_regression.").size();
  if(nameW < 16)
    nameW = 16;
  for(size_t i = 0; i < nParams; ++i)
  {
    if(desc.params[i].path.size() > nameW)
      nameW = desc.params[i].path.size();
  }
  const size_t valW = 12;
  size_t consW = std::string("constraints").size();
  if(consW < 8)
    consW = 8;
  for(size_t i = 0; i < nParams; ++i)
  {
    const size_t cl = constraintLabel(desc.params[i]).size();
    if(cl > consW)
      consW = cl;
  }

  os << "  " << std::left << std::setw(static_cast<int>(nameW)) << "GP_regression."
     << "  |  " << std::right << std::setw(static_cast<int>(valW)) << "value"
     << "  |  " << std::left << std::setw(static_cast<int>(consW)) << "constraints"
     << std::endl;

  for(size_t i = 0; i < nParams; ++i)
  {
    const ParamRow& p = desc.params[i];
    const std::string valS = formatParamValue(p.value);
    const std::string cons = constraintLabel(p);
    os << "  " << std::left << std::setw(static_cast<int>(nameW)) << p.path
       << "  |  " << std::right << std::setw(static_cast<int>(valW)) << valS
       << "  |  " << std::left << std::setw(static_cast<int>(consW)) << cons
       << std::endl;
  }
  return os;
}

inline std::string modelDescriptionToString(const ModelDescription& desc,
                                            const double* objectiveOverride = 0)
{
  std::ostringstream oss;
  formatModelDescription(desc, oss, objectiveOverride);
  return oss.str();
}

#endif /* CMODELDESCRIPTION_H */
