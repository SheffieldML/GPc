/* Structured model / parameter description for display (CIP-001E / REQ-000F).
   Persistence (.model streams) is separate; this is for human and façade renderers. */
#ifndef CMODELDESCRIPTION_H
#define CMODELDESCRIPTION_H

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

inline std::ostream& formatModelDescription(const ModelDescription& desc, std::ostream& os)
{
  if(!desc.title.empty())
    os << desc.title << std::endl;
  for(size_t i = 0; i < desc.headerLines.size(); ++i)
    os << desc.headerLines[i] << std::endl;
  for(size_t i = 0; i < desc.params.size(); ++i)
  {
    const ParamRow& row = desc.params[i];
    if(!row.role.empty())
      os << row.role << " ";
    os << row.path << ": " << row.value;
    if(row.fixed)
      os << " [fixed]";
    if(!row.transform.empty())
      os << " (" << row.transform << ")";
    os << std::endl;
  }
  if(desc.hasLogLikelihood)
    os << "Log likelihood: " << desc.logLikelihood << std::endl;
  return os;
}

inline std::string modelDescriptionToString(const ModelDescription& desc)
{
  std::ostringstream oss;
  formatModelDescription(desc, oss);
  return oss.str();
}

#endif /* CMODELDESCRIPTION_H */
