#include "CJson.h"
#include "ndlexceptions.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#ifdef GPC_YAML
#include <yaml-cpp/yaml.h>
#endif

using namespace std;

namespace {

class Parser
{
 public:
  explicit Parser(const string& t) : text(t), pos(0) {}

  CJson parseDocument()
  {
    skipWs();
    CJson v = parseValue(0);
    skipWs();
    if(pos != text.size())
      fail("unexpected trailing characters");
    return v;
  }

 private:
  const string& text;
  size_t pos;

  [[noreturn]] void fail(const string& what) const
  {
    ostringstream os;
    os << "JSON parse error at byte " << pos << ": " << what;
    throw ndlexceptions::Error(os.str());
  }

  void skipWs()
  {
    while(pos < text.size())
    {
      const char c = text[pos];
      if(c == ' ' || c == '\t' || c == '\n' || c == '\r')
        ++pos;
      else
        break;
    }
  }

  bool matchWord(const char* w)
  {
    size_t n = 0;
    while(w[n])
      ++n;
    if(text.compare(pos, n, w) == 0)
    {
      pos += n;
      return true;
    }
    return false;
  }

  CJson parseValue(int depth)
  {
    if(depth > 256)
      fail("nesting too deep");
    if(pos >= text.size())
      fail("unexpected end of input");
    const char c = text[pos];
    if(c == '{')
      return parseObject(depth);
    if(c == '[')
      return parseArray(depth);
    if(c == '"')
      return CJson::makeString(parseString());
    if(c == 't')
    {
      if(matchWord("true"))
        return CJson::makeBool(true);
      fail("invalid literal");
    }
    if(c == 'f')
    {
      if(matchWord("false"))
        return CJson::makeBool(false);
      fail("invalid literal");
    }
    if(c == 'n')
    {
      if(matchWord("null"))
        return CJson();
      fail("invalid literal");
    }
    return parseNumber();
  }

  CJson parseNumber()
  {
    const size_t start = pos;
    if(pos < text.size() && text[pos] == '-')
      ++pos;
    size_t digits = 0;
    while(pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
    {
      ++pos;
      ++digits;
    }
    if(digits == 0)
      fail("invalid number");
    if(pos < text.size() && text[pos] == '.')
    {
      ++pos;
      size_t frac = 0;
      while(pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
      {
        ++pos;
        ++frac;
      }
      if(frac == 0)
        fail("invalid number");
    }
    if(pos < text.size() && (text[pos] == 'e' || text[pos] == 'E'))
    {
      ++pos;
      if(pos < text.size() && (text[pos] == '+' || text[pos] == '-'))
        ++pos;
      size_t ex = 0;
      while(pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
      {
        ++pos;
        ++ex;
      }
      if(ex == 0)
        fail("invalid number");
    }
    const string tok = text.substr(start, pos - start);
    const double v = strtod(tok.c_str(), 0);
    if(!std::isfinite(v))
      fail("number out of range");
    return CJson::makeNumber(v);
  }

  static void appendUtf8(string& out, unsigned int cp)
  {
    if(cp < 0x80)
      out += static_cast<char>(cp);
    else if(cp < 0x800)
    {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    else
    {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  string parseString()
  {
    ++pos;  // opening quote
    string out;
    while(true)
    {
      if(pos >= text.size())
        fail("unterminated string");
      const char c = text[pos++];
      if(c == '"')
        return out;
      if(static_cast<unsigned char>(c) < 0x20)
        fail("control character in string");
      if(c != '\\')
      {
        out += c;
        continue;
      }
      if(pos >= text.size())
        fail("unterminated escape");
      const char e = text[pos++];
      switch(e)
      {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u':
        {
          if(pos + 4 > text.size())
            fail("truncated \\u escape");
          unsigned int cp = 0;
          for(int i = 0; i < 4; ++i)
          {
            const char h = text[pos++];
            cp <<= 4;
            if(h >= '0' && h <= '9') cp |= static_cast<unsigned int>(h - '0');
            else if(h >= 'a' && h <= 'f') cp |= static_cast<unsigned int>(h - 'a' + 10);
            else if(h >= 'A' && h <= 'F') cp |= static_cast<unsigned int>(h - 'A' + 10);
            else fail("invalid \\u escape");
          }
          if(cp >= 0xD800 && cp <= 0xDFFF)
            fail("surrogate pairs are not supported");
          appendUtf8(out, cp);
          break;
        }
        default:
          fail("invalid escape");
      }
    }
  }

  CJson parseArray(int depth)
  {
    ++pos;  // [
    CJson arr = CJson::makeArray();
    skipWs();
    if(pos < text.size() && text[pos] == ']')
    {
      ++pos;
      return arr;
    }
    while(true)
    {
      skipWs();
      arr.push(parseValue(depth + 1));
      skipWs();
      if(pos >= text.size())
        fail("unterminated array");
      if(text[pos] == ',')
      {
        ++pos;
        continue;
      }
      if(text[pos] == ']')
      {
        ++pos;
        return arr;
      }
      fail("expected ',' or ']'");
    }
  }

  CJson parseObject(int depth)
  {
    ++pos;  // {
    CJson obj = CJson::makeObject();
    skipWs();
    if(pos < text.size() && text[pos] == '}')
    {
      ++pos;
      return obj;
    }
    while(true)
    {
      skipWs();
      if(pos >= text.size() || text[pos] != '"')
        fail("expected string key");
      const string key = parseString();
      if(obj.has(key))
        fail("duplicate key '" + key + "'");
      skipWs();
      if(pos >= text.size() || text[pos] != ':')
        fail("expected ':'");
      ++pos;
      skipWs();
      obj.set(key, parseValue(depth + 1));
      skipWs();
      if(pos >= text.size())
        fail("unterminated object");
      if(text[pos] == ',')
      {
        ++pos;
        continue;
      }
      if(text[pos] == '}')
      {
        ++pos;
        return obj;
      }
      fail("expected ',' or '}'");
    }
  }
};

void escapeString(string& out, const string& s)
{
  out += '"';
  for(size_t i = 0; i < s.size(); ++i)
  {
    const char c = s[i];
    switch(c)
    {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if(static_cast<unsigned char>(c) < 0x20)
        {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
          out += buf;
        }
        else
          out += c;
    }
  }
  out += '"';
}

string numberToString(double v)
{
  if(!std::isfinite(v))
    throw ndlexceptions::Error("JSON cannot represent a non-finite number");
  char buf[40];
  // 17 significant digits round-trips any double.
  snprintf(buf, sizeof(buf), "%.17g", v);
  // Prefer the shortest representation that still round-trips.
  for(int prec = 1; prec < 17; ++prec)
  {
    char shortBuf[40];
    snprintf(shortBuf, sizeof(shortBuf), "%.*g", prec, v);
    if(strtod(shortBuf, 0) == v)
      return shortBuf;
  }
  return buf;
}

}  // namespace

CJson CJson::makeBool(bool v)
{
  CJson j;
  j.kind = Bool;
  j.boolVal = v;
  return j;
}

CJson CJson::makeNumber(double v)
{
  CJson j;
  j.kind = Number;
  j.numVal = v;
  return j;
}

CJson CJson::makeString(const string& v)
{
  CJson j;
  j.kind = String;
  j.strVal = v;
  return j;
}

CJson CJson::makeArray()
{
  CJson j;
  j.kind = Array;
  return j;
}

CJson CJson::makeObject()
{
  CJson j;
  j.kind = Object;
  return j;
}

CJson CJson::parse(const string& text)
{
  Parser p(text);
  return p.parseDocument();
}

#ifdef GPC_YAML
namespace {
// YAML core-schema scalars: quoted scalars are strings; plain ones are null,
// bool, number or string. yaml-cpp marks quoted scalars with the tag "!".
CJson fromYamlNode(const YAML::Node& n)
{
  switch(n.Type())
  {
  case YAML::NodeType::Null:
    return CJson();
  case YAML::NodeType::Scalar:
  {
    const string& t = n.Scalar();
    if(n.Tag() == "!")
      return CJson::makeString(t);
    if(t == "null" || t == "Null" || t == "NULL" || t == "~" || t.empty())
      return CJson();
    if(t == "true" || t == "True" || t == "TRUE")
      return CJson::makeBool(true);
    if(t == "false" || t == "False" || t == "FALSE")
      return CJson::makeBool(false);
    if(t == ".inf" || t == ".Inf" || t == ".INF" || t == "+.inf")
      throw ndlexceptions::Error("YAML: infinity is not representable in the interchange format");
    if(t == ".nan" || t == ".NaN" || t == ".NAN" || t == "-.inf")
      throw ndlexceptions::Error("YAML: non-finite numbers are not representable in the interchange format");
    const char* b = t.c_str();
    char* e = 0;
    const double v = strtod(b, &e);
    if(e != b && *e == '\0' && !std::isspace(static_cast<unsigned char>(t[0])))
      return CJson::makeNumber(v);
    return CJson::makeString(t);
  }
  case YAML::NodeType::Sequence:
  {
    CJson a = CJson::makeArray();
    for(size_t i = 0; i < n.size(); ++i)
      a.push(fromYamlNode(n[i]));
    return a;
  }
  case YAML::NodeType::Map:
  {
    CJson o = CJson::makeObject();
    for(YAML::const_iterator it = n.begin(); it != n.end(); ++it)
      o.set(it->first.as<string>(), fromYamlNode(it->second));
    return o;
  }
  default:
    throw ndlexceptions::Error("YAML: unsupported node type");
  }
}
}  // namespace
#endif

static bool hasYamlExtension(const string& path)
{
  const size_t dot = path.rfind('.');
  if(dot == string::npos)
    return false;
  const string ext = path.substr(dot);
  return ext == ".yaml" || ext == ".yml";
}

CJson CJson::parseFile(const string& path)
{
  if(hasYamlExtension(path))
  {
#ifdef GPC_YAML
    try
    {
      return fromYamlNode(YAML::LoadFile(path));
    }
    catch(const YAML::BadFile&)
    {
      throw ndlexceptions::FileReadError(path);
    }
    catch(const YAML::Exception& e)
    {
      throw ndlexceptions::Error("YAML parse error in " + path + ": " + e.what());
    }
#else
    throw ndlexceptions::Error("'" + path + "' is YAML, but this build has no YAML reader. "
                               "Rebuild with YAML=1 (needs yaml-cpp) or convert to JSON first "
                               "(gp-scuderia: gp_scuderia.io.to_json_twin).");
#endif
  }
  ifstream in(path.c_str(), ios::in | ios::binary);
  if(!in)
    throw ndlexceptions::FileReadError(path);
  ostringstream ss;
  ss << in.rdbuf();
  return parse(ss.str());
}

bool CJson::asBool() const
{
  if(kind != Bool)
    throw ndlexceptions::Error("JSON value is not a boolean");
  return boolVal;
}

double CJson::asNumber() const
{
  if(kind != Number)
    throw ndlexceptions::Error("JSON value is not a number");
  return numVal;
}

const string& CJson::asString() const
{
  if(kind != String)
    throw ndlexceptions::Error("JSON value is not a string");
  return strVal;
}

size_t CJson::size() const
{
  if(kind != Array && kind != Object)
    throw ndlexceptions::Error("JSON value is not an array or object");
  return items.size();
}

const CJson& CJson::at(size_t i) const
{
  if(kind != Array)
    throw ndlexceptions::Error("JSON value is not an array");
  if(i >= items.size())
    throw ndlexceptions::Error("JSON array index out of range");
  return items[i];
}

CJson& CJson::at(size_t i)
{
  if(kind != Array)
    throw ndlexceptions::Error("JSON value is not an array");
  if(i >= items.size())
    throw ndlexceptions::Error("JSON array index out of range");
  return items[i];
}

void CJson::push(const CJson& v)
{
  if(kind != Array)
    throw ndlexceptions::Error("JSON value is not an array");
  items.push_back(v);
}

bool CJson::has(const string& key) const
{
  if(kind != Object)
    return false;
  for(size_t i = 0; i < objKeys.size(); ++i)
    if(objKeys[i] == key)
      return true;
  return false;
}

const CJson& CJson::at(const string& key) const
{
  if(kind != Object)
    throw ndlexceptions::Error("JSON value is not an object (looking up '" + key + "')");
  for(size_t i = 0; i < objKeys.size(); ++i)
    if(objKeys[i] == key)
      return items[i];
  throw ndlexceptions::Error("missing JSON key '" + key + "'");
}

CJson& CJson::at(const string& key)
{
  if(kind != Object)
    throw ndlexceptions::Error("JSON value is not an object (looking up '" + key + "')");
  for(size_t i = 0; i < objKeys.size(); ++i)
    if(objKeys[i] == key)
      return items[i];
  throw ndlexceptions::Error("missing JSON key '" + key + "'");
}

void CJson::set(const string& key, const CJson& v)
{
  if(kind != Object)
    throw ndlexceptions::Error("JSON value is not an object");
  for(size_t i = 0; i < objKeys.size(); ++i)
    if(objKeys[i] == key)
    {
      items[i] = v;
      return;
    }
  objKeys.push_back(key);
  items.push_back(v);
}

bool CJson::erase(const string& key)
{
  if(kind != Object)
    return false;
  for(size_t i = 0; i < objKeys.size(); ++i)
    if(objKeys[i] == key)
    {
      objKeys.erase(objKeys.begin() + static_cast<long>(i));
      items.erase(items.begin() + static_cast<long>(i));
      return true;
    }
  return false;
}

void CJson::dumpTo(string& out, int indent, int depth) const
{
  const bool pretty = indent >= 0;
  const string pad = pretty ? string(static_cast<size_t>(indent * (depth + 1)), ' ') : "";
  const string padClose = pretty ? string(static_cast<size_t>(indent * depth), ' ') : "";
  switch(kind)
  {
    case Null: out += "null"; break;
    case Bool: out += boolVal ? "true" : "false"; break;
    case Number: out += numberToString(numVal); break;
    case String: escapeString(out, strVal); break;
    case Array:
    {
      if(items.empty())
      {
        out += "[]";
        break;
      }
      // Keep arrays of scalars on one line: matrices stay readable.
      bool scalars = true;
      for(size_t i = 0; i < items.size(); ++i)
        if(items[i].kind == Array || items[i].kind == Object)
          scalars = false;
      out += '[';
      for(size_t i = 0; i < items.size(); ++i)
      {
        if(i)
          out += scalars ? ", " : ",";
        if(pretty && !scalars)
          out += "\n" + pad;
        items[i].dumpTo(out, indent, depth + 1);
      }
      if(pretty && !scalars)
        out += "\n" + padClose;
      out += ']';
      break;
    }
    case Object:
    {
      if(items.empty())
      {
        out += "{}";
        break;
      }
      out += '{';
      for(size_t i = 0; i < items.size(); ++i)
      {
        if(i)
          out += ',';
        if(pretty)
          out += "\n" + pad;
        escapeString(out, objKeys[i]);
        out += pretty ? ": " : ":";
        items[i].dumpTo(out, indent, depth + 1);
      }
      if(pretty)
        out += "\n" + padClose;
      out += '}';
      break;
    }
  }
}

string CJson::dump(int indent) const
{
  string out;
  dumpTo(out, indent, 0);
  if(indent >= 0)
    out += '\n';
  return out;
}

void CJson::dumpFile(const string& path, int indent) const
{
  ofstream out(path.c_str(), ios::out | ios::binary);
  if(!out)
    throw ndlexceptions::FileWriteError(path);
  out << dump(indent);
  if(!out)
    throw ndlexceptions::FileWriteError(path);
}
