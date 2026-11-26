#ifndef CJSON_H
#define CJSON_H

// CIP-0016 Phase A1: a deliberately small JSON value, parser and writer.
// It exists so that GPc can read and write the gp-scuderia interchange format
// without a new build dependency. It is not a general-purpose JSON library:
// numbers are doubles, objects preserve insertion order, and strings support the
// standard escapes (\uXXXX is decoded to UTF-8 for the basic multilingual plane).
// All failures throw ndlexceptions::Error with a byte offset.

#include <cstddef>
#include <string>
#include <vector>

class CJson
{
 public:
  enum Type { Null, Bool, Number, String, Array, Object };

  CJson() : kind(Null), boolVal(false), numVal(0.0) {}
  static CJson makeBool(bool v);
  static CJson makeNumber(double v);
  static CJson makeString(const std::string& v);
  static CJson makeArray();
  static CJson makeObject();

  // Parse a complete document; trailing non-whitespace is an error.
  static CJson parse(const std::string& text);
  static CJson parseFile(const std::string& path);

  // indent < 0 gives compact output; otherwise pretty-printed.
  std::string dump(int indent = 2) const;
  void dumpFile(const std::string& path, int indent = 2) const;

  Type type() const { return kind; }
  bool isNull() const { return kind == Null; }
  bool isBool() const { return kind == Bool; }
  bool isNumber() const { return kind == Number; }
  bool isString() const { return kind == String; }
  bool isArray() const { return kind == Array; }
  bool isObject() const { return kind == Object; }

  bool asBool() const;
  double asNumber() const;
  const std::string& asString() const;

  // Array access.
  size_t size() const;  // array length or object member count
  const CJson& at(size_t i) const;
  CJson& at(size_t i);
  void push(const CJson& v);

  // Object access. at(key) throws if the key is missing.
  bool has(const std::string& key) const;
  const CJson& at(const std::string& key) const;
  CJson& at(const std::string& key);
  // Insert-or-assign, preserving first-insertion order.
  void set(const std::string& key, const CJson& v);
  bool erase(const std::string& key);
  const std::vector<std::string>& keys() const { return objKeys; }

 private:
  Type kind;
  bool boolVal;
  double numVal;
  std::string strVal;
  std::vector<CJson> items;          // array elements or object values
  std::vector<std::string> objKeys;  // object keys, parallel to items

  void dumpTo(std::string& out, int indent, int depth) const;
};

#endif
