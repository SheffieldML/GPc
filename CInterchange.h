#ifndef CINTERCHANGE_H
#define CINTERCHANGE_H

// CIP-0016 Phase A1: canonical C++ reader/writer for the gp-scuderia GP model
// interchange format (JSON; YAML is converted to JSON at the boundary).
//
// The reader owns the format-to-GPc mappings (lengthscale <-> inverseWidth, the
// ARD split, beta = 1/variance, zero mean, noise = kernel White + likelihood
// variance) and refuses anything it cannot represent exactly. The Python
// façade and the `gp` CLI delegate here.

#include <memory>
#include <string>
#include <vector>

#include "CJson.h"
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"

class CInterchange
{
 public:
  // Wire-format version this reader accepts.
  static const char* schemaVersion() { return "0.1"; }

  // Build a model from a parsed document. Throws ndlexceptions::Error naming the
  // offending field for unknown types, dimension mismatches, wrong schema
  // version, missing data, truncated parameter lists, or unsupported features.
  static std::unique_ptr<CInterchange> fromJson(const CJson& doc);
  static std::unique_ptr<CInterchange> fromFile(const std::string& path);

  // Document describing the current model: structure and extension (x-*) fields
  // come from the document that was read; parameter values are read back from
  // the model. A stale x-reference block is dropped.
  CJson toJson() const;
  void writeFile(const std::string& path) const;

  // Describe an existing GPc model (e.g. read from a .model file) as a document.
  // Refuses anything the wire format cannot carry exactly: unsupported kernels,
  // White components (their noise semantics differ from the wire format in
  // .model files), centred or scaled targets, multi-output, PITC. The result is
  // verified by rebuilding it and comparing log likelihoods.
  static CJson exportModel(const CGp& model);

  // Write the model as a GPc .model stream. Refuses exact models with a White
  // kernel: the stream does not record the additive-noise opt-in, so reading it
  // back would change the likelihood.
  void writeGpModel(const std::string& path, const std::string& comment = "") const;

  CGp& gp() { return *pgp; }
  const CGp& gp() const { return *pgp; }
  const CMatrix& inputs() const { return X; }
  const CMatrix& targets() const { return Y; }

  // log p(y) of the model as parameterised (the negated GPc objective).
  double logLikelihood() const;

  // Predictive mean and variance of y_* (latent plus likelihood noise), one value
  // per row of Xnew for single-output models.
  void predict(const CMatrix& Xnew, std::vector<double>& mean, std::vector<double>& var) const;

  CInterchange(const CInterchange&) = delete;
  CInterchange& operator=(const CInterchange&) = delete;
  ~CInterchange();

 private:
  CInterchange() {}

  CJson doc;                 // source document (structure + extension fields)
  CMatrix X, Y, Z;
  std::unique_ptr<CCmpndKern> kern;
  std::unique_ptr<CGaussianNoise> noise;
  bool hasWhite = false;
  double targetJacobian = 0.0;  // -N log s of a normalizer, added to the reported log likelihood
  std::vector<CKern*> leaves;  // atomic kernels in document order (owned by kern)
  std::unique_ptr<CGp> pgp;    // declared last: destroyed first

  void writeKernelParams(CJson& node, size_t& leafIndex) const;
};

#endif
