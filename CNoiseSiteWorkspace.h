/* CIP-0033 Phase D: inference-owned site view (dual-write with CNoise caches). */
#ifndef CNOISESITEWORKSPACE_H
#define CNOISESITEWORKSPACE_H

#include "CMatrix.h"

// Non-owning view target for ADF / predictive site caches that historically
// lived only on CNoise::mu / varSigma. Inference algorithms (IVM, PPA) may
// attach one of these; CNoise::setMus / setVarSigmas dual-write when attached.
// Full strip of likelihood-owned matrices is a follow-on once all readers use
// the workspace (REQ-0019 remaining criterion).
struct CNoiseSiteWorkspace
{
  CMatrix mu;
  CMatrix varSigma;

  void resize(unsigned int n, unsigned int d)
  {
    mu.resize(n, d);
    varSigma.resize(n, d);
  }

  void zeros()
  {
    mu.zeros();
    varSigma.zeros();
  }
};

#endif /* CNOISESITEWORKSPACE_H */
