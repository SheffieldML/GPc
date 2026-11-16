/* CIP-0012: reusable Cholesky / trsm reverse-mode adjoints.
 * GPc stores upper factors U with K = U^T U (CIP-0005). Murray / GPy use
 * lower L with K = L L^T; map once via L ↔ U^T inside these helpers.
 */
#ifndef CCHOLADJOINT_H
#define CCHOLADJOINT_H

#include "CMatrix.h"

// Unblocked Column-recurrence oracle (GPy cholesky_vjp spirit on lower).
// Ubar cotangent w.r.t. upper U; Kbar returned symmetric (full).
void cholAdjointUpperUnblocked(CMatrix& Kbar, const CMatrix& U, const CMatrix& Ubar);

// Murray Level-3 blocked reverse (production path). blockSize default 128.
void cholAdjointUpperBlocked(CMatrix& Kbar, const CMatrix& U, const CMatrix& Ubar,
                             unsigned int blockSize = 128);

// Adjoint of B = U^{-T} Kuf (one trsm): given Bbar, fill KufBar and Ubar.
// KufBar := U^{-1} Bbar; Ubar := -B KufBar^T (upper, full diagonal).
// cholAdjoint* Kbar uses tril pairing (off-diag = 2 ∂L/∂K_ij); halve off-diag
// before contracting with kernel getGradParams (Legacy Frobenius storage).
void trsmAdjointUT(CMatrix& KufBar, CMatrix& Ubar,
                   const CMatrix& U, const CMatrix& B, const CMatrix& Bbar);

#endif
