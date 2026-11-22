/* CIP-001B: LQ factorisation of wide matrices + reverse-mode adjoint.
 *
 * Reference: M. Seeger, A. Hetzel, Z. Dai, E. Meissner, N. D. Lawrence (2017).
 * Auto-differentiating linear algebra. arXiv:1710.08717
 *   §3.3 Bayesian linear regression — LQ of [I, α^{1/2} X] (avoid Gram+chol)
 *   §4.3 LQ forward / backward (copyltu adjoint used here)
 *
 * Forward: W = L Q with W ∈ R^{m×n}, m ≤ n (e.g. [I, A] or y-augmented).
 * GPc prediction reuses upper R = L^T in LcholC (CIP-0005 upper convention).
 *
 * Hot path: pass Qout=nullptr to skip dorglq; keep Householder (W, tau) and
 * call lqFormQ later only when lqAdjoint needs explicit Q (CoE 2026-11-22).
 */
#ifndef CLQADJOINT_H
#define CLQADJOINT_H

#include "CMatrix.h"
#include <vector>

// Build W = [I_m | A] with A m×nA into W (m × (m+nA)).
void lqBuildStackedIA(CMatrix& W, const CMatrix& A);

// LQ of wide W (m×n, m≤n). On success:
//   W = L_lower Q with Q Q^T = I; Lupper := L_lower^T (sign-normalised diag > 0).
//   Qout is m×n with orthonormal rows (optional; pass nullptr to skip dorglq).
//   tauOut receives Householder τ when non-null (needed for lazy lqFormQ).
//   logAbsDetL = 2 Σ log|L_ii| = log|W W^T|.
// W is overwritten with the LAPACK Householder form of the factorisation.
void lqFactorWide(CMatrix& Lupper, double& logAbsDetL,
                  CMatrix* Qout, CMatrix& W,
                  std::vector<double>* tauOut = 0);

// Convenience: LQ of [I|A], fill Lupper and logAbsDetL; optionally Q / tau / Whh.
void lqFactorIA(CMatrix& Lupper, double& logAbsDetL,
                CMatrix* Qout, const CMatrix& A,
                CMatrix* WhhOut = 0, std::vector<double>* tauOut = 0);

// Form Q via dorglq from Householder left by lqFactorWide/IA (sign-normalised).
void lqFormQ(CMatrix& Qout, const CMatrix& Whh, const std::vector<double>& tau,
             const CMatrix& Lupper);

// Flip signs so diag(Llower) > 0; apply the same flips to rows of Q if non-null.
// If rowFlipped non-null, rowFlipped[i]=true iff row i was negated.
void lqSignNormaliseLower(CMatrix& Llower, CMatrix* Q,
                          std::vector<char>* rowFlipped = 0);

// Adjoint of W = L Q (wide, full row-rank) — Seeger et al. arXiv:1710.08717 §4.3.
// L is m×m lower (or pass upper and set LIsUpper=true — converted internally).
// Q is m×n with Q Q^T = I. Lbar/Qbar are cotangents w.r.t. L and Q.
// Fills Wbar (m×n).
void lqAdjoint(CMatrix& Wbar,
               const CMatrix& L, const CMatrix& Q,
               const CMatrix& Lbar, const CMatrix& Qbar,
               bool LIsUpper = false);

#endif
