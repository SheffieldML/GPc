/* CIP-0012: Cholesky / trsm reverse-mode adjoints (upper storage). */
#include "CCholAdjoint.h"
#include "ndlexceptions.h"
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;

namespace {

void upperToLower(CMatrix& L, const CMatrix& U)
{
  const unsigned int n = U.getRows();
  DIMENSIONMATCH(U.isSquare());
  L.resize(n, n);
  L.zeros();
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = 0; j <= i; j++)
      L.setVal(U.getVal(j, i), i, j); // L = U^T
  }
  L.setTriangular(true);
}

void lowerToSymmetric(CMatrix& Kbar, const CMatrix& tril)
{
  const unsigned int n = tril.getRows();
  Kbar.resize(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    Kbar.setVal(tril.getVal(i, i), i, i);
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = tril.getVal(i, j);
      Kbar.setVal(v, i, j);
      Kbar.setVal(v, j, i);
    }
  }
  Kbar.setSymmetric(true);
}

// GPy cholesky_vjp on lower L: dL (tril) -> dK (tril), in place on d.
void cholVjpLowerUnblockedInPlace(CMatrix& d, const CMatrix& L)
{
  const unsigned int n = L.getRows();
  DIMENSIONMATCH(d.dimensionsMatch(L));
  for(int k = (int)n - 1; k >= 0; k--)
  {
    if(k < (int)n - 1)
    {
      // tmp = (Dk + Dk.T) @ Lk
      vector<double> tmp(n - k - 1, 0.0);
      for(unsigned int i = 0; i < n - k - 1; i++)
      {
        double s = 0.0;
        for(unsigned int j = 0; j < n - k - 1; j++)
        {
          const double Dij = d.getVal(k + 1 + i, k + 1 + j);
          const double Dji = d.getVal(k + 1 + j, k + 1 + i);
          s += (Dij + Dji) * L.getVal(k + 1 + j, k);
        }
        tmp[i] = s;
      }
      for(unsigned int i = 0; i < n - k - 1; i++)
      {
        double v = d.getVal(k + 1 + i, k) - tmp[i];
        v /= L.getVal(k, k);
        d.setVal(v, k + 1 + i, k);
      }
      double diagCorr = 0.0;
      for(unsigned int i = 0; i < n - k - 1; i++)
        diagCorr += d.getVal(k + 1 + i, k) * L.getVal(k + 1 + i, k);
      d.setVal(d.getVal(k, k) - diagCorr, k, k);
    }
    d.setVal(d.getVal(k, k) * 0.5 / L.getVal(k, k), k, k);
    // Zero strict upper for cleanliness.
    for(unsigned int j = k + 1; j < n; j++)
      d.setVal(0.0, k, j);
  }
}

void phiTrilHalfDiag(CMatrix& out, const CMatrix& A)
{
  const unsigned int n = A.getRows();
  out.resize(n, n);
  out.zeros();
  for(unsigned int i = 0; i < n; i++)
  {
    out.setVal(0.5 * A.getVal(i, i), i, i);
    for(unsigned int j = 0; j < i; j++)
      out.setVal(A.getVal(i, j), i, j);
  }
  out.setTriangular(true);
}

void cholSymbolicRevLower(CMatrix& dK, const CMatrix& L, const CMatrix& Lbar)
{
  // Murray symbolic: Φ(L^T Lbar) then two triangular solves.
  const unsigned int n = L.getRows();
  CMatrix LtLbar(n, n);
  LtLbar.gemm(L, Lbar, 1.0, 0.0, "t", "n");
  CMatrix P;
  phiTrilHalfDiag(P, LtLbar);
  // S = P + P^T
  CMatrix S(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    S.setVal(2.0 * P.getVal(i, i), i, i); // P_ii already halved → full diag before Φ again
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = P.getVal(i, j);
      S.setVal(v, i, j);
      S.setVal(v, j, i);
    }
  }
  // Actually Φ(P) has half diag; P+P^T should use tril(P)+tril(P)^T with diag = P_ii (already half of original).
  // GPy: P = phi(L.T @ Lbar); return phi(solve(L, solve(L, P+P.T, trans=True).T, trans=True))
  for(unsigned int i = 0; i < n; i++)
  {
    S.setVal(P.getVal(i, i), i, i); // P_ii is already 0.5 * (L^T Lbar)_ii; P+P^T diag = 2*P_ii
    S.setVal(2.0 * P.getVal(i, i), i, i);
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = P.getVal(i, j);
      S.setVal(v, i, j);
      S.setVal(v, j, i);
    }
  }
  CMatrix Y(S);
  // solve L^T Y = S  => Y.trsm(L, 1, "l", "l", "t", "n")
  CMatrix Ltri(L);
  Ltri.setTriangular(true);
  Y.trsm(Ltri, 1.0, "l", "l", "t", "n");
  // solve L Z = Y^T ... GPy: solve_tril(L, solve_tril(L, P+P.T, trans=True).T, trans=True)
  // First: X = solve(L, S, trans=True) i.e. L^T X = S
  // Then: Z = solve(L, X.T, trans=True) i.e. L^T Z = X.T
  CMatrix XT(n, n);
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j < n; j++)
      XT.setVal(Y.getVal(j, i), i, j);
  XT.trsm(Ltri, 1.0, "l", "l", "t", "n");
  phiTrilHalfDiag(dK, XT);
}

}  // namespace

void cholAdjointUpperUnblocked(CMatrix& Kbar, const CMatrix& U, const CMatrix& Ubar)
{
  DIMENSIONMATCH(U.isSquare());
  DIMENSIONMATCH(Ubar.dimensionsMatch(U));
  CMatrix L, Lbar, d;
  upperToLower(L, U);
  upperToLower(Lbar, Ubar);
  d.deepCopy(Lbar);
  cholVjpLowerUnblockedInPlace(d, L);
  lowerToSymmetric(Kbar, d);
}

void cholAdjointUpperBlocked(CMatrix& Kbar, const CMatrix& U, const CMatrix& Ubar,
                             unsigned int blockSize)
{
  DIMENSIONMATCH(U.isSquare());
  DIMENSIONMATCH(Ubar.dimensionsMatch(U));
  if(blockSize < 1)
    throw ndlexceptions::Error("cholAdjointUpperBlocked: blockSize must be >= 1");
  const unsigned int n = U.getRows();
  CMatrix L, d;
  upperToLower(L, U);
  upperToLower(d, Ubar);
  if(n <= blockSize)
  {
    CMatrix dK;
    cholSymbolicRevLower(dK, L, d);
    lowerToSymmetric(Kbar, dK);
    return;
  }
  // Work on lower d in place (Murray blocked / GPy cholesky_vjp_blocked).
  for(int k = (int)n; k >= 0; k -= (int)blockSize)
  {
    const int j = max(0, k - (int)blockSize);
    if(j >= k)
      continue;
    const unsigned int jb = (unsigned)j;
    const unsigned int kb = (unsigned)k;
    const unsigned int nb = kb - jb;          // diagonal block rows
    const unsigned int nbelow = n - kb;       // rows below
    const unsigned int nleft = jb;            // columns left

    // Extract panels into temporaries for BLAS.
    CMatrix D(nb, nb), Dbar(nb, nb);
    for(unsigned int r = 0; r < nb; r++)
      for(unsigned int c = 0; c < nb; c++)
      {
        D.setVal(L.getVal(jb + r, jb + c), r, c);
        Dbar.setVal(d.getVal(jb + r, jb + c), r, c);
      }
    D.setTriangular(true);

    CMatrix Cpanel, Cbar, R, Rbar, B, Bbar;
    if(nbelow > 0 && nb > 0)
    {
      Cpanel.resize(nbelow, nb);
      Cbar.resize(nbelow, nb);
      for(unsigned int r = 0; r < nbelow; r++)
        for(unsigned int c = 0; c < nb; c++)
        {
          Cpanel.setVal(L.getVal(kb + r, jb + c), r, c);
          Cbar.setVal(d.getVal(kb + r, jb + c), r, c);
        }
      // Cbar = solve(D, Cbar.T, trans=True).T  => D^T X = Cbar^T, then Cbar = X^T
      CMatrix CbarT(nb, nbelow);
      for(unsigned int r = 0; r < nb; r++)
        for(unsigned int c = 0; c < nbelow; c++)
          CbarT.setVal(Cbar.getVal(c, r), r, c);
      CbarT.trsm(D, 1.0, "l", "l", "t", "n");
      for(unsigned int r = 0; r < nbelow; r++)
        for(unsigned int c = 0; c < nb; c++)
          Cbar.setVal(CbarT.getVal(c, r), r, c);
    }
    if(nleft > 0)
    {
      R.resize(nb, nleft);
      Rbar.resize(nb, nleft);
      for(unsigned int r = 0; r < nb; r++)
        for(unsigned int c = 0; c < nleft; c++)
        {
          R.setVal(L.getVal(jb + r, c), r, c);
          Rbar.setVal(d.getVal(jb + r, c), r, c);
        }
    }
    if(nbelow > 0 && nleft > 0)
    {
      B.resize(nbelow, nleft);
      Bbar.resize(nbelow, nleft);
      for(unsigned int r = 0; r < nbelow; r++)
        for(unsigned int c = 0; c < nleft; c++)
        {
          B.setVal(L.getVal(kb + r, c), r, c);
          Bbar.setVal(d.getVal(kb + r, c), r, c);
        }
      // Bbar -= Cbar @ R
      Bbar.gemm(Cbar, R, -1.0, 1.0, "n", "n");
    }
    if(nbelow > 0 && nb > 0)
    {
      // Dbar = tril(Dbar) - tril(Cbar.T @ C)
      CMatrix CtC(nb, nb);
      CtC.gemm(Cbar, Cpanel, 1.0, 0.0, "t", "n");
      for(unsigned int r = 0; r < nb; r++)
      {
        for(unsigned int c = 0; c < r; c++)
          Dbar.setVal(Dbar.getVal(r, c) - CtC.getVal(r, c), r, c);
        Dbar.setVal(Dbar.getVal(r, r) - CtC.getVal(r, r), r, r);
        for(unsigned int c = r + 1; c < nb; c++)
          Dbar.setVal(0.0, r, c);
      }
    }
    // Unblocked VJP on diagonal block.
    cholVjpLowerUnblockedInPlace(Dbar, D);

    if(nleft > 0)
    {
      // Rbar -= Cbar.T @ B + (Dbar + Dbar.T) @ R
      if(nbelow > 0)
        Rbar.gemm(Cbar, B, -1.0, 1.0, "t", "n");
      // Dbar is stored lower-triangular; (Dbar+Dbar.T) has diag 2*Dbar_ii.
      CMatrix Dsym(nb, nb);
      Dsym.zeros();
      for(unsigned int r = 0; r < nb; r++)
      {
        Dsym.setVal(2.0 * Dbar.getVal(r, r), r, r);
        for(unsigned int c = 0; c < r; c++)
        {
          const double v = Dbar.getVal(r, c);
          Dsym.setVal(v, r, c);
          Dsym.setVal(v, c, r);
        }
      }
      Rbar.gemm(Dsym, R, -1.0, 1.0, "n", "n");
    }

    // Write panels back into d.
    for(unsigned int r = 0; r < nb; r++)
      for(unsigned int c = 0; c < nb; c++)
        d.setVal(Dbar.getVal(r, c), jb + r, jb + c);
    if(nbelow > 0 && nb > 0)
      for(unsigned int r = 0; r < nbelow; r++)
        for(unsigned int c = 0; c < nb; c++)
          d.setVal(Cbar.getVal(r, c), kb + r, jb + c);
    if(nleft > 0)
      for(unsigned int r = 0; r < nb; r++)
        for(unsigned int c = 0; c < nleft; c++)
          d.setVal(Rbar.getVal(r, c), jb + r, c);
    if(nbelow > 0 && nleft > 0)
      for(unsigned int r = 0; r < nbelow; r++)
        for(unsigned int c = 0; c < nleft; c++)
          d.setVal(Bbar.getVal(r, c), kb + r, c);
  }
  lowerToSymmetric(Kbar, d);
}

void trsmAdjointUT(CMatrix& KufBar, CMatrix& Ubar,
                   const CMatrix& U, const CMatrix& B, const CMatrix& Bbar)
{
  // B = U^{-T} Kuf  ⇒  Kuf = U^T B.
  // Differentiating at fixed Kuf: dB = -U^{-T} (dU)^T B, so
  //   <Bbar, dB> = -<B (U^{-1} Bbar)^T, dU> = -<B KufBar^T, dU>.
  // Thus KufBar = U^{-1} Bbar and Ubar = -B KufBar^T (upper entries; full diagonal).
  DIMENSIONMATCH(U.isSquare());
  DIMENSIONMATCH(B.getRows() == U.getRows());
  DIMENSIONMATCH(Bbar.dimensionsMatch(B));
  const unsigned int m = B.getRows();
  KufBar.deepCopy(Bbar);
  CMatrix Utri(U);
  Utri.setTriangular(true);
  KufBar.trsm(Utri, 1.0, "l", "u", "n", "n"); // KufBar = U^{-1} Bbar

  Ubar.resize(m, m);
  Ubar.gemm(B, KufBar, -1.0, 0.0, "n", "t"); // -B KufBar^T
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < i; j++)
      Ubar.setVal(0.0, i, j);
  Ubar.setTriangular(true);
}
