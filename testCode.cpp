// Self-contained smoke test for the matrix algebra and special functions that the
// GP code leans on most.  Needs no fixtures (no MATLAB or matio): every check is
// analytic (A*inv(A) = I, U'U = B) or against reference values computed with
// scipy.special.  The fixture-driven tests are testMatrix and testNdlutil.
#include <cmath>
#include <iostream>
#include "CMatrix.h"
using namespace std;

static int failures = 0;

static void check(bool ok, const char* what)
{
  if(ok)
    cout << "PASS: " << what << endl;
  else
  {
    cout << "FAILURE: " << what << endl;
    failures++;
  }
}

static void checkClose(double got, double want, double tol, const char* what)
{
  bool ok = fabs(got - want) <= tol * (1.0 + fabs(want));
  if(!ok)
    cout << "  got " << got << ", want " << want << endl;
  check(ok, what);
}

int main()
{
  const unsigned int n = 3;
  CMatrix identity(n, n);
  identity.eye();

  // General (non-symmetric) matrix: column-major values.
  double a[9] = {1.1, -1, 3, 4, 5.4, 6, 7, 8.2, 9};
  CMatrix A0(n, n, a);

  CMatrix Ainv(A0);
  Ainv.inv();
  check(multiply(A0, Ainv).equals(identity, 1e-10), "A * inv(A) = I (in-place inv)");
  check(multiply(A0, inv(A0)).equals(identity, 1e-10), "A * inv(A) = I (inv function)");
  check(inv(Ainv).equals(A0, 1e-10), "inv(inv(A)) = A");

  // Symmetric positive definite matrix: upper Cholesky factor U with U'U = B.
  double b[9] = {18.0455,  4.9297,  3.6417,
                  4.9297,  5.9624, -4.7233,
                  3.6417, -4.7233,  7.8203};
  CMatrix B(n, n, b);
  B.setSymmetric(true);
  CMatrix U = chol(B);
  CMatrix UtU = multiply(U, "t", U, "n");
  check(UtU.equals(B, 1e-10), "chol(B)' * chol(B) = B");
  bool upper = true;
  for(unsigned int i = 1; i < n; i++)
    for(unsigned int j = 0; j < i; j++)
      if(U.getVal(i, j) != 0.0)
        upper = false;
  check(upper, "chol(B) is upper triangular");

  // Special functions at x = 1.03 (scipy.special.digamma / gammaln / gamma).
  const double x = 1.03;
  checkClose(ndlutil::digamma(x), -0.5289210872854305, 1e-10, "digamma(1.03)");
  checkClose(ndlutil::gammaln(x), -0.016586853877569738, 1e-10, "gammaln(1.03)");
  checkClose(ndlutil::gamma(x), 0.9835499505538241, 1e-10, "gamma(1.03)");
  checkClose(exp(ndlutil::gammaln(x)), ndlutil::gamma(x), 1e-10, "exp(gammaln) = gamma");

  cout << "Number of failures: " << failures << endl;
  return failures > 0 ? 1 : 0;
}
