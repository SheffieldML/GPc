#ifndef NDLASSERT_H
#define NDLASSERT_H

/* CIP-0013 P2 / CIP-002D QW2: GPC_CHECKS selects check level.
 *   0 — assert path (RELEASE=1 / -DNDEBUG removes them)
 *   1 — cheap default: dimensions and API-boundary invariants (inline)
 *   2 — full: + per-element BOUNDCHECK and MATRIXPROPERTIES
 * Default: 1 for non-RELEASE builds; RELEASE=1 sets -DGPC_CHECKS=0 -DNDEBUG.
 * Legacy _NDLCASSERT still forces the assert path. */
#ifndef GPC_CHECKS
#  ifdef _NDLCASSERT
#    define GPC_CHECKS 0
#  else
#    define GPC_CHECKS 1
#  endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define GPC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#  define GPC_UNLIKELY(x) (x)
#endif

#if GPC_CHECKS >= 1
#include "ndlexceptions.h"
#include <stdexcept>

inline void ndlCheckBounds(bool cond)
{
  if(GPC_UNLIKELY(!cond))
    throw std::out_of_range("Bound check failed.");
}
inline void ndlCheckDimension(bool cond)
{
  if(GPC_UNLIKELY(!cond))
    throw ndlexceptions::Error("Dimension match check failed.");
}
inline void ndlCheckCharArguments(bool cond)
{
  if(GPC_UNLIKELY(!cond))
    throw ndlexceptions::Error("Char argument check failed.");
}
inline void ndlCheckMatrixProperties(bool cond)
{
  if(GPC_UNLIKELY(!cond))
    throw ndlexceptions::Error("Matrix properties check failed");
}
inline void ndlCheckSanity(bool cond)
{
  if(GPC_UNLIKELY(!cond))
    throw ndlexceptions::Error("Sanity check failed.");
}
inline void ndlCheckZeroOrPositive(bool cond)
{
  if(GPC_UNLIKELY(!cond))
    throw ndlexceptions::Error("Zero or positive check failed.");
}
#endif /* GPC_CHECKS >= 1 */

#if GPC_CHECKS == 0
#include <cassert>
#define BOUNDCHECK(x) assert(x)
#define DIMENSIONMATCH(x) assert(x)
#define CHARARGUMENTS(x) assert(x)
#define MATRIXPROPERTIES(x) assert(x)
#define SANITYCHECK(x) assert(x)
#define CHECKZEROORPOSITIVE(x) assert(x)
#elif GPC_CHECKS == 1
/* Cheap default: API / dimension invariants only (CIP-002D QW2). */
#define BOUNDCHECK(x) ((void)0)
#define DIMENSIONMATCH(x) ndlCheckDimension(x)
#define CHARARGUMENTS(x) ndlCheckCharArguments(x)
#define MATRIXPROPERTIES(x) ((void)0)
#define SANITYCHECK(x) ndlCheckSanity(x)
#define CHECKZEROORPOSITIVE(x) ndlCheckZeroOrPositive(x)
#else /* GPC_CHECKS >= 2 */
#define BOUNDCHECK(x) ndlCheckBounds(x)
#define DIMENSIONMATCH(x) ndlCheckDimension(x)
#define CHARARGUMENTS(x) ndlCheckCharArguments(x)
#define MATRIXPROPERTIES(x) ndlCheckMatrixProperties(x)
#define SANITYCHECK(x) ndlCheckSanity(x)
#define CHECKZEROORPOSITIVE(x) ndlCheckZeroOrPositive(x)
#endif /* GPC_CHECKS */

#endif /* NDLASSERT_H */
