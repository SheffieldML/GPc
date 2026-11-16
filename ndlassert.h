#ifndef NDLASSERT_H
#define NDLASSERT_H

/* CIP-0013 P2: GPC_CHECKS selects throwing bounds/dimension checks vs assert.
 * Default: on (1) for debug builds; RELEASE=1 sets -DGPC_CHECKS=0 -DNDEBUG.
 * Legacy _NDLCASSERT still forces the assert path. */
#ifndef GPC_CHECKS
#  ifdef _NDLCASSERT
#    define GPC_CHECKS 0
#  else
#    define GPC_CHECKS 1
#  endif
#endif

void ndlCheckBounds(bool cond);
void ndlCheckDimension(bool cond);
void ndlCheckCharArguments(bool cond);
void ndlCheckMatrixProperties(bool cond);
void ndlCheckSanity(bool cond);
void ndlCheckZeroOrPositive(bool cond);

#if GPC_CHECKS
#include "ndlexceptions.h"
#define BOUNDCHECK(x) ndlCheckBounds(x)
#define DIMENSIONMATCH(x) ndlCheckDimension(x)
#define CHARARGUMENTS(x) ndlCheckCharArguments(x)
#define MATRIXPROPERTIES(x) ndlCheckMatrixProperties(x)
#define SANITYCHECK(x) ndlCheckSanity(x)
#define CHECKZEROORPOSITIVE(x) ndlCheckZeroOrPositive(x)
#else
#include <cassert>
#define BOUNDCHECK(x) assert(x)
#define DIMENSIONMATCH(x) assert(x)
#define CHARARGUMENTS(x) assert(x)
#define MATRIXPROPERTIES(x) assert(x)
#define SANITYCHECK(x) assert(x)
#define CHECKZEROORPOSITIVE(x) assert(x)
#endif /* GPC_CHECKS */

#endif /* NDLASSERT_H */
