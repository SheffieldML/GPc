#include "ndlassert.h"
#include <stdexcept>
#include "ndlexceptions.h"

/* Bodies always compiled so RELEASE and checking objects can link together.
 * Macros in ndlassert.h select assert vs these helpers via GPC_CHECKS. */
void ndlCheckBounds(bool cond)
{
  if(cond)
    return;
  else
  {
    throw std::out_of_range("Bound check failed.");
  }
}
void ndlCheckDimension(bool cond)
{
  if(cond)
    return;
  else
  {
    throw ndlexceptions::Error("Dimension match check failed.");
  }
}
void ndlCheckCharArguments(bool cond)
{
  if(cond)
    return;
  else
  {
    throw ndlexceptions::Error("Char argument check failed.");
  }
}
void ndlCheckMatrixProperties(bool cond)
{
  if(cond)
    return;
  else
  {
    throw ndlexceptions::Error("Matrix properties check failed");
  }
}
void ndlCheckSanity(bool cond)
{
  if(cond)
    return;
  else
  {
    throw ndlexceptions::Error("Sanity check failed.");
  }
}
void ndlCheckZeroOrPositive(bool cond)
{
  if(cond)
    return;
  else
  {
    throw ndlexceptions::Error("Zero or positive check failed.");
  }
}
