/* Default text progress for CAdam / CStochasticOptimisable (CIP-001E style). */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#define GPC_STOCH_ISATTY _isatty
#define GPC_STOCH_FILENO _fileno
#else
#include <unistd.h>
#define GPC_STOCH_ISATTY isatty
#define GPC_STOCH_FILENO fileno
#endif

#include "CStochasticOptimisable.h"
#include <cstdio>
#include <memory>
#include <sstream>

namespace {

bool streamIsTty(std::ostream& os)
{
  if(&os != &std::cout && &os != &std::cerr)
    return false;
  FILE* fp = (&os == &std::cerr) ? stderr : stdout;
#ifdef _WIN32
  HANDLE h = GetStdHandle((&os == &std::cerr) ? STD_ERROR_HANDLE
                                              : STD_OUTPUT_HANDLE);
  if(h != INVALID_HANDLE_VALUE)
  {
    DWORD mode = 0;
    if(GetConsoleMode(h, &mode))
      SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
  }
#endif
  return GPC_STOCH_ISATTY(GPC_STOCH_FILENO(fp)) != 0;
}

unsigned int countNewlines(const std::string& block)
{
  unsigned int n = 0;
  for(size_t i = 0; i < block.size(); ++i)
  {
    if(block[i] == '\n')
      ++n;
  }
  return n;
}

}  // namespace

void CStochasticOptimisable::installDefaultTextProgress(
    std::ostream& os, int verbosity, const std::string& optimiserName,
    std::function<std::string()> snapshot)
{
  if(verbosity <= 0)
  {
    clearProgressObserver();
    return;
  }

  // Shared mutable state for ANSI in-place rewrite across Adam steps.
  const auto linesPrinted = std::make_shared<unsigned int>(0);
  const auto useInPlace = std::make_shared<bool>(streamIsTty(os));

  setProgressObserver(
      [&os, verbosity, optimiserName, snapshot, linesPrinted, useInPlace](
          const StochasticOptimiserEvent& ev) {
        if(verbosity <= 0)
          return;

        std::ostringstream status;
        status << "Running " << optimiserName << " · iter " << ev.iteration
               << " · objective " << ev.objectiveEstimate
               << " · batch " << ev.batchSize << '\n';
        std::string block = status.str();
        // Model snapshot is for the live TTY rewrite (paramz-style). On pipes,
        // keep a single status line per Iter so logs stay readable.
        if(*useInPlace && snapshot)
        {
          const std::string snap = snapshot();
          if(!snap.empty())
          {
            block += snap;
            if(snap[snap.size() - 1] != '\n')
              block += '\n';
          }
        }
        if(ev.terminal)
          block += "Warning: Maximum number of iterations has been exceeded\n";

        if(*useInPlace)
        {
          if(*linesPrinted > 0)
            os << "\033[" << *linesPrinted << "A\r";
          os << block;
          os << "\033[J" << std::flush;
          *linesPrinted = countNewlines(block);
          if(ev.terminal)
          {
            os << '\n' << std::flush;
            *linesPrinted = 0;
          }
        }
        else
        {
          os << block << std::flush;
        }
      });
}
