/* Command-line interface for deep GP (CIP-0035) — ./deepgp. */
#ifndef DEEPGP_H
#define DEEPGP_H

#include "CAdam.h"
#include "CClctrl.h"
#include "CDeepGp.h"

int main(int argc, char* argv[]);

class CCldeepgp : public CClctrl
{
public:
  CCldeepgp(int argc, char** argv);
  void learn();
  void display();
  void helpInfo();
  void helpHeader();
};

#endif
