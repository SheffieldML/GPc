/* Command-line interface for Bayesian GP-LVM (CIP-0017 / CIP-0030) — ./bgplvm. */
#ifndef BGPLVM_H
#define BGPLVM_H

#include "CClctrl.h"
#include "CBayesGplvm.h"
#include "CSviBayesGplvm.h"
#include "CAdam.h"

int main(int argc, char* argv[]);

class CClbgplvm : public CClctrl
{
public:
  CClbgplvm(int argc, char** argv);
  void learn();
  void display();
  void gnuplot();
  void helpInfo();
  void helpHeader();
};

#endif
