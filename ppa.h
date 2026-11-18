/* Command-line interface for Probabilistic Point Assimilation (PPA).
   Restored for CIP-0021; mirrors ivm.h / CClivm. */

#ifndef PPA_H
#define PPA_H
#include <fstream>
#include "ndlexceptions.h"
#include "ndlstrutil.h"
#include "CMatrix.h"
#include "CKern.h"
#include "CPpa.h"
#include "CClctrl.h"

int main(int argc, char* argv[]);

class CClppa : public CClctrl {
 public:
  CClppa(int argc, char** argv);
  void relearn();
  void learn();
  void test();
  void logLikelihood();
  void trainLogLikelihood();
  void predict();
  void classOneProbabilities();
  void display();
  void gnuplot();

  void helpInfo();
  void helpHeader();
};

#else /* PPA_H */
#endif /* PPA_H */
