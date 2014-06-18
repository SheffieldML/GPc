/* This file provides the command line interface to the IVM using the class CClctrl (Command line control) as a base class.

   23/10/2005 Modifications for compiling under Visual C++ by Neil.*/

#ifndef IVM_H
#define IVM_H
#include <fstream>
#include "ndlexceptions.h"
#include "ndlstrutil.h"
#include "CMatrix.h"
#include "CKern.h"
#include "CIvm.h"
#include "CClctrl.h"

int main(int argc, char* argv[]);

class CClivm : public CClctrl {
 public:
  CClivm(int argc, char** argv);
  void relearn();
  void learn();
  void test();
  void logLikelihood();
  void activeSetLogLikelihood();
  void predict();
  void classOneProbabilities();
  void display();
  void gnuplot();

  void helpInfo();
  void helpHeader();
};

#else /* IVM_H */
#endif /* IVM_H */
