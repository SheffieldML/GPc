/* This file provides the command line interface to the GPLVM using the class CClctrl (Command line control) as a base class.

21/10/2005 Extensive updates to the way command line arguments are read so that the argument's long name needn't start with the same letter as the short name. GP-DM model implemented in CGplvm and accessible through a new flag -D.

20/10/2005 Many files updated to accommodate compilation under MSVC. This work was kindly done by William V. Baxter.*/

#ifndef GPLVM_H
#define GPLVM_H
#include <fstream>
#include <cstdlib>
#include "ndlexceptions.h"
#include "ndlstrutil.h"
#include "CMatrix.h"
#include "CKern.h"
#include "CGplvm.h"
#include "CClctrl.h"

int main(int argc, char* argv[]);

class CClgplvm : public CClctrl {
 public:
  CClgplvm(int argc, char** argv);
  void learn();
  void display();
  void gnuplot();

  void helpInfo();
  void helpHeader();
};
void addWangPrior(CKern* kern, double parameter);

#else /* GPLVM_H */
#endif /* GPLVM_H */
