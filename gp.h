#ifndef GP_H
#define GP_H
#include <fstream>
#include <cmath>
#include <cstdlib>
#include "ndlexceptions.h"
#include "ndlstrutil.h"
#include "CMatrix.h"
#include "CKern.h"
#include "CGp.h"
#include "CClctrl.h"
#include "CHmc.h"

int main(int argc, char* argv[]);

class CClgp : public CClctrl {
 public:
  CClgp(int argc, char** argv);
  void learn();
  void relearn();
  void sample();
  void display();
  void gnuplot();
  // CIP-0016 A2: interchange format (JSON) commands.
  void learnFrom();
  void exportJson();
  void importJson();

  void helpInfo();
  void helpHeader();
};
//void addWangPrior(CKern* kern, double parameter);

#else /* GP_H */
#endif /* GP_H */
