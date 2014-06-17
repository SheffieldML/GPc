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
