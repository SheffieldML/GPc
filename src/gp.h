#ifndef GP_H
#define GP_H
#include <fstream>
#include <cstdlib>
#include "ndlexceptions.h"
#include "ndlstrutil.h"
#include "CMatrix.h"
#include "CKern.h"
#include "CGp.h"
#include "CClctrl.h"

int main(int argc, char* argv[]);

class CClgp : public CClctrl {
 public:
  CClgp(int argc, char** argv);
  void learn();
  void relearn();
  void display();
  void gnuplot();

  void helpInfo();
  void helpHeader();
};
//void addWangPrior(CKern* kern, double parameter);

#else /* GP_H */
#endif /* GP_H */
