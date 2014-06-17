#
#  Top Level Makefile for GP
#  Version 0.11
#  July 6, 2005
#  Dec 23, 2008
#  Jan 18, 2009 --- using gcc -MM

include make.ubuntu

all: gp

gp: gp.o CClctrl.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o gp gp.o CGp.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

gp.o: gp.cpp gp.h ndlexceptions.h ndlstrutil.h CMatrix.h ndlassert.h \
  CNdlInterfaces.h ndlutil.h ndlfortran.h lapack.h CKern.h CTransform.h \
  CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h CClctrl.h
	$(CC) -c gp.cpp -o gp.o $(CCFLAGS)

CClctrl.o: CClctrl.cpp CClctrl.h ndlstrutil.h ndlexceptions.h ndlutil.h \
  ndlassert.h ndlfortran.h CMatrix.h CNdlInterfaces.h lapack.h
	$(CC) -c CClctrl.cpp -o CClctrl.o $(CCFLAGS)

CGp.o: CGp.cpp CGp.h CMltools.h ndlassert.h ndlexceptions.h ndlstrutil.h \
  COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h ndlfortran.h \
  lapack.h CKern.h CTransform.h CDataModel.h CDist.h CNoise.h
	$(CC) -c CGp.cpp -o CGp.o $(CCFLAGS)

CNoise.o: CNoise.cpp CNoise.h ndlexceptions.h ndlutil.h ndlassert.h \
  ndlfortran.h ndlstrutil.h CMatrix.h CNdlInterfaces.h lapack.h \
  CTransform.h COptimisable.h CDist.h CKern.h CDataModel.h
	$(CC) -c CNoise.cpp -o CNoise.o $(CCFLAGS)

CKern.o: CKern.cpp CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h \
  CDataModel.h CDist.h
	$(CC) -c CKern.cpp -o CKern.o $(CCFLAGS)

CTransform.o: CTransform.cpp CTransform.h CMatrix.h ndlassert.h \
  ndlexceptions.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h \
  lapack.h
	$(CC) -c CTransform.cpp -o CTransform.o $(CCFLAGS)

COptimisable.o: COptimisable.cpp COptimisable.h CMatrix.h ndlassert.h \
  ndlexceptions.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h \
  lapack.h
	$(CC) -c COptimisable.cpp -o COptimisable.o $(CCFLAGS)

CDist.o: CDist.cpp CDist.h CMatrix.h ndlassert.h ndlexceptions.h \
  CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h \
  CTransform.h
	$(CC) -c CDist.cpp -o CDist.o $(CCFLAGS)

CMatrix.o: CMatrix.cpp CMatrix.h ndlassert.h ndlexceptions.h \
  CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h
	$(CC) -c CMatrix.cpp -o CMatrix.o $(CCFLAGS)

ndlutil.o: ndlutil.cpp ndlutil.h ndlassert.h ndlexceptions.h ndlfortran.h
	$(CC) -c ndlutil.cpp -o ndlutil.o $(CCFLAGS)

ndlstrutil.o: ndlstrutil.cpp ndlstrutil.h ndlexceptions.h
	$(CC) -c ndlstrutil.cpp -o ndlstrutil.o $(CCFLAGS)

ndlassert.o: ndlassert.cpp ndlassert.h ndlexceptions.h
	$(CC) -c ndlassert.cpp -o ndlassert.o $(CCFLAGS)

# Collected FORTRAN utilities.
ndlfortran.o: ndlfortran.f 
	$(FC) -c ndlfortran.f -o ndlfortran.o $(FCFLAGS)


clean:
	rm *.o
