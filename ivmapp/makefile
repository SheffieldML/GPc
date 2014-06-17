#
#  Top Level Makefile for IVM
#  Version 0.11
#  May 1, 2005
#  Dec 23, 2008

include make.linux

all: ivm 

ivm: ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o ivm  ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

ivm.o: ivm.cpp CIvm.h CKern.h CMatrix.h ivm.h CClctrl.h
	$(CC) -c ivm.cpp -o ivm.o $(CCFLAGS)

CClctrl.o: CClctrl.cpp ndlstrutil.h CClctrl.h
	$(CC) -c CClctrl.cpp -o CClctrl.o $(CCFLAGS)

CIvm.o: CIvm.cpp CIvm.h CMatrix.h CNdlInterfaces.h COptimisable.h
	$(CC) -c CIvm.cpp -o CIvm.o $(CCFLAGS)

CNoise.o: CNoise.cpp CNoise.h COptimisable.h CTransform.h
	$(CC) -c CNoise.cpp -o CNoise.o $(CCFLAGS)

CKern.o: CKern.cpp CKern.h CMatrix.h CTransform.h CNdlInterfaces.h
	$(CC) -c CKern.cpp -o CKern.o $(CCFLAGS)

CTransform.o: CTransform.cpp CTransform.h CMatrix.h ndlutil.h
	$(CC) -c CTransform.cpp -o CTransform.o $(CCFLAGS)

COptimisable.o: COptimisable.cpp COptimisable.h
	$(CC) -c COptimisable.cpp -o COptimisable.o $(CCFLAGS)

CDist.o: CDist.cpp CTransform.h CMatrix.h
	$(CC) -c CDist.cpp -o CDist.o $(CCFLAGS)

CMatrix.o: CMatrix.cpp CMatrix.h CNdlInterfaces.h ndlutil.h lapack.h ndlfortran.h
	$(CC) -c CMatrix.cpp -o CMatrix.o $(CCFLAGS)

ndlutil.o: ndlutil.cpp ndlutil.h ndlfortran.h
	$(CC) -c ndlutil.cpp -o ndlutil.o $(CCFLAGS)

ndlstrutil.o: ndlstrutil.cpp ndlstrutil.h
	$(CC) -c ndlstrutil.cpp -o ndlstrutil.o $(CCFLAGS)

ndlassert.o: ndlassert.cpp ndlassert.h
	$(CC) -c ndlassert.cpp -o ndlassert.o $(CCFLAGS)

# Collected FORTRAN utilities.
ndlfortran.o: ndlfortran.f
	$(FC) -c ndlfortran.f -o ndlfortran.o $(FCFLAGS)


clean:
	rm *.o
