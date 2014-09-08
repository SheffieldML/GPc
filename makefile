#
#  Top Level Makefile for GPLVM
#  Version 0.11
#  July 6, 2005
#  Dec 23, 2008
# dependencies created with gcc -MM XXX.cpp

include make.linux

all: gplvm ivm gp

gplvm: gplvm.o CClctrl.o CGplvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o gplvm gplvm.o CGplvm.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

gplvm.o: gplvm.cpp gplvm.h ndlexceptions.h ndlstrutil.h CMatrix.h \
  ndlassert.h CNdlInterfaces.h ndlutil.h ndlfortran.h lapack.h CKern.h \
  CTransform.h CDataModel.h CDist.h CGplvm.h CMltools.h COptimisable.h \
  CNoise.h CClctrl.h
	$(CC) -c gplvm.cpp -o gplvm.o $(CCFLAGS)

ivm: ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o ivm  ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

ivm.o: ivm.cpp CIvm.h CKern.h CMatrix.h ivm.h CClctrl.h
	$(CC) -c ivm.cpp -o ivm.o $(CCFLAGS)

gp: gp.o CClctrl.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o gp gp.o CGp.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

gp.o: gp.cpp gp.h ndlexceptions.h ndlstrutil.h CMatrix.h ndlassert.h \
  CNdlInterfaces.h ndlutil.h ndlfortran.h lapack.h CKern.h CTransform.h \
  CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h CClctrl.h
	$(CC) -c gp.cpp -o gp.o $(CCFLAGS)

tests: testDist testGp testIvm testKern testMatrix testMltools testNdlutil testNoise testTransform  

testDist: testDist.o CMatrix.o ndlfortran.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testDist testDist.o CMatrix.o ndlfortran.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o $(LDFLAGS) 

testDist.o: testDist.cpp CDist.h CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testDist.cpp -o testDist.o $(CCFLAGS)

testGp: testGp.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o
	$(LD) ${XLINKERFLAGS} -o testGp  testGp.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o $(LDFLAGS)

testGp.o: testGp.cpp CGp.h CKern.h CMatrix.h CClctrl.h
	$(CC) -c testGp.cpp -o testGp.o $(CCFLAGS)
 
testIvm: testIvm.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o
	$(LD) ${XLINKERFLAGS} -o testIvm  testIvm.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o $(LDFLAGS)

testIvm.o: testIvm.cpp CIvm.h CKern.h CMatrix.h CClctrl.h 
	$(CC) -c testIvm.cpp -o testIvm.o $(CCFLAGS)

testKern: testKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testKern testKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o $(LDFLAGS) 

testKern.o: testKern.cpp CKern.h CDist.h CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testKern.cpp -o testKern.o $(CCFLAGS)

testMatrix: testMatrix.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testMatrix testMatrix.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o $(LDFLAGS) 

testMatrix.o: testMatrix.cpp CMatrix.h CClctrl.h
	$(CC) -c testMatrix.cpp  -o testMatrix.o $(CCFLAGS)

testMltools: testMltools.o CMltools.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CClctrl.o 
	$(LD) ${XLINKERFLAGS} -o testMltools  testMltools.o CMltools.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CClctrl.o $(LDFLAGS)

testMltools.o: testMltools.cpp CMltools.h CKern.h CMatrix.h CClctrl.h 
	$(CC) -c testMltools.cpp -o testMltools.o $(CCFLAGS)

testNdlutil: testNdlutil.o ndlutil.o ndlstrutil.o CMatrix.o ndlfortran.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testNdlutil testNdlutil.o ndlutil.o ndlstrutil.o CMatrix.o ndlfortran.o CClctrl.o $(LDFLAGS)

testNdlutil.o: testNdlutil.cpp ndlutil.h CClctrl.h
	$(CC) -c testNdlutil.cpp -o testNdlutil.o $(CCFLAGS)

testNoise: testNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testNoise  testNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o $(LDFLAGS)

testNoise.o: testNoise.cpp CNoise.h CMatrix.h CClctrl.h
	$(CC) -c testNoise.cpp -o testNoise.o $(CCFLAGS)

testTransform: testTransform.o CMatrix.o ndlfortran.o  CTransform.o ndlutil.o ndlstrutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testTransform testTransform.o CMatrix.o ndlfortran.o CTransform.o ndlutil.o ndlstrutil.o CClctrl.o $(LDFLAGS) 

testTransform.o: testTransform.cpp CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testTransform.cpp -o testTransform.o $(CCFLAGS)

CClctrl.o: CClctrl.cpp CClctrl.h ndlstrutil.h ndlexceptions.h ndlutil.h \
  ndlassert.h ndlfortran.h CMatrix.h CNdlInterfaces.h lapack.h
	$(CC) -c CClctrl.cpp -o CClctrl.o $(CCFLAGS)

CGplvm.o: CGplvm.cpp CGplvm.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CGplvm.cpp -o CGplvm.o $(CCFLAGS)

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

ndlassert.o: ndlassert.cpp ndlassert.h ndlexceptions.h
	$(CC) -c ndlassert.cpp -o ndlassert.o $(CCFLAGS)

# Collected FORTRAN utilities.
ndlfortran.o: ndlfortran.f
	$(FC) -c ndlfortran.f -o ndlfortran.o $(FCFLAGS)


clean:
	rm *.o
