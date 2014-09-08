
include make.gutsy

all: testDist testGp testIvm testKern testMatrix testMltools testNdlutil testNoise testTransform  

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

CClctrl.o: CClctrl.cpp CClctrl.h ndlstrutil.h ndlutil.h ndlexceptions.h
	$(CC) -c CClctrl.cpp -o CClctrl.o $(CCFLAGS)

CDist.o: CDist.cpp CTransform.h CMatrix.h
	$(CC) -c CDist.cpp -o CDist.o $(CCFLAGS)

CGp.o: CGp.cpp CGp.h CMatrix.h CNdlInterfaces.h COptimisable.h CMltools.h CDataModel.h
	$(CC) -c CGp.cpp -o CGp.o $(CCFLAGS)

CIvm.o: CIvm.cpp CIvm.h CMatrix.h CNdlInterfaces.h COptimisable.h CMltools.h CDataModel.h
	$(CC) -c CIvm.cpp -o CIvm.o $(CCFLAGS)

CKern.o: CKern.cpp CKern.h CMatrix.h CTransform.h CNdlInterfaces.h
	$(CC) -c CKern.cpp -o CKern.o $(CCFLAGS)

CMatrix.o: CMatrix.cpp CMatrix.h ndlutil.h CNdlInterfaces.h lapack.h ndlfortran.h CDataModel.h
	$(CC) -c CMatrix.cpp -o CMatrix.o $(CCFLAGS)

CMltools.o: CMltools.cpp CMltools.h CMatrix.h CNdlInterfaces.h COptimisable.h CDataModel.h
	$(CC) -c CMltools.cpp -o CMltools.o $(CCFLAGS)

CNoise.o: CNoise.cpp CNoise.h COptimisable.h CTransform.h 
	$(CC) -c CNoise.cpp -o CNoise.o $(CCFLAGS)

COptimisable.o: COptimisable.cpp COptimisable.h ndlfortran.h
	$(CC) -c COptimisable.cpp -o COptimisable.o $(CCFLAGS)

CTransform.o: CTransform.cpp CTransform.h CMatrix.h ndlutil.h
	$(CC) -c CTransform.cpp -o CTransform.o $(CCFLAGS)

ndlutil.o: ndlutil.cpp ndlutil.h ndlfortran.h
	$(CC) -c ndlutil.cpp -o ndlutil.o $(CCFLAGS)

ndlstrutil.o: ndlstrutil.cpp ndlstrutil.h
	$(CC) -c ndlstrutil.cpp -o ndlstrutil.o $(CCFLAGS)

# Collected FORTRAN utilities.
ndlfortran.o: ndlfortran.f 
	$(FC) -c ndlfortran.f -o ndlfortran.o $(FCFLAGS)

clean:
	rm *.o
