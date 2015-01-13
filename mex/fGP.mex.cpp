#include <mex.h>
#include <iostream>
#include <sstream>

#include "gp.h"

static CGp *globalModel = NULL;
static CGaussianNoise *globalNoise = NULL;
static CCmpndKern *globalKern = NULL;
static CMatrix *globalXtrain = NULL, *globalYtrain = NULL;

void gpTrain(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ]);
void gpRetrain(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ]);
void gpClear();
void gpQuery(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ]);

/* function [ flag ] = fGP( 'train', X, Y ) */
void mexFunction(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ]) {
	if (nrhs < 1) {
		std::stringstream err;
		err << "Usage: [result] = fGP(command, inputs)\n" <<
				"\tPossible commands are: train, retrain, clear, query\n" <<
				"\tType fGP(command) for further help on a particular command." << std::endl;
		mexErrMsgTxt(err.str().c_str());
	}

	/* COMMAND */
	if ( !mxIsChar(prhs[0]) || mxGetNumberOfElements(prhs[0]) != mxGetN(prhs[0]) ) {
		std::stringstream err;
		err << "command: Expected string (row vector of chars), got " <<
				mxGetM(prhs[0]) << "x" << mxGetN(prhs[0]) << " of " << mxGetClassName(prhs[0]) << '!';
		mexErrMsgTxt(err.str().c_str());
	}
	std::string command(mxArrayToString(prhs[0]));
	if ( command == "" || command == "help" ) {
		std::stringstream err;
		err << "Usage: [result] = fGP(command, inputs)\n" <<
				"\tPossible commands are: help, train, query\n" <<
				"\tType fGP(command) for further help on a particular command." << std::endl;
		mexErrMsgTxt(err.str().c_str());
	} else if (command == "train") {
		gpTrain(nlhs, plhs, nrhs-1, prhs+1);
	} else if (command == "retrain") {
		gpRetrain(nlhs, plhs, nrhs-1, prhs+1);
	} else if (command == "clear") {
		gpClear();
	} else if (command == "query") {
		gpQuery(nlhs, plhs, nrhs-1, prhs+1);
	} else {
		mexErrMsgTxt("Invalid command!");
	}
}


void gpTrain(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ]) {
	unsigned iters = 100;
	unsigned verbose = 1;

	if (nrhs < 3 || nrhs > 4) {
		std::stringstream err;
		err << "Usage: fGP('train', kernel(s), X, Y [, verbose])\n" <<
				"\tPossible kernels: r (RBF), e (Exponential), w (white noise), l (linear), B (bias)\n" <<
				"\tVerbose range: 0...3\n" <<
				"\tExpected 3 inputs, got " << nrhs << '!' << std::endl;
		mexErrMsgTxt(err.str().c_str());
	}

	/* kernel list */
	if ( !mxIsChar(prhs[0]) || mxGetNumberOfElements(prhs[0]) != mxGetN(prhs[0]) ) {
		std::stringstream err;
		err << "kernel list: Expected string (row vector of chars), got " <<
				mxGetM(prhs[0]) << "x" << mxGetN(prhs[0]) << " of " << mxGetClassName(prhs[0]) << '!';
		mexErrMsgTxt(err.str().c_str());
	}
	std::string kernels(mxArrayToString(prhs[0]));

	/* input data X */
	if ( mxIsEmpty(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsDouble(prhs[1]) ) {
		std::stringstream err;
		err << "X data: Expected non-empty array of real doubles, got " <<
				mxGetM(prhs[1]) << "x" << mxGetN(prhs[1]) << " of " << mxGetClassName(prhs[1]) << '!';
		mexErrMsgTxt(err.str().c_str());
	}
	double * Xd = mxGetPr(prhs[1]);

	/* input data Y */
	if ( mxIsEmpty(prhs[2]) || mxIsComplex(prhs[2]) || !mxIsDouble(prhs[2]) || mxGetN(prhs[2]) != 1 || mxGetM(prhs[1]) != mxGetM(prhs[2])) {
		std::stringstream err;
		err << "Y data: Expected vector " << mxGetM(prhs[1]) << "x1 of real doubles, got " <<
				mxGetM(prhs[2]) << "x" << mxGetN(prhs[2]) << " of " << mxGetClassName(prhs[2]) << '!';
		mexErrMsgTxt(err.str().c_str());
	}
	double * Yd = mxGetPr(prhs[2]);

	/* verbose flag */
	if (nrhs > 3) {
		if ( mxGetNumberOfElements(prhs[3]) != 1 || !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || *mxGetPr(prhs[3]) < 0 || *mxGetPr(prhs[3]) > 3) {
			std::stringstream err;
			err << "Flag `verbose`: Expected real double scalar from [0;3], got " <<
					mxGetM(prhs[3]) << "x" << mxGetN(prhs[3]) << " of " << mxGetClassName(prhs[3]) << '!';
			mexErrMsgTxt(err.str().c_str());
		}
		verbose = (unsigned) mxGetScalar(prhs[3]);
	}

	// Both Matlab and GPc use column-major order
	if (globalXtrain) {
		delete globalXtrain;
		delete globalYtrain;
	}
	globalXtrain = new CMatrix(mxGetM(prhs[1]), mxGetN(prhs[1]), Xd);
	globalYtrain = new CMatrix(mxGetM(prhs[2]), mxGetN(prhs[2]), Yd);

	/* training itself */

	// kernels
	if(globalKern) {
		delete globalKern;
	}
	globalKern = new CCmpndKern(*globalXtrain);
	for (unsigned i = 0; i < kernels.size(); ++i) {
		switch (kernels[i]) {
			case 'r':
				globalKern->addKern(new CRbfKern(*globalXtrain));
				break;
			case 'e':
				globalKern->addKern(new CExpKern(*globalXtrain));
				break;
			case 'w':
				globalKern->addKern(new CWhiteKern(*globalXtrain));
				break;
			case 'l':
				globalKern->addKern(new CLinKern(*globalXtrain));
				break;
			case 'B':
				globalKern->addKern(new CBiasKern(*globalXtrain));
				break;
			default:
				std::stringstream err;
				err << "The kernel string ('" << kernels << "') contains illegal character '" <<
						kernels[i] << "' at position " << i+1 << " (unrecognised kernel type)!";
				mexErrMsgTxt(err.str().c_str());
		}
	}

	if (globalNoise) {
		delete globalNoise;
	}
	globalNoise = new CGaussianNoise(globalYtrain);
	globalNoise->setBias(0.0);
	CMatrix scale(1, globalYtrain->getCols(), 1.0);
	CMatrix bias(1, globalYtrain->getCols(), 0.0);
	bias.deepCopy(meanCol(*globalYtrain));
//	if(scaleData) { false by default
//		scale.deepCopy(stdCol(*globalYtrain));
//	}

	int approxType = CGp::FTC;
	int activeSetSize = -1;
	if (globalModel) {
		delete globalModel;
	}
	globalModel = new CGp(globalKern, globalNoise, globalXtrain, approxType, activeSetSize, verbose);

	globalModel->setDefaultOptimiser(CGp::SCG);
	globalModel->setBetaVal(1);
	globalModel->setScale(scale);
	globalModel->setBias(bias);
	globalModel->updateM();
	globalModel->setOutputScaleLearnt(false);
	globalModel->optimise(iters);
}

// basicly the same thing as Train, but no parameter settings, just new data and optimisation
void gpRetrain(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ]) {

	unsigned iters = 100;
	unsigned verbose = 1;

	if (nrhs < 2 || nrhs > 3) {
		std::stringstream err;
		err << "Usage: fGP('retrain', X, Y [, verbose])\n" <<
				"\tVerbose range: 0...3\n" <<
				"\tExpected 2 inputs, got " << nrhs << '!' << std::endl;
		mexErrMsgTxt(err.str().c_str());
	}

	if (!globalModel) {
		mexErrMsgTxt("Model was not previously trained and thus cannot be retrained!");
	}

	/* input data X */
	if ( mxIsEmpty(prhs[0]) || mxIsComplex(prhs[0]) || !mxIsDouble(prhs[0]) || mxGetN(prhs[0]) != globalModel->getInputDim()) {
		std::stringstream err;
		err << "X data: Expected non-empty Mx" << globalModel->getInputDim() << " array of real doubles, got " <<
				mxGetM(prhs[0]) << "x" << mxGetN(prhs[0]) << " of " << mxGetClassName(prhs[0]) << '!';
		mexErrMsgTxt(err.str().c_str());
	}
	double * Xd = mxGetPr(prhs[0]);

	/* input data Y */
	if ( mxIsEmpty(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsDouble(prhs[1]) || mxGetN(prhs[1]) != 1 || mxGetM(prhs[1]) != mxGetM(prhs[0])) {
		std::stringstream err;
		err << "Y data: Expected vector " << mxGetN(prhs[0]) << "x1 of real doubles, got " <<
				mxGetM(prhs[1]) << "x" << mxGetN(prhs[1]) << " of " << mxGetClassName(prhs[1]) << '!';
		mexErrMsgTxt(err.str().c_str());
	}
	double * Yd = mxGetPr(prhs[1]);

	/* verbose flag */
	if (nrhs > 2) {
		if ( mxGetNumberOfElements(prhs[2]) != 1 || !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || *mxGetPr(prhs[2]) < 0 || *mxGetPr(prhs[2]) > 3) {
			std::stringstream err;
			err << "Flag `verbose`: Expected real double scalar from [0;3], got " <<
					mxGetM(prhs[2]) << "x" << mxGetN(prhs[2]) << " of " << mxGetClassName(prhs[2]) << '!';
			mexErrMsgTxt(err.str().c_str());
		}
		verbose = (unsigned) mxGetScalar(prhs[2]);
	}

	if (globalXtrain) {
		delete globalXtrain;
		delete globalYtrain;
	}
	globalXtrain = new CMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), Xd);
	globalYtrain = new CMatrix(mxGetM(prhs[1]), mxGetN(prhs[1]), Yd);

	if (globalNoise) {
		delete globalNoise;
	}
	globalNoise = new CGaussianNoise(globalYtrain);
	globalNoise->setBias(0.0);
	CMatrix scale(1, globalYtrain->getCols(), 1.0);
	CMatrix bias(1, globalYtrain->getCols(), 0.0);
	bias.deepCopy(meanCol(*globalYtrain));
//	if(scaleData) { false by default
//		scale.deepCopy(stdCol(*globalYtrain));
//	}

	int approxType = CGp::FTC;
	int activeSetSize = -1;
	if (globalModel) {
		delete globalModel;
	}
	globalModel = new CGp(globalKern, globalNoise, globalXtrain, approxType, activeSetSize, verbose);

	globalModel->setDefaultOptimiser(CGp::SCG);
	globalModel->setBetaVal(1);
	globalModel->setScale(scale);
	globalModel->setBias(bias);
	globalModel->updateM();
	globalModel->setOutputScaleLearnt(false);
	globalModel->optimise(iters);
}

// clear all the static globals
void gpClear() {
	if (globalModel) {
		delete globalModel;
		globalModel = NULL;
	}
	if (globalNoise) {
		delete globalNoise;
		globalNoise = NULL;
	}
	if (globalKern) {
		delete globalKern;
		globalKern = NULL;
	}
	if (globalXtrain) {
		delete globalXtrain;
		delete globalYtrain;
		globalXtrain = NULL;
		globalYtrain = NULL;
	}
}

void gpQuery(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ]) {
	if (nrhs != 1) {
		std::stringstream err;
		err << "Usage: [Y, varY] = fGP('query', X)\n" <<
				"\tExpected 1 input, got " << nrhs << '!' << std::endl;
		mexErrMsgTxt(err.str().c_str());
	}

	if (!globalModel) {
		mexErrMsgTxt("Model was not previously trained and thus cannot be queried!");
	}

	/* input data X */
	if ( mxIsEmpty(prhs[0]) || mxIsComplex(prhs[0]) || !mxIsDouble(prhs[0]) || mxGetN(prhs[0]) != globalModel->getInputDim()) {
		std::stringstream err;
		err << "X data: Expected non-empty Mx" << globalModel->getInputDim() << " array of real doubles, got " <<
				mxGetM(prhs[0]) << "x" << mxGetN(prhs[0]) << " of " << mxGetClassName(prhs[0]) << '!';
		mexErrMsgTxt(err.str().c_str());
	}
	double * Xd = mxGetPr(prhs[0]);
	unsigned len = mxGetM(prhs[0]);
	CMatrix X(len, mxGetN(prhs[0]), Xd);
	CMatrix Y(len, 1);

	if (nlhs <= 1) {
		globalModel->out(Y, X);
	} else {
		CMatrix Yv(len, 1);
		globalModel->out(Y, Yv, X);
		plhs[1] = mxCreateDoubleMatrix(len, 1, mxREAL);
		double *variance = mxGetPr(plhs[1]);
		memcpy(variance, Yv.getVals(), len * sizeof(double));
	}

	plhs[0] = mxCreateDoubleMatrix(len, 1, mxREAL);
	double *predictions = mxGetPr(plhs[0]);
	memcpy(predictions, Y.getVals(), len * sizeof(double));
}



