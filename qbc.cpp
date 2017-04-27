#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include "engine.h"
using namespace std;

#define RED "\e[31m"
#define GREEN "\e[32m"
#define YELLOW "\e[33m"
#define BLUE "\e[34m"
#define MAGENTA "\e[35m"
#define CYAN "\e[36m"
#define LIGHT_GRAY "\e[37m"
#define DARK_GRAY "\e[90m"
#define LIGHT_RED "\e[91m"
#define LIGHT_GREEN "\e[92m"
#define LIGHT_YELLOW "\e[93m"
#define LIGHT_BLUE "\e[94m" 
#define LIGHT_MAGENTA "\e[95m"
#define LIGHT_CYAN "\e[96m"
//case WHITE: // white
#define NORMAL "\e[0m"
#define BOLD "\e[1m"
#define UNDERLINE "\e[4m"

class Oracle {
	public:
		Oracle(int dim) : dim(dim){
			// assert (dim >= 0);
			coefs = new double[dim+1];
		}
		~Oracle() {
			if (coefs != NULL) {
				delete []coefs;
				coefs = NULL;
			}
		}

		void readCoef() {
			for (int i = 0; i < dim+1; i++) {
				cout << "  coef[" << i << "]=";
				cin >> coefs[i];
			}
		}

		void setCoefs(double* cfs) {
			for (int i = 0; i < dim+1; i++)
				coefs[i] = cfs[i];
		}

		int classify(double* vals) {
			double res = 0;
			for (int i = 0; i < dim; i++)
				res += vals[i] * coefs[i];
			res += coefs[dim];
			return (res>=0? 1 : -1);
		}

		void output() {
			for (int i = 0; i < dim+1; i++)
				cout << i << " -- " << coefs[i] << endl;
		}

	private:
		int dim;
		double* coefs;
};



int main() {
	int dim = 2;
	int init_sample_num = 8;

	srand(time(NULL));
	cout << "dim:? ";
	cin >> dim;
	Oracle oracle(dim);
	oracle.readCoef();
	//oracle.output();

	double *cselect = new double [init_sample_num];
	for (int i = 0; i < init_sample_num; i++)
		cselect[i] = i+1;

	double *cdata = new double[dim * init_sample_num];
	for (int i = 0; i < init_sample_num; i++) {
		for (int j = 0; j < dim; j++) {
			cdata[j * init_sample_num + i] = rand() % 100;
		}
	}

	double *clabel = new double[init_sample_num];
	for (int i = 0; i < init_sample_num; i++) {
		double temp[20];
		for (int j = 0; j < dim; j++) {
			temp[j] = cdata[j * init_sample_num + i];
		}

		clabel[i] = oracle.classify(temp);
		if (clabel[i] == 1)
			cout << GREEN;
		else
			cout << RED;
		cout << "init-sample" << i << "---- <";
		for (int k = 0; k < dim; k++) {
			cout << cdata[k * init_sample_num + i];
			if (k < dim-1)
				cout << ", ";
		}
		cout << "> ";
		cout << NORMAL << endl;
	}

	double *ccoef = new double[init_sample_num];
	ccoef[0] = 0;
	for (int i = 0; i < dim; i++)
		ccoef[0] += cdata[i * init_sample_num] * cdata[i * init_sample_num];
	ccoef[0] = sqrt(ccoef[0]);
	ccoef[0] = 1.0 / ccoef[0];

	double cdim[1];
	cdim[0] = dim;
	double cn[1];
	cn[0] = init_sample_num+1;
	double *cx = new double[dim + 1];
	cx[0] = 0;
	double cy[1];
	cy[0] = 1;


	Engine *ep;
	mxArray *select=NULL, *data=NULL, *label=NULL;
	mxArray *D = NULL, *coef = NULL, *errors=NULL, *n=NULL, *new_x=NULL, *w1=NULL, *w2=NULL, *y=NULL, *w=NULL, *aco1=NULL,*aco2=NULL, *cease=NULL;

	if (!(ep = engOpen(NULL))) {
		cerr << "\nCan't start MATLAB engine1\n";
		return EXIT_FAILURE;
	}
	engSetVisible(ep, 0);

	select = mxCreateDoubleMatrix(1, init_sample_num, mxREAL);
	data = mxCreateDoubleMatrix(init_sample_num, dim, mxREAL);
	label = mxCreateDoubleMatrix(init_sample_num, 1, mxREAL);
	coef = mxCreateDoubleMatrix(init_sample_num, 1, mxREAL);
	D = mxCreateDoubleMatrix(1, 1, mxREAL);
	n = mxCreateDoubleMatrix(1, 1, mxREAL);
	w = mxCreateDoubleMatrix(2, 1, mxREAL);
	w1= mxCreateDoubleMatrix(2, 1, mxREAL);
	w2= mxCreateDoubleMatrix(2, 1, mxREAL);
	y = mxCreateDoubleMatrix(1, 1, mxREAL);
	cease = mxCreateDoubleMatrix(1, 1, mxREAL);

	memcpy((void *)mxGetPr(select), (void *)cselect, sizeof(cselect) * init_sample_num);
	memcpy((void *)mxGetPr(data), (void *)cdata, sizeof(cdata) * init_sample_num * dim);
	memcpy((void *)mxGetPr(label), (void *)clabel, sizeof(clabel) * init_sample_num);
	memcpy((void *)mxGetPr(coef), (void *)ccoef, sizeof(ccoef) * init_sample_num);
	memcpy((void *)mxGetPr(n), (void *)cn, sizeof(cn));
	memcpy((void *)mxGetPr(D), (void *)(cdim), sizeof(cdim));


	int nsamples = 30;
	for(int i = init_sample_num + 1; i<50; i++){
		cn[0]=i;
//%function [new_x,X_train,w1,w2] = newx(X_train, Y_train, coef, selection, D)
		memcpy((void *)mxGetPr(n), (void *)cn, sizeof(cn));
		engPutVariable(ep, "X_train", data);
		engPutVariable(ep, "Y_train", label);
		engPutVariable(ep, "coef", coef);
		engPutVariable(ep, "selection", select);
		engPutVariable(ep, "dim", D);
		engEvalString(ep, "newx");

		aco1 = mxCreateDoubleMatrix(i-1,1, mxREAL);
		aco2 = mxCreateDoubleMatrix(i-1,1, mxREAL);
		data = mxCreateDoubleMatrix(i,dim, mxREAL);

		new_x = engGetVariable(ep,"new_x");
		data = engGetVariable(ep,"X_train");
		w1 = engGetVariable(ep,"w1");
		w2 = engGetVariable(ep,"w2");
		aco1 = engGetVariable(ep,"aco1");
		aco2 = engGetVariable(ep,"aco2");
		cease= engGetVariable(ep,"n");

		/**********************************************************************/
		cout << BLUE << "-------------------query" << i << "------------------" << NORMAL;

		if(*mxGetPr(cease) == 50000){
			nsamples = i-1;
			printf("No new_x generated\n");
			break;
		}		
		for(int j=0; j<dim; j++) 
			cx[j] = *(mxGetPr(new_x)+j);
		cy[0] = oracle.classify(cx);

	    if (cy[0] > 0)	cout << RED;
		else cout << GREEN;
		cout << "<";
		for(int j=0; j<dim; j++){ 
			cout << cx[j];
			if (j!=dim-1)
				cout << ", ";
		}
		cout << ">";
		cout << NORMAL;
		//cy[0] = (cx[0]*2-cx[1]>0) ? 1:-1;
		//		cy[0]= somefunction(new_x); //cx[]=new_x; generated label of new_x;

		/**********************************************************************/
//%function [errors,coef,w,selection,Y_train] = predict(X_train, Y_train,w1,w2,coef,ii,selection,y)  
		memcpy((void *)mxGetPr(y), (void *)cy, sizeof(cy));
		engPutVariable(ep, "X_train", data);
		engPutVariable(ep, "Y_train", label);
		engPutVariable(ep, "w1", w1);
		engPutVariable(ep, "w2", w2);
		engPutVariable(ep, "coef", coef);
		engPutVariable(ep, "ii", n);
		engPutVariable(ep, "selection", select);
		engPutVariable(ep, "y", y);
		engPutVariable(ep, "aco1", aco1);
		engPutVariable(ep, "aco2", aco2);
		engEvalString(ep, "predict");

		select = mxCreateDoubleMatrix(1,i, mxREAL);
		label = mxCreateDoubleMatrix(i,1, mxREAL);
		coef = mxCreateDoubleMatrix(i,1, mxREAL);
		errors = mxCreateDoubleMatrix(i-3,1, mxREAL);

		errors = engGetVariable(ep,"error");
		coef = engGetVariable(ep,"coef");
		w = engGetVariable(ep,"w");
		select = engGetVariable(ep,"selection");
		label = engGetVariable(ep,"Y_train");
		//for(int j=0; j<6; j++){ printf("select is  %f\t\n", *(mxGetPr(select)+j));}
		cout << CYAN << "->->->->error_rate " << *(mxGetPr(errors)) << "\n" << NORMAL;
		for(int j=0; j<dim; j++){ 
			cout << *(mxGetPr(w)+j) << " * x_" << j;
			if (j != dim - 1)
				cout << " + ";
		}
		cout << " >= 0." << endl;
	}

	engClose(ep);	
	return EXIT_SUCCESS;
}
