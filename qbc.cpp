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



int main(){
	Engine *ep;
	mxArray *mselect=NULL, *mdata=NULL, *mlabel=NULL;
	mxArray *mcoef=NULL, *merrors=NULL, *n=NULL, *new_x=NULL, *w1=NULL, *w2=NULL, *y=NULL, *w=NULL, *aco1=NULL,*aco2=NULL, *cease=NULL;


	int dim = 2;
	int n_init = 3;

	srand(time(NULL));
	cout << "dim:? ";
	cin >> dim;
	Oracle oracle(dim);
	oracle.readCoef();
	//oracle.output();

	double *cselect = new double [n_init];
	for (int i = 0; i < n_init; i++)
		cselect[i] = i+1;

	double *cdata = new double[dim * n_init];
	for (int i = 0; i < n_init; i++) {
		for (int j = 0; j < dim; j++) {
			cdata[j * n_init + i] = rand() % 100;
		}
	}

	double *clabel = new double[n_init];
	for (int i = 0; i < n_init; i++) {
		double temp[20];
		for (int j = 0; j < dim; j++) {
			temp[j] = cdata[j * n_init + i];
		}

		clabel[i] = oracle.classify(temp);
		cout << "init-sample" << i << "---- <";
		for (int k = 0; k < dim; k++) {
			cout << cdata[k * n_init + i];
			if (k < dim-1)
				cout << ", ";
		}
		cout << "> L" << clabel[i] << endl;
	}

	double *ccoef = new double[n_init];
	ccoef[0] = 0;
	for (int i = 0; i < dim; i++)
		ccoef[0] += cdata[i * n_init] * cdata[i * n_init];
	ccoef[0] = sqrt(ccoef[0]);
	ccoef[0] = 1.0 / ccoef[0];
	/*
	   double cselect[3]{1,2,3}; 
	   double cdata[6]{1,2,4,5,1,3};
	   double clabel[3]{-1,1,1}; 
	   double ccoef[3]{0.14}; 
	   */
	//	double cerrors[]{-1}; 
	double cn[1];
	cn[0] = n_init+1;
	double *cx = new double[dim + 1];
	cx[0] = 0;
	double cy[1];
	cy[0] = 1;

	if (!(ep = engOpen(NULL))) {
		cerr << "\nCan't start MATLAB engine1\n";
		return EXIT_FAILURE;
	}
	engSetVisible(ep, 0);

	mselect = mxCreateDoubleMatrix(1, n_init, mxREAL);
	mdata = mxCreateDoubleMatrix(n_init, dim, mxREAL);
	mlabel = mxCreateDoubleMatrix(n_init, 1, mxREAL);
	mcoef = mxCreateDoubleMatrix(n_init, 1, mxREAL);
	n = mxCreateDoubleMatrix(1, 1, mxREAL);
	w = mxCreateDoubleMatrix(2, 1, mxREAL);
	w1= mxCreateDoubleMatrix(2, 1, mxREAL);
	w2= mxCreateDoubleMatrix(2, 1, mxREAL);
	y = mxCreateDoubleMatrix(1, 1, mxREAL);
	cease = mxCreateDoubleMatrix(1, 1, mxREAL);

	memcpy((void *)mxGetPr(mselect), (void *)cselect, sizeof(cselect) * n_init);
	memcpy((void *)mxGetPr(mdata), (void *)cdata, sizeof(cdata) * n_init * dim);
	memcpy((void *)mxGetPr(mlabel), (void *)clabel, sizeof(clabel) * n_init);
	memcpy((void *)mxGetPr(mcoef), (void *)ccoef, sizeof(ccoef) * n_init);
	memcpy((void *)mxGetPr(n), (void *)cn, sizeof(cn));

	/*
	for (int i = 0; i < n_init; i++) 
		cout << "mselect" << i << " n " << cselect[i] << "-> " << *(double*)(mxGetPr(mselect) + i) << endl;
	*/

	int nsamples = 0;
	for(int i = n_init + 1; i<50; i++){
		cn[0]=i;
		memcpy((void *)mxGetPr(n), (void *)cn, sizeof(cn));
		engPutVariable(ep, "X_train", mdata);
		engPutVariable(ep, "Y_train", mlabel);
		engPutVariable(ep, "coef", mcoef);
		engPutVariable(ep, "selection", mselect);
		engPutVariable(ep, "dim", dim);
		engEvalString(ep, "newx");

		aco1 = mxCreateDoubleMatrix(i-1,1, mxREAL);
		aco2 = mxCreateDoubleMatrix(i-1,1, mxREAL);
		mdata = mxCreateDoubleMatrix(i,dim, mxREAL);

		new_x = engGetVariable(ep,"new_x");
		mdata = engGetVariable(ep,"X_train");
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
		memcpy((void *)mxGetPr(y), (void *)cy, sizeof(cy));
		engPutVariable(ep, "X_train", mdata);
		engPutVariable(ep, "Y_train", mlabel);
		engPutVariable(ep, "w1", w1);
		engPutVariable(ep, "w2", w2);
		engPutVariable(ep, "coef", mcoef);
		engPutVariable(ep, "ii", n);
		engPutVariable(ep, "selection", mselect);
		engPutVariable(ep, "y", y);
		engPutVariable(ep, "aco1", aco1);
		engPutVariable(ep, "aco2", aco2);
		engEvalString(ep, "predict");

		mselect = mxCreateDoubleMatrix(1,i, mxREAL);
		mlabel = mxCreateDoubleMatrix(i,1, mxREAL);
		mcoef = mxCreateDoubleMatrix(i,1, mxREAL);
		merrors = mxCreateDoubleMatrix(i-3,1, mxREAL);

		merrors = engGetVariable(ep,"error");
		mcoef = engGetVariable(ep,"coef");
		w = engGetVariable(ep,"w");
		mselect = engGetVariable(ep,"selection");
		mlabel = engGetVariable(ep,"Y_train");
		//for(int j=0; j<6; j++){ printf("mselect is  %f\t\n", *(mxGetPr(mselect)+j));}
		cout << CYAN << "->->->->error_rate " << *(mxGetPr(merrors)) << "\n" << NORMAL;
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
