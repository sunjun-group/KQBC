#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"

int main(){
	Engine *ep1;
	Engine *ep2;
	mxArray *select=NULL, *data=NULL, *label=NULL,  *coef=NULL, *errors=NULL, *n=NULL, *new_x=NULL, *w1=NULL, *w2=NULL, *y=NULL, *w=NULL, *aco1=NULL,*aco2=NULL, *cease=NULL;
	double cselect[3]{1,2,3}; 
	double cdata[12]{1,2,4,5,1,3};
	double clabel[3]{-1,1,1}; 
	double ccoef[3]{0.14}; 
//	double cerrors[]{-1}; 
	double cn[1]{4};
	double cx[4]{0};
	double cy[1]{1};

	if (!(ep1 = engOpen(""))) {
		fprintf(stderr, "\nCan't start MATLAB engine1\n");
		return EXIT_FAILURE;
	}
	if (!(ep2 = engOpen(""))) {
		fprintf(stderr, "\nCan't start MATLAB engine2\n");
		return EXIT_FAILURE;
	}
	char buffer1[8*1024];
	char buffer2[8*1024];
	engSetVisible(ep1, 0);
	engSetVisible(ep2, 0);
	engOutputBuffer(ep1, buffer1, 8*1024);
	engOutputBuffer(ep2, buffer2, 8*1024);

	select = mxCreateDoubleMatrix(1,3, mxREAL);
	data = mxCreateDoubleMatrix(3,2, mxREAL);
	label = mxCreateDoubleMatrix(3,1, mxREAL);
	coef = mxCreateDoubleMatrix(3,1, mxREAL);
	n = mxCreateDoubleMatrix(1,1, mxREAL);
	w = mxCreateDoubleMatrix(2,1, mxREAL);
	w1= mxCreateDoubleMatrix(2,1, mxREAL);
	w2= mxCreateDoubleMatrix(2,1, mxREAL);
	y = mxCreateDoubleMatrix(1,1, mxREAL);
	cease = mxCreateDoubleMatrix(1,1, mxREAL);

	memcpy((void *)mxGetPr(select), (void *)cselect, sizeof(cselect));
	memcpy((void *)mxGetPr(data), (void *)cdata, sizeof(cdata));
	memcpy((void *)mxGetPr(label), (void *)clabel, sizeof(clabel));
	memcpy((void *)mxGetPr(coef), (void *)ccoef, sizeof(ccoef));
	memcpy((void *)mxGetPr(n), (void *)cn, sizeof(cn));

	for(int i=4; i<50; i++){
		cn[0]=i;
		memcpy((void *)mxGetPr(n), (void *)cn, sizeof(cn));
		engPutVariable(ep1, "X_train", data);
		engPutVariable(ep1, "Y_train", label);
		engPutVariable(ep1, "coef", coef);
		engPutVariable(ep1, "selection", select);
		engEvalString(ep1, "newx");

		aco1 = mxCreateDoubleMatrix(i-1,1, mxREAL);
		aco2 = mxCreateDoubleMatrix(i-1,1, mxREAL);
		data = mxCreateDoubleMatrix(i,2, mxREAL);

		new_x = engGetVariable(ep1,"new_x");
		data = engGetVariable(ep1,"X_train");
		w1 = engGetVariable(ep1,"w1");
		w2 = engGetVariable(ep1,"w2");
		aco1 = engGetVariable(ep1,"aco1");
		aco2 = engGetVariable(ep1,"aco2");
		cease= engGetVariable(ep1,"n");

/**********************************************************************/
printf("\n--------------------------------Times: %d\n", i);
		if(*mxGetPr(cease) == 50000){printf("No new_x generated\n");break;}		
		for(int j=0; j<2; j++){ 
			cx[j] = *(mxGetPr(new_x)+j);
			printf("generated x is  %f\t\n", cx[j]);
		}
		cy[0] = (cx[0]*2-cx[1]>0) ? 1:-1;
//		cy[0]= somefunction(new_x); //cx[]=new_x; generated label of new_x;

/**********************************************************************/
		memcpy((void *)mxGetPr(y), (void *)cy, sizeof(cy));
		engPutVariable(ep2, "X_train", data);
		engPutVariable(ep2, "Y_train", label);
		engPutVariable(ep2, "w1", w1);
		engPutVariable(ep2, "w2", w2);
		engPutVariable(ep2, "coef", coef);
		engPutVariable(ep2, "ii", n);
		engPutVariable(ep2, "selection", select);
		engPutVariable(ep2, "y", y);
		engPutVariable(ep2, "aco1", aco1);
		engPutVariable(ep2, "aco2", aco2);
		engEvalString(ep2, "predict");

		select = mxCreateDoubleMatrix(1,i, mxREAL);
		label = mxCreateDoubleMatrix(i,1, mxREAL);
		coef = mxCreateDoubleMatrix(i,1, mxREAL);
		errors = mxCreateDoubleMatrix(i-3,1, mxREAL);

		errors = engGetVariable(ep2,"error");
		coef = engGetVariable(ep2,"coef");
		w = engGetVariable(ep2,"w");
		select = engGetVariable(ep2,"selection");
		label = engGetVariable(ep2,"Y_train");

		std::cout << buffer1 << std::endl;
		std::cout << buffer2 << std::endl;
//for(int j=0; j<6; j++){ printf("select is  %f\t\n", *(mxGetPr(select)+j));}
printf("error is  %f\t\n", *(mxGetPr(errors)));
for(int j=0; j<2; j++){ printf("w is  %f\t\n", *(mxGetPr(w)+j));}
	}

	engClose(ep1);	
	engClose(ep2);
	return EXIT_SUCCESS;
}
