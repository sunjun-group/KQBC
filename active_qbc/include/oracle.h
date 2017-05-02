#ifndef __oracle__
#define __oracle__
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <cmath>
#include <armadillo>

class Oracle {
	public:
		Oracle(){}

		void readCoef(int _dim);

		double classify(arma::vec x);

		/*
		void output() {
			for (int i = 0; i < _dim+1; i++)
				std::cout << i << " -- " << _w[i] << std::endl;
		}
		*/

	private:
		int _dim;
		arma::vec _w;
};
#endif
