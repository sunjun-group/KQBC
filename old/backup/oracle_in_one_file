#ifndef __oracle__
#define __oracle__
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <cmath>
#include <armadillo>

bool TEST = false;

class Oracle {
	public:
		Oracle(){}

		void readCoef(int _dim) {
			if (TEST) {
				_dim = 3;
				_w.zeros(_dim);
				_w[0] = -1;
				_w[1] = 2;
				_w[2] = 0;
			} else {
				//for (int i = 0; i < _dim+1; i++) {
				_w.zeros(_dim);
				for (int i = 0; i < _dim; i++) {
					std::cout << "  coef[" << i << "]=";
					std::cin >> _w[i];
				}
			}
			//_w.resize(_dim);
		}

		double classify(arma::vec x) {
			double res = dot(x, _w);
			//res += _w[_dim];
			return (res>=0? 1 : -1);
		}

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
