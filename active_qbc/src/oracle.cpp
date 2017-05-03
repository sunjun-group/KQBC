#include "oracle.h"

extern const bool TEST;

void Oracle::readCoef(int _dim) {
	if (TEST) {
		_dim = 3;
		_w.zeros(_dim);
		_w[0] = -1;
		_w[1] = 2;
		_w[2] = 0;

		_w[0] = -8;
		_w[1] = 3;
		_w[2] = 500;
	} else {
		//for (int i = 0; i < _dim+1; i++) {
		_w.zeros(_dim);
		for (int i = 0; i < _dim; i++) {
			std::cout << "  coef[" << i << "]=";
			std::cin >> _w[i];
		}
	}
}

double Oracle::classify(arma::vec x) {
	double res = dot(x, _w);
	//res += _w[_dim];
	return (res>=0? 1 : -1);
}