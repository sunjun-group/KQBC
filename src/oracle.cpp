#include "oracle.h"

extern const bool TEST;

void Oracle::readCoef(int dim) {
	if (TEST) {
		//_dim = 3;
		_dim = 3;
		_dim = rand() % 2 + 3;
		_w.zeros(_dim);
		for (int i = 0; i < _dim; i++)
			_w[i] = rand() % 13 - 7;
		std::cout << "oracle: dim=" << _dim << " --->" << _w.t();
		//_w = _w / _w[1];
		//std::cout << "oracle: ratio" << " --->" << _w.t();
		/*
		_w[0] = -3;
		_w[1] = 0;
		_w[2] = 2;

		_w[0] = -1;
		_w[1] = 2;
		_w[2] = 0;


		_w[0] = 0;
		_w[1] = -1;
		_w[2] = 2;
		*/
		/*
		_w[0] = 500;
		_w[1] = -8;
		_w[2] = 3;
		*/
	} else {
		//for (int i = 0; i < _dim+1; i++) {
		_dim = dim;
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
