#ifndef __polynomial__
#define __polynomial__

#include <cmath>
#include <cfloat>
#include <stdarg.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <armadillo>
#include "color.h"

#define PRECISION 1
const double UPBOUND = pow(0.1, PRECISION);
static bool _roundoff(double x, double& roundx)
{
	if (std::abs(x) <= UPBOUND) {
		roundx = 0;
		return true;
	}
	roundx = nearbyint(x);
	if ((roundx  >= x * (1 - UPBOUND) && roundx  <= x * (1 + UPBOUND))
			|| (roundx  <= x * (1 - UPBOUND) && roundx  >= x * (1 + UPBOUND))) {
		return true;
	}
	return false;
}


static bool _roundoffvector(std::vector<double>& v) {
	std::vector<double> ret = v;
	for (size_t i = 0; i < v.size(); i++) {
		if (_roundoff(v[i], ret[i]) == false)
			return false;
	}
	v = ret;
	return true;
}

static bool scale(std::vector<double>& v, double times) {
	if (times == 0) return false;
	//std::cout << poly.get_dim() << "--";
	for (size_t i = 0; i < v.size(); i++)
		v[i] *= times;
	return true;
}

static void printVector(std::vector<double>& v) {
	std::cout << "(";
	for (size_t i = 0; i < v.size(); i++) {
		std::cout << v[i];
		if (i < v.size() - 1)
			std::cout << ", ";
	}
	std::cout << ")";
}

static int gcd(int a, int b) {
	if (a==1 || b==1)
		return 1;
	if (a==0 || b==0)
		return (a+b)>0? (a+b):1;
	if (a==b)
		return a;
	if (a<b) {
		int tmp = a;
		a = b;
		b = tmp;
	}
	return gcd(b, a%b);
}

static int ngcd(std::vector<double> v) {
	for (size_t i = 0; i < v.size(); i++) {
		if ((int)v[i] - v[i] != 0)
			return 1;
	}
	int coffgcd = std::abs((int)v[1]);
	for (size_t i = 2; i < v.size(); i++) {
		if (coffgcd == 1)
			break;
		coffgcd = gcd(coffgcd, std::abs((int)v[i]));
	}
	return coffgcd;
}

class Polynomial{
	public:
		Polynomial() {}

		void setValues(arma::vec v) {
			_dim = v.n_elem;
			for(int i = 0; i < _dim; i++)
				_values.push_back(v.at(i));
		}

		void setValues(std::vector<double> v) {
			_values = v;
		}

		void setNames(std::vector<std::string> names) {
			_names = names;
		}

		
		std::vector<double> roundoff() {
			std::vector<double> ret = _values;

			double max = 0;
			for (int i = 0; i < _dim; i++) {
				if (std::abs(_values[i]) > max) {
					max = std::abs(_values[i]);
				}
			}
			//std::cout << "dim: " << _dim << std::endl;
			double min = max;
			for (int i = 0; i < _dim; i++) {
				if (std::abs(_values[i]) == 0) continue;
				/*if (std::abs(_values[i]) * pow(100, PRECISION) < max) {
					ret[i] = 0;
					continue;
				}
				*/
				if (std::abs(_values[i]) < min) {
					min = std::abs(_values[i]);
				}
			}
			scale(ret, 1.0/min);
			//std::cout << "min=" << min << std::endl;

			int scale_up = 2;
			while(scale_up <= 100) {
				if (_roundoffvector(ret) == true) {
					break;
				} 
				scale(ret, (1.0 * scale_up)/(scale_up-1));
				scale_up++;
			}
			if (scale_up > 100) {
				for (int i = 0; i < _dim - 1; i++) {
					_roundoff(ret[i] / min, ret[i]);
				}
			}

			int poly_gcd = ngcd(ret);
			if (poly_gcd > 1) {
				for (int i = 1; i < _dim; i++) {
					ret[i] = ret[i] / poly_gcd;
				}
			}
			ret[0] = floor((int)ret[0] / poly_gcd);
			//std::cout << "\tAfter roundoff: " << e << NORMAL << std::endl;
			std::cout << GREEN << "After roundoff: ";
			printVector(ret);
			std::cout << std::endl << NORMAL;
			return ret;
		}

		bool isSimilar(Polynomial& e2, int m_precision = 2) {
			return false;
			double ratio = 0;
			int i;
			double max1 = std::abs(_values[0]), max2 = std::abs(e2._values[0]);
			for (i = 0; i < _dim; i++) {
				if (max1 < std::abs(_values[i])) max1 = std::abs(_values[i]); 
				if (max2 < std::abs(e2._values[i])) max1 = std::abs(e2._values[i]); 
			}

			//-----------------------------------------------------------------------------------
			//std::cout << YELLOW << *this << RED << " VVVSSS " << YELLOW << e2 << std::endl << CYAN;
			/*
			std::cout << CYAN << "Ratio: ";
			for (i = 0; i < _dim; i++) {
				if ((_values[i] != 0) && (e2._values[i] != 0)) {
					ratio = e2._values[i] / _values[i];
					std::cout << "[" << i << "]:" << ratio << "  ";
				} else {
					std::cout << "[" << i << "]:nan  ";
				}
			}
			std::cout << std::endl << NORMAL;
			*/
			//-----------------------------------------------------------------------------------

			for (i = 0; i < _dim; i++) {
				if ((_values[i] != 0) && (e2._values[i] != 0)) {
					ratio = _values[i] / e2._values[i];
					break;
				}
			}
			if (i >= _dim)
				return -1;
			//std::cout << "1[ratio=" << ratio <<"]\n";
			double down, up;
			if (ratio >= 0) {
				down = ratio * (1 - pow(0.1, m_precision));
				up = ratio * (1 + pow(0.1, m_precision));
			}
			else {
				up = ratio * (1 - pow(0.1, m_precision));
				down = ratio * (1 + pow(0.1, m_precision));
			}
			//std::cout << "[" << down << ", " << ratio << ", " << up << "]";
			for (int i = 0; i < _dim; i++) {
				if (std::abs(_values[i]) / max1 <= std::pow(0.01, m_precision)) continue;
				if (std::abs(e2._values[i]) / max2 <= std::pow(0.01, m_precision)) continue; 
				if (e2._values[i] >= 0) {
					if ((_values[i] < e2._values[i] * down) || (_values[i] > e2._values[i] * up))
						return false;
				}
				else {
					if ((_values[i] < e2._values[i] * up) || (_values[i] > e2._values[i] * down))
						return false;
				}
			}
			return true;
		}

	private:
		int _dim;
		std::vector<double> _values;
		std::vector<std::string> _names;
};
#endif
