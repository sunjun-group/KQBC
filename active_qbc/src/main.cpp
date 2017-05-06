//
//  qbclearner.cpp
//  explicit
//
//  Created by Li Li on 3/8/16.
//  Copyright © 2016 Lilissun. All rights reserved.
//

#include "qbc.h"
#include "oracle.h"
//#include <random>

int _status = 0;
int upbound = 100;
bool TEST = false;
const int MAXN = 5000000;
Oracle oracle;

arma::vec activeSampling(arma::vec w1, arma::vec w2) {
	//std::cout << YELLOW << ">>>>>>>>>>>>>>>>>" << __FILE__ << ":" << __LINE__ << "-activeSampling-------------------" << NORMAL << std::endl;
	int n = 0;
	arma::vec s;
	size_t size = w1.n_rows;
	while (++n <= MAXN) {
		arma::vec s = arma::randi<arma::mat> (size, arma::distr_param(0, upbound));
		s.at(0) = 1;
		if (dot(s, w1) * dot(s, w2) < 0) {
			return s;
		}
	}
	std::cout << RED << "Tried " << MAXN << " times, can not find solutions inside the following matrix: \n" << w1.t() << w2.t() << NORMAL;
	_status = 1;
	return s;
}


double classify(arma::vec s) {
	return oracle.classify(s);
}


int main(int argc, char** argv) {
	if (argc > 1)
		TEST = true;
	arma::arma_rng::set_seed_random();
	std::cout.precision(16);
	//std::cout.setf(std::ios::fixed);
	//std::cout << std::fixed;

	int dim = 3;
	int init_sample_num = 8;

	srand(time(NULL));
	if (!TEST) {
		std::cout << "dimension:? ";
		std::cin >> dim;
	}
	std::cout << __FILE__ << ":" << __LINE__ << std::endl;
	//Oracle oracle;
	oracle.readCoef(dim);
	//oracle.output();
	//QBCLearner l({"x", "y", "z"});
	//l.add({1.0, 1.0, 1.0}, 1.0);
	//l.add({-1.0, -1.0, -1.0}, -1.0);
	QBCLearner l({"1", "x1", "x2"});
	l.categorizeF = classify;
	l.samplingF = activeSampling;
	std::cout << __FILE__ << ":" << __LINE__ << std::endl;
	/*
	   l.add({1.0, 1.0}, 1.0);
	   l.add({1.0, 0.0}, 1.0);
	   l.add({1.0, -1.0}, -1.0);
	   l.add({-1.0, 1.0}, -1.0);
	   */

	for (int i = 0; i < init_sample_num; i++) {
		arma::vec s = arma::randi<arma::mat> (dim, arma::distr_param(0, 128));
		s.at(0) = 1;
		double y = l.categorizeF(s);
		l.addVec(s, y);
		//std::cout << __FILE__ << ":" << __LINE__ << "----|//"<< std::endl;
	}
	/*
	   l.add({1.0, 1.0}, 1.0);
	   l.add({2.0, 2.0}, 1.0);
	   l.add({3.0, -3.0}, -1.0);
	   l.add({-4.0, 4.0}, -1.0);
	   */
	l.learn_linear(100);
	std::cout << l << std::endl;
	l.roundoff();
	std::cout << l << std::endl;
	//l.add({2.0, 0.0}, -1.0);
	//QBCLearner l({"x"});
	//l.add({1.0}, 1.0);
	//l.add({2.0}, 1.0);
	//l.add({3.0}, 1.0);
	//l.add({-1.0}, -1.0);
	//tmpl.add({-2.0}, -1.0);
	//l.add({-3.0}, -1.0);
	//l.learn_linear(10);
	//LinearConstraint cons = learn_linear(10);
	return 0;
}
