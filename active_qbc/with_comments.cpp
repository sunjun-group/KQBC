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
int itime = 0;
const double tolerance = 1.0e-10;
const int MAX_ITERATION = 30;
int upbound = 100;

bool QBCLearner::increase_problem_size()
{
	auto new_length = _data.n_rows == 0 ? qbc_learner_default_problem_size : _data.n_rows * 2;
	_data.resize(new_length, _names.size());
	_labels.resize(new_length);
	return true;
}

bool QBCLearner::add(const std::vector<double> &values, const double &y)
{
	if (_data_occupied == _data.n_rows) {
		if (increase_problem_size() == false) {
			return false;
		}
	}

	std::cout << "++* ";
	for (size_t index = 0; index != values.size(); ++index) {
		const double &value = values.at(index);
		//std::cout << " >" << value;
		_data.at(_data_occupied, index) = value;
	}
	//std::cout << std::endl;

	_labels.at(_data_occupied) = y;
	++_data_occupied;

	//_data.resize(_data_occupied, values.size());
	//_labels.resize(_data_occupied);
	//std::cout << "->" << _data << std::endl;

	return true;
}

bool QBCLearner::addVec(const arma::vec &x, const double &y)
{
	//std::cout << "x:"  << x;
	//std::cout << "y:"  << y;
	//std::cout << "YY: \n" << _labels;
	if (_data_occupied == _data.n_rows) {
		if (increase_problem_size() == false) {
			return false;
		}
	}

	//std::cout << "++ ";
	_data.row(_data_occupied) = x.t();
	_labels.at(_data_occupied) = y;
	//std::cout << "X" << _data;
	//std::cout << "X: " << _data.n_rows << " * " << _data.n_cols << std::endl;
	//std::cout << "Y: " << _labels.n_rows << " * " << _labels.n_cols << std::endl;

	if (y > 0) std::cout << GREEN;
	else std::cout << RED;
	std::cout << "@" << _data_occupied << "++" << x.t() << NORMAL; 
	//std::cout << "YY: \n" << _labels;

	++_data_occupied;
	_data.resize(_data_occupied, _names.size());
	_labels.resize(_data_occupied);
	//_data.resize(_data_occupied, values.size());
	//_labels.resize(_data_occupied);
	//std::cout << "->" << _data << std::endl;

	return true;
}
/*
   bool QBCLearner::add(const std::map<std::string, double> &valuator, const double &y)
   {
   std::vector<double> values;
   for (const auto &name : _names) {
   values.push_back(valuator.at(name));
   }
   return add(values, y);
   }
   */

void QBCLearner::clear()
{
	_data_occupied = 0;
}

arma::vec QBCLearner::hit_and_run(arma::vec xpoint, arma::mat A /*constraintMat*/, size_t T) 
{
	//int dim = xpoint.size();
	//std::cout << "x:\n" << xpoint;
	//std::cout << "A:\n" << A;
	//std::cout << "T:\n" << T << std::endl;

	//std::default_random_engine eg(time(0)); //seed
	//std::normal_distribution<> rnd(0, 1);
	//std::cout << RED << "\n**************************HIT && RUN************************************>>\n" << BLUE;
	arma::colvec x = arma::vectorise(xpoint);
	int dim = x.size();
	arma::mat u = arma::randn<arma::mat>(T, dim);
	//std::cout << "u:\n" << u;
	arma::mat Au = u * A.t();
	//std::cout << "Au:\n" << Au;
	arma::mat nu = sum(u % u, 1);
	//std::cout << "nu:\n" << nu;
	arma::colvec l = arma::randu<arma::colvec>(T);
	//std::cout << "l:\n" << l;

	for(size_t t = 0; t < T; ++t) 
	{
		//std::cout << YELLOW << "----" << t << BLUE << std::endl;
		arma::mat Ax = A * x;
		arma::mat ratio = -Ax / Au.row(t).t();
		//std::cout << "ratio: \n" << ratio;
		double mn = std::numeric_limits<double>::min();
		double mx = std::numeric_limits<double>::max();
		for (size_t ii = 0; ii < Au.n_cols; ++ii) {
			double value = Au(t, ii);
			double ratio_value = ratio(ii);
			if (value > 0 && ratio_value > mn) mn = ratio_value;
			if (value < 0 && ratio_value < mx) mx = ratio_value;
		}
		//std::cout << "mx=" << mx << std::endl;
		//std::cout << "mn=" << mn << std::endl;
		//arma::mat xut = x.t() * u.row(t).t();
		//double disc = std::pow(xut(0, 0), 2) - nu(t) * (std::pow(norm(x), 2) - 1);
		double disc = std::pow(dot(x.t(), u.row(t).t()), 2) - nu(t) * (std::pow(norm(x), 2) - 1);
		//std::cout << "disc=" << disc << std::endl;
		//double disc = std::pow(x.t() * u.row(t).t(), 2) - nu(t) * (std::pow(norm(x), 2)- 1);
		//double disc = 0; 
		if (disc < 0) {
			//std::cout << "negative disc " << disc <<  ". Probably x is not a ' ... 'feasable point.\n";
			//std::cout << "x";
			disc = 0;
		}
		//double hl = (-xut(0, 0) + std::sqrt(disc)) / nu(t);
		//double ll = (-xut(0, 0) - std::sqrt(disc)) / nu(t);
		//double xut = dot(x.t(), u.row(t).t());
		//std::cout << "xut=" << xut << std::endl;
		double hl = (-dot(x.t(), u.row(t).t()) + std::sqrt(disc)) / nu(t);
		double ll = (-dot(x.t(), u.row(t).t()) - std::sqrt(disc)) / nu(t);

		if (hl < mx) mx = hl;
		if (ll > mn) mn = ll;
		//std::cout << "hl=" << hl << std::endl;
		//std::cout << "ll=" << ll << std::endl;
		x = x + u.row(t).t() * (mn + l(t) * (mx - mn));
		//std::cout << "x:" << x;
	}
	//std::cout << "\n**************************HIT && RUN************************************<<\n" << NORMAL;
	return x;
}


bool QBCLearner::learn_linear(size_t T) {
	_data.resize(_data_occupied, _names.size());
	_labels.resize(_data_occupied);

	arma::mat coefs;
	arma::mat errors;
	arma::vec w1, w2;
	arma::vec co1, co2;
	arma::colvec coef;
	arma::mat K = _data * _data.t();
	arma::mat A;
	arma::vec preds;
	double errate;
	for(int iteration = 0; iteration <= MAX_ITERATION; iteration++) {
		std::cout << GREEN << BOLD << "################################################################################################## " << iteration << NORMAL << std::endl;
		arma::uvec selection(_data_occupied);
		for (size_t i = 0; i < _data_occupied; i++)
		{
			selection[i] = i;
			//std::cout << "selection:" << i << "\n" << selection << std::endl;
		}

		coef = arma::zeros(_data_occupied);
		coef.at(0) = _labels.at(0)/sqrt(K.at(0, 0));

		//std::cout << "data: \n" << _data << std::endl;
		//std::cout << "label: \n" << _labels.t() << std::endl;
		//std::cout << "K: \n" << K << std::endl;
		//std::cout << "selection:\n" << selection << std::endl;

		//for (size_t ii = 1; ii != length; ++ii) {
		arma::mat Ksub = K.submat(selection, selection);
		arma::mat S;
		arma::mat U;
		arma::schur(U, S, Ksub);
		//std::cout << "Ksub:\n" << Ksub; 
		//std::cout << "U:\n" << U; 
		//std::cout << "S:\n" << S; 

		arma::vec Sdiag = S.diag();
		//std::cout << "Sdiag:\n" << Sdiag; 
		arma::uvec Sall(_data_occupied);
		arma::uvec Sselect(_data_occupied);
		int ac = 0;
		for (size_t i = 0; i != Sdiag.size(); ++i) {
			Sall[i] = i;
			if (Sdiag.at(i) > tolerance) {
				Sselect[ac++] = i;
			}
		}
		Sselect.resize(ac);
		//std::cout << "Sselect:\n" << Sselect; 
		//std::cout << "Sall:\n" << Sall; 

		arma::uvec first_element;
		first_element << 0;

		//std::cout << "-----> \n" << Sdiag.submat(Sselect, first_element);
		//arma::vec SI = arma::pow(Sdiag.submat(Sselect, first_element), -0.5);
		arma::vec SI = arma::pow(Sdiag.submat(Sselect, first_element), -0.5);
		//std::cout << "SI:\n" << SI; 

		//arma::mat Sidiag = diagmat(SI);
		//std::cout << "Sidiag:\n" << Sidiag;
		//arma::mat usub = U.submat(Sall, Sselect);
		//arma::mat usub = U.cols(Sselect);
		//std::cout << "Usub:\n" << usub;


		//arma::mat A = U.submat(Sall, Sselect) * diagmat(SI);
		A = U.cols(Sselect) * diagmat(SI);
		//std::cout << "A:\n" << A; 


		//arma::mat Ydiag = diagmat(_labels.submat(selection, first_element));
		//std::cout << "Ydiag:\n" << Ydiag;
		//arma::mat Kselection = K.submat(selection, selection);
		//std::cout << "Kselection:\n" << Kselection; 
		arma::mat restri = diagmat(_labels.submat(selection, first_element)) * K.submat(selection, selection) * A;
		//std::cout << "restri:\n" << restri;
		//arma::vec co1 = arma::pinv(A) * coef.submat(selection, first_element);

		arma::mat pinvA = pinv(A);
		//std::cout << "pinvA:\n" << pinvA;

		//std::cout << "coef:\n" << coef; 
		arma::mat coefselection = coef.submat(selection, first_element);
		//std::cout << "coefselection:\n" << coefselection; 

		co1 = arma::pinv(A) * coef.submat(selection, first_element);
		//std::cout << "-->co1:" << co1;
		co2 = hit_and_run(co1, restri, T);
		co1 = hit_and_run(co2, restri, T);
		std::cout << "---->co1:" << co1.t();
		std::cout << "---->co2:" << co2.t();

		arma::vec aco1 = A * co1;
		arma::vec aco2 = A * co2;
		//std::cout << "-->A:" << A;
		//std::cout << "-->aco1:" << aco1;
		//std::cout << "-->aco2:" << aco2;

		//std::cout << "selection:\n" << selection; 
		//std::cout << "sall:\n" << Sall; 
		arma::mat sub1 = _data.rows(selection);
		arma::mat sub2 = _data.rows(selection);
		//std::cout << "-->sub1:" << sub1;
		//std::cout << "-->sub2:" << sub2;


		//arma::vec w1 = _data.rows(selection).t() * (A * co1);
		//arma::vec w2 = _data.rows(selection).t() * (A * co2);
		w1 = _data.rows(selection).t() * (A * co1);
		w2 = _data.rows(selection).t() * (A * co2);

		std::cout << "---->w1: " << w1.t();
		std::cout << "---->w2: " << w2.t();

		if (w1.size() == w2.size()) {
			size_t j;
			for (j = 0; j < w1.size(); j++) {
				if (w1(j) != w2(j))
					break;
			}
			if (j >= w1.size()) {
				std::cout << GREEN << "Converged. Found out the weight vector: \n" << w1;
				//_weight = w1;
				break;
				//return true;
			}
		}

		//std::cout << BLUE << "-----------------" << __FILE__ << ":" << __LINE__ << "--------------------" << NORMAL << std::endl;
		arma::vec xx = samplingF(w1, w2);
		//std::cout << BLUE << "-----------------" << __FILE__ << ":" << __LINE__ << "--------------------" << NORMAL << std::endl;
		if (_status != 0) break;
		double yy = categorizeF(xx); 
		addVec(xx, yy);
		K = _data * _data.t();
		//std::cout << "K:\n" << K; 
		//}

		double pred1 = dot(_data.row(_data_occupied-1), w1);
		double pred2 = dot(_data.row(_data_occupied-1), w2);
		std::cout << "pred1: " << pred1 << std::endl;
		std::cout << "pred2: " << pred2 << std::endl;

		//std::cout << "data:\n" << _data; 
		//std::cout << "lables:\n" << _labels.t(); 
		//std::cout << "_data_occupied:" << _data_occupied << std::endl; 
		if (_labels.at(_data_occupied-1) * pred1 >= 0) {
			coef = A * co1;
		} else {
			coef = A * co2;
		}
		std::cout << BLUE << "-----------------" << __FILE__ << ":" << __LINE__ << "--------------------" << NORMAL << std::endl;
		std::cout << "K:\n" << K; 

		//std::cout << "coef: \n" << coef.t();
		// TODO ...
		coef.at(_data_occupied-1) = 0;
		std::cout << BLUE << "-----------------" << __FILE__ << ":" << __LINE__ << "--------------------" << NORMAL << std::endl;
		std::cout << "K:\n" << K; 
		coef.resize(_data_occupied);
		std::cout << "coef: \n" << coef.t();
		std::cout << "data:\n" << _data.t(); 
		std::cout << "labels:\n" << _labels.t(); 
		preds = _labels % (K * coef);
		std::cout << "preds: " << preds.t() << std::endl;
		errate = arma::sum(preds<=0) / _data_occupied;
		errors << errate;
		_weight = _data.t() * coef;
		std::cout << "weight:\n" << YELLOW << BOLD << _weight.t() << NORMAL;
		std::cout << "Step: " << _data_occupied << std::endl;
		//std::cout << "\nselection:\n" << selection;
		std::cout << "error: " << errate * 100 << "%\n"; 
	}
	//}
	return false;
}


arma::vec activeSampling(arma::vec w1, arma::vec w2) {
	//std::cout << YELLOW << ">>>>>>>>>>>>>>>>>" << __FILE__ << ":" << __LINE__ << "-activeSampling-------------------" << NORMAL << std::endl;
	int n = 0;
	arma::vec s;
	size_t size = w1.n_rows;
	//std::cout << size << std::endl;
	while (++n <= 50000) {
		//arma::vec s = arma::randi<arma::mat> (size, arma::distr_param(0, n * upbound));
		arma::vec s = arma::randi<arma::mat> (size, arma::distr_param(0, upbound));
		//std::cout << n << "--" << s.t();
		if (dot(s, w1) * dot(s, w2) < 0) {
			//std::cout << YELLOW << "<<<<<<<<<<<<<<<<<" << __FILE__ << ":" << __LINE__ << "-activeSampling-------------------" << NORMAL << std::endl;
			return s;
		}
	}
	std::cout << RED << "Tried 50000 times, can not find solutions inside the following matrix: " << w1.t() << " -> " << w2.t() << NORMAL;
	_status = 1;
	return s;
}


Oracle oracle;

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

	int dim = 2;
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
	QBCLearner l({"x1", "x2"});
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
		arma::vec s = arma::randi<arma::mat> (dim, arma::distr_param(0, 16));
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
