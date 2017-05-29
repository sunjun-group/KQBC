//
//  qbclearner.cpp
//  explicit
//
//  Created by Li Li on 3/8/16.
//  Copyright © 2016 Lilissun. All rights reserved.
//

#include "qbc.h"

bool vec_simplify(arma::vec& v) {
	int expn = 99999999; 
	for (size_t ir = 0; ir < v.n_rows; ir++) {
		if (v[ir] == 0) break;
		int en = std::log(std::abs(v[ir])) / std::log(10);
		//std::cout << ir << "-->" << en << std::endl;
		if (en < expn) 
			expn = en;
	}
	v = v / std::pow(10, expn);
	return true;
}


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

	for (size_t index = 0; index != values.size(); ++index) {
		const double &value = values.at(index);
		_data.at(_data_occupied, index) = value;
	}

	_labels.at(_data_occupied) = y;
	++_data_occupied;

	return true;
}

bool QBCLearner::addVec(const arma::vec &x, const double &y)
{
	//std::cout << "X" << _data;
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
	//std::cout << "X: " << _data.n_rows << " * " << _data.n_cols << std::endl;
	//std::cout << "Y: " << _labels.n_rows << " * " << _labels.n_cols << std::endl;

	if (y > 0) std::cout << GREEN;
	else std::cout << RED;
	//std::cout << "@" << _data_occupied << "++" << x.t() << NORMAL; 
	std::cout << "@" << _data_occupied << "++  " << BOLD;
	for (size_t i = 1; i < x.n_elem; i++)
		std::cout << (int)x.at(i) << "  "; 
	std::cout << NORMAL << std::endl;
	//std::cout << "YY: \n" << _labels;

	++_data_occupied;
	_data.resize(_data_occupied, _names.size());
	_labels.resize(_data_occupied);
	//_data.resize(_data_occupied, values.size());
	//_labels.resize(_data_occupied);
	//std::cout << "->" << _data << std::endl;

	return true;
}

void QBCLearner::clear()
{
	_data_occupied = 0;
}

arma::vec QBCLearner::hit_and_run(arma::vec xpoint, arma::mat A /*constraintMat*/, size_t T) 
{
	//int dim = xpoint.size();
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
		double disc = std::pow(dot(x.t(), u.row(t).t()), 2) - nu(t) * (std::pow(norm(x), 2) - 1);
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

#ifdef _MISTRAL_
	int times = 100;
#endif
#ifdef _LOG_
	ofstream of("./log");
	of << std::setprecision(4);
#endif

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
	coef = arma::zeros(_data_occupied);
	coef.at(0) = _labels.at(0)/sqrt(K.at(0, 0));
	arma::vec pre_weight;

	Polynomial poly, pre_poly;

	int iteration;
	for(iteration = 0; iteration <= MAX_ITERATION; iteration++) {
		//std::cout << GREEN << BOLD << "################################################################################################## " << iteration << NORMAL << std::endl;
		std::cout << BOLD << BLUE << "----------------------------------------------------------------------------------------------------------------------> " << iteration << NORMAL << std::endl;
		arma::uvec selection(_data_occupied);
		for (size_t i = 0; i < _data_occupied; i++)
		{
			selection[i] = i;
			//std::cout << "selection:" << i << "\n" << selection << std::endl;
		}

		/*
		   coef = arma::zeros(_data_occupied);
		   coef.at(0) = _labels.at(0)/sqrt(K.at(0, 0));
		   */

		arma::mat Ksub = K.submat(selection, selection);
		arma::mat S;
		arma::mat U;
		arma::schur(U, S, Ksub);

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

		arma::uvec first_element;
		first_element << 0;

		arma::vec SI = arma::pow(Sdiag.submat(Sselect, first_element), -0.5);
		A = U.cols(Sselect) * diagmat(SI);
		arma::mat restri = diagmat(_labels.submat(selection, first_element)) * K.submat(selection, selection) * A;

		arma::mat pinvA = pinv(A);

		//arma::mat coefselection = coef.submat(selection, first_element);

		int n_hit_run = 1;
hit_run_again:
		co1 = arma::pinv(A) * coef.submat(selection, first_element);
		co2 = hit_and_run(co1, restri, T);
		co1 = hit_and_run(co2, restri, T);
		//std::cout << "---->co1:" << co1.t();
		//std::cout << "---->co2:" << co2.t();

		arma::vec aco1 = A * co1;
		arma::vec aco2 = A * co2;


		w1 = _data.rows(selection).t() * (A * co1);
		w2 = _data.rows(selection).t() * (A * co2);

		//std::cout << "---->w1: " << w1.t();
		//std::cout << "---->w2: " << w2.t();

		if (w1.size() == w2.size()) {
			size_t j;
			for (j = 0; j < w1.size(); j++) {
				if (w1(j) != w2(j))
					break;
			}
			if (j >= w1.size()) {
				std::cout << GREEN << "Converged. Found out the weight vector: \n" << w1;
				_weight = w1;
				break;
			}
		}

		vec_simplify(w1);
		vec_simplify(w2);
		//cout << "w1:" << w1.t();
		//cout << "w2:" << w2.t();

		/*
		dbg_print();
		z3::context cont;
		*/
		cout << "last_learn: " << pre_weight.t();
		Polynomial p1(w1), p2(w2);
		//cout << " *p1: " << p1.to_string() << endl;
		//cout << " *p2: " << p2.to_string() << endl;
		/*
		if (p1.is_similar(p2)) {
			cout << BOLD << GREEN << "w1 ~= w2. for " << n_hit_run << " times"<< endl << NORMAL;
			n_hit_run ++;
			if (n_hit_run >= 2) {
			//if (n_hit_run >= 3) {
				cout << BOLD << GREEN << "CONVERGED>>>>" << endl << NORMAL;
				//_weight = w1;
				_weight = pre_weight;
				vec_simplify(_weight);
				break;
			} else {
				goto hit_run_again;
			}
		}
		*/
		//*
		arma::vec xx = samplingMixed(w1, w2);
#if 0
#ifdef _RAND_
		arma::vec xx = samplingRandomly(w1, w2);
#endif
#ifdef _Z3_
		arma::vec xx = samplingByZ3(w1, w2);
#endif
#ifdef _MISTRAL_
		arma::vec xx = samplingRandomly(w1, w2);
		if (_status != 0) {
			//*
			xx = samplingByMistral(w1*times, w2*times);
			//arma::vec xx = samplingS(w1*times, w2*times);
			while ((times <= 10000) && (_status!=0)) {
				times *= 10;
				xx = samplingByMistral(w1*times, w2*times);
			}
			if (_status == 0) {
				times = 100;
				std::cout << "xx->" << xx.t();
			} 
		}
#endif
#endif
		if (_status != 0) {
			break;
		}


		//	*/
		double yy = categorizeF(xx); 
		addVec(xx, yy);
#ifdef _LOG_
		of << iteration + 1 << ": " << xx.t();
#endif
#ifdef _DBG_
		double w1xx = dot(w1, xx);
		double w2xx = dot(w2, xx);
#if 0
		cout << "  w1(xx) = " << w1xx << endl;
		cout << "  w2(xx) = " << w2xx << endl;
#endif
		if (w1xx * w2xx > 0) {
			cerr << "Fatal error! w1(xx) * w2(xx) > 0.\n";
			cout << "w1: " << w1.t() << endl;
			cout << "w2: " << w2.t() << endl;
			cout << "xx: " << xx.t() << endl;
			cout << "  w1(xx) = " << w1xx << endl;
			cout << "  w2(xx) = " << w2xx << endl;
			exit(1);
		}
#endif

		K = _data * _data.t();

		//}

		double pred1 = dot(_data.row(_data_occupied-1), w1);
		//double pred2 = dot(_data.row(_data_occupied-1), w2);
		//std::cout << "pred1: " << pred1 << std::endl;
		//std::cout << "pred2: " << pred2 << std::endl;

		//std::cout << "data:\n" << _data; 
		//std::cout << "lables:\n" << _labels.t(); 
		//std::cout << "_data_occupied:" << _data_occupied << std::endl; 
		dbg_print();
		if (_labels.at(_data_occupied-1) * pred1 >= 0) {
			coef = A * co1;
		} else {
			coef = A * co2;
		}
		//std::cout << BLUE << "-----------------" << __FILE__ << ":" << __LINE__ << "--------------------" << NORMAL << std::endl;

		coef.resize(_data_occupied);
		coef.at(_data_occupied-1) = 0;
		vec_simplify(coef);
		//std::cout << BLUE << "-----------------" << __FILE__ << ":" << __LINE__ << "--------------------" << NORMAL << std::endl;
		//std::cout << "K:\n" << K; 
		//std::cout << "coef: \n" << coef.t();
		//std::cout << "data:\n" << _data.t(); 
		//std::cout << "labels:\n" << _labels.t(); 
		preds = _labels % (K * coef);
		//std::cout << "preds: " << preds.t() << std::endl;
		errate = arma::sum(preds<=0) / _data_occupied;
		errors << errate;

		_weight = _data.t() * coef;
		vec_simplify(_weight);

		if (iteration >= 1) {
			pre_poly = poly;
			/*arma::vec ratio = _weight / pre_weight;
			std::cout << CYAN << "Ratio: " << ratio.t() << NORMAL;
			*/
		}

		dbg_print();
		//std::cout << "weight:" << YELLOW << _weight.t() << NORMAL;
		std::cout << *this;
		dbg_print();
		//poly.setValues(_weight);
		for (size_t k = 0; k < _weight.n_elem; k++)
			poly.set_coef(k, _weight.at(k));
		//dbg_print();
		if (iteration >= 1) {
			if (poly.is_similar(pre_poly)) {
				std::cout << YELLOW << "converged.\n" << NORMAL;
				break;
			}
		}
		dbg_print();
		//std::cout << "Step: " << _data_occupied << std::endl;
		//std::cout << "\nselection:\n" << selection;
		if (errate > 0)
			std::cout << "accuracy: " << (1-errate) * 100 << "%\n"; 
		pre_weight = _weight;
	}

#ifdef _LOG_
		of.close();
#endif
	//dbg_print();
	if (iteration >= MAX_ITERATION)
		return false;
	//dbg_print();
	return true;
}
