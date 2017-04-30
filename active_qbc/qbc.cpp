//
//  qbclearner.cpp
//  explicit
//
//  Created by Li Li on 3/8/16.
//  Copyright © 2016 Lilissun. All rights reserved.
//

#include "qbc.h"
//#include <random>

int itime = 0;
const double tolerance = 1.0e-10;

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

	std::cout << "++ ";
	for (size_t index = 0; index != values.size(); ++index) {
		const double &value = values.at(index);
		std::cout << " >" << value;
		_data.at(_data_occupied, index) = value;
	}
	std::cout << std::endl;

	_labels.at(_data_occupied) = y;
	++_data_occupied;

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
		if (disc < 0) 
		{
			std::cout << "negative disc " << disc <<  ". Probably x is not a ' ... 'feasable point.\n";
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

//LinearConstraint QBCLearner::learn_linear(size_t T)
void QBCLearner::learn_linear(size_t T)
{
	_data.resize(_data_occupied, _names.size());
	_labels.resize(_data_occupied);

	arma::mat coefs;
	arma::mat errors;

	arma::mat K = _data * _data.t();
	size_t length = _labels.size();

	arma::colvec coef = arma::zeros(length);
	coef.at(0) = _labels.at(0)/std::sqrt(K.at(0, 0));

	arma::uvec selection;
	selection << 0;

	for (size_t ii = 1; ii != length; ++ii) {
		arma::uvec extension = selection;
		extension << ii;
		arma::mat Ksub = K.submat(extension, extension);
		arma::mat S;
		arma::mat U;
		arma::schur(U, S, Ksub);

		arma::vec Sdiag = S.diag();
		arma::uvec Sall;
		arma::uvec Sselect;
		for (size_t i = 0; i != Sdiag.size(); ++i) {
			Sall << i;
			if (Sdiag.at(i) > tolerance) {
				Sselect << i;
			}
		}

		arma::uvec first_element;
		first_element << 0;

		arma::vec SI = arma::pow(Sdiag.submat(Sselect, first_element), -0.5);
		arma::mat A = U.submat(Sall, Sselect) * SI.diag();

		arma::mat restri = _labels.submat(Sselect, first_element) * K.submat(selection, extension) * A;
		arma::vec co1 = arma::pinv(A) * coef.submat(extension, first_element);
		arma::vec co2 = hit_and_run(co1, restri, T);
		co1 = hit_and_run(co2, restri, T);
		std::cout << "co1:" << co1;
		std::cout << "co2:" << co2;

		/*arma::vec temp = K.row(ii);
		  temp = temp.submat(first_element, extension);
		  double pred1 = dot(temp, A * co1);
		  double pred2 = dot(temp, A * co2);
		  */
		arma::uvec iivec;
		iivec << ii;
		double pred1 = dot(K.submat(iivec, extension), A * co1);
		double pred2 = dot(K.submat(iivec, extension), A * co2);

		if (pred1 * pred2 <= 0) {
			itime++;
			selection = extension;
			if (_labels.at(ii) * pred1 >= 0) {
				coef.submat(first_element, extension) = A * co1;
			} else {
				coef.submat(first_element, extension) = A * co2;
			}
		}

		// TODO ...
		coefs = arma::join_horiz(coefs, coef);
		arma::vec preds = _labels % (K * coef);
		double errate = arma::sum(preds<=0) / double(length);
		errors << errate;
		std::cout << "------------------------------------------------\n";
		std::cout << "Step: " << ii << "\nselection:\n" << selection << "\ncoef: \n" << _data.t() * coef << "\nerror: " << errate * 100 << "%\n"; 

		//errors << 
	}
	std::cout << "\ncoefs:\n" << coefs;
	std::cout << "TIMES: " << itime << std::endl;

}


arma::vec QBCLearner::new_query(size_t T) {
	_data.resize(_data_occupied, _names.size());
	_labels.resize(_data_occupied);

	arma::mat coefs;
	arma::mat errors;
	arma::uvec selection(_data_occupied);
	for (size_t i = 0; i < _data_occupied; i++)
	{
		//selection << i;
		selection[i] = i;
		//std::cout << "selection:" << i << "\n" << selection << std::endl;
	}

	arma::mat K = _data * _data.t();
	arma::colvec coef = arma::zeros(_data_occupied);
	coef.at(0) = _labels.at(0)/sqrt(K.at(0, 0));

	std::cout << "data: \n" << _data << std::endl;
	std::cout << "label: \n" << _labels << std::endl;
	std::cout << "K: \n" << K << std::endl;
	std::cout << "selection:\n" << selection << std::endl;

	//for (size_t ii = 1; ii != length; ++ii) {
	arma::mat Ksub = K.submat(selection, selection);
	arma::mat S;
	arma::mat U;
	arma::schur(U, S, Ksub);
	std::cout << "Ksub:\n" << Ksub; 
	std::cout << "U:\n" << U; 
	std::cout << "S:\n" << S; 

	arma::vec Sdiag = S.diag();
	std::cout << "Sdiag:\n" << Sdiag; 
	arma::uvec Sall(_data_occupied);
	arma::uvec Sselect(_data_occupied);
	int k = 0;
	for (size_t i = 0; i != Sdiag.size(); ++i) {
		Sall[i] = i;
		//Sall << i;
		if (Sdiag.at(i) > tolerance) {
			Sselect[k++] = i;
		}
	}
	Sselect.resize(k);
	std::cout << "Sselect:\n" << Sselect; 
	std::cout << "Sall:\n" << Sall; 

	arma::uvec first_element;
	first_element << 0;

	std::cout << "-----> \n" << Sdiag.submat(Sselect, first_element);
	//arma::vec SI = arma::pow(Sdiag.submat(Sselect, first_element), -0.5);
	arma::vec SI = arma::pow(Sdiag.submat(Sselect, first_element), -0.5);
	std::cout << "SI:\n" << SI; 

	arma::mat Sidiag = diagmat(SI);
	std::cout << "Sidiag:\n" << Sidiag;
	arma::mat usub = U.submat(Sall, Sselect);
	std::cout << "Usub:\n" << usub;


	arma::mat A = U.submat(Sall, Sselect) * diagmat(SI);
	std::cout << "A:\n" << A; 


	arma::mat Ydiag = diagmat(_labels.submat(selection, first_element));
	//std::cout << "Ydiag:\n" << Ydiag;
	arma::mat Kselection = K.submat(selection, selection);
	//std::cout << "Kselection:\n" << Kselection; 
	arma::mat restri = diagmat(_labels.submat(selection, first_element)) * K.submat(selection, selection) * A;
	//std::cout << "restri:\n" << restri;
	//arma::vec co1 = arma::pinv(A) * coef.submat(selection, first_element);
	
	arma::mat pinvA = pinv(A);
	//std::cout << "pinvA:\n" << pinvA;

	std::cout << "coef:\n" << coef; 
	arma::mat coefselection = coef.submat(selection, first_element);
	//std::cout << "coefselection:\n" << coefselection; 

	arma::vec co1 = arma::pinv(A) * coef.submat(selection, first_element);
	std::cout << "-->co1:" << co1;
	arma::vec co2 = hit_and_run(co1, restri, T);
	co1 = hit_and_run(co2, restri, T);
	std::cout << "-->co1:" << co1;
	std::cout << "-->co2:" << co2;

	arma::vec aco1 = A * co1;
	arma::vec aco2 = A * co2;
	std::cout << "-->A:" << A;
	std::cout << "-->aco1:" << aco1;
	std::cout << "-->aco2:" << aco2;

	std::cout << "selection:\n" << selection; 
	std::cout << "sall:\n" << Sall; 
	arma::mat sub1 = _data.rows(selection);
	arma::mat sub2 = _data.rows(selection);
	std::cout << "-->sub1:" << sub1;
	std::cout << "-->sub2:" << sub2;

	
	arma::vec w1 = _data.rows(selection).t() * (A * co1);
	arma::vec w2 = _data.rows(selection).t() * (A * co2);

	std::cout << "---->w1: " << w1;
	std::cout << "---->w2: " << w2;
	return w1;
}


int main()
{
	//QBCLearner l({"x", "y", "z"});
	//l.add({1.0, 1.0, 1.0}, 1.0);
	//l.add({-1.0, -1.0, -1.0}, -1.0);
	QBCLearner l({"x", "y"});
	/*
	l.add({1.0, 1.0}, 1.0);
	l.add({1.0, 0.0}, 1.0);
	l.add({1.0, -1.0}, -1.0);
	l.add({-1.0, 1.0}, -1.0);
	*/

	l.add({1.0, 1.0}, 1.0);
	l.add({2.0, 2.0}, 1.0);
	l.add({3.0, -3.0}, -1.0);
	l.add({-4.0, 4.0}, -1.0);
	l.new_query(6);
	/*
	   l.add({1.0, 1.0}, 1.0);
	   l.add({1.0, 0.0}, 1.0);
	   l.add({1.0, -1.0}, -1.0);
	   l.add({-1.0, 1.0}, -1.0);
	   l.add({1.0, 1.0}, 1.0);
	   l.add({1.0, 0.0}, 1.0);
	   l.add({1.0, -1.0}, -1.0);
	   l.add({-1.0, 1.0}, -1.0);
	   l.add({1.0, 1.0}, 1.0);
	   l.add({1.0, 0.0}, 1.0);
	   l.add({1.0, -1.0}, -1.0);
	   l.add({-1.0, 1.0}, -1.0);
	   l.add({1.0, 1.0}, 1.0);
	   l.add({1.0, 0.0}, 1.0);
	   l.add({1.0, -1.0}, -1.0);
	   l.add({-1.0, 1.0}, -1.0);
	   l.add({1.0, 1.0}, 1.0);
	   l.add({1.0, 0.0}, 1.0);
	   l.add({1.0, -1.0}, -1.0);
	   l.add({-1.0, 1.0}, -1.0);
	   l.add({1.0, 1.0}, 1.0);
	   l.add({1.0, 0.0}, 1.0);
	   l.add({1.0, -1.0}, -1.0);
	   l.add({-1.0, 1.0}, -1.0);*/
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
}
