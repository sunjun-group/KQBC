#ifndef __qbclearner__
#define __qbclearner__

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <map>
#include <cmath>
#include <armadillo>
#include "color.h"
#include <nk/polynomial.h>

#include "CNode.h"
#include "Constraint.h"
#include "term.h"
#include <map>
#include <string>


const double tolerance = 1.0e-10;
const int MAX_ITERATION = 256;
const size_t qbc_learner_default_problem_size = 1 << 16;
const int MAXN = 500000;
//const int MAXN = 999999;
extern int _status;
const int upbound = 1000;

bool vec_simplify(arma::vec& v);

class QBCLearner {
	protected:
		std::vector<std::string> _names;
		size_t _data_occupied;
		arma::mat _data;
		arma::vec _labels;
		arma::vec _weight;
		std::vector<Term*> terms;

	public:
		QBCLearner(const std::vector<std::string> &names) : _names(names)
															, _data_occupied(0L), _data(0ULL, names.size()), _labels(0ULL) {}
		~QBCLearner() {}

	protected:
		bool increase_problem_size();

	private:
		void setupSolver() {
			terms.clear();
			terms.push_back(VariableTerm::make("ConstantOne"));
			for (size_t i = 1; i < _names.size(); i++) {
				//std::cout << "_names[" << i << "] = " << _names[i] << "\n";
				terms.push_back(VariableTerm::make(_names[i]));
			}
		}

		Constraint toConstraint(arma::vec w, bool b = true) {
			//std::cout << "convert >>>w: " << w.t() << "-----";
			map<Term*, long int> map0;
			//std::cout << std::flush;
			for (size_t i = 1; i < _names.size(); i++) {
				map0[terms[i]] = (long int)(w.at(i));
				//std::cout << "-- map0[" << terms[i]->to_string() << ":" << (long int)(w.at(i)) << "]\n";
			}
			Term* term0 = ArithmeticTerm::make(map0, w.at(0));
			//std::cout << "   - term0:  " << term0->to_string() << std::endl;
			if (b) {
				Constraint c(term0, ConstantTerm::make(0), ATOM_GEQ);
				//std::cout << "-> constraint:  " << c << std::endl;
				return c;
			} else {
				Constraint c(term0, ConstantTerm::make(0), ATOM_LT);
				//std::cout << "-> constraint:  " << c << std::endl;
				return c;
			}
		}

		bool solve(Constraint& c, arma::vec& v) {
			//std::cout << GREEN << "solve constraint:" << c << std::endl << NORMAL;
			if (c.sat() == false) {
				//std::cout << "not sat!\n";
				return false;
			}
			v.resize(_names.size());
			map<Term*, SatValue> solution;
			c.get_assignment(solution);
			//std::cout << "sat!\n solution:\n";
			for (auto it = solution.begin(); it != solution.end(); ++it) {
				for (size_t i = 1; i < _names.size(); i++) {
					if (it->first == terms.at(i)) {
						//std::cout << "#######" << terms.at(i)->to_string() << " --> " << it->second.value << std::endl;
						v.at(i) = it->second.value.to_double();
						break;
					}
				}
			}
			v.at(0) = 1;
			//std::cout << " =>=>=>=>=>=> " << v.t(); 
			return true;
		}

	public:
		arma::vec samplingRandomly(arma::vec w1, arma::vec w2) {
			arma::vec s;
			int n = 0;
			size_t size = w1.n_rows;
			while (++n <= MAXN) {
				arma::vec s = arma::randi<arma::mat> (size, arma::distr_param(-upbound-n/100, upbound+n/100));
				s.at(0) = 1;
				if (dot(s, w1) * dot(s, w2) < 0) {
					_status = 0;
					return s;
				}
			}

			_status = 1;
			return s;
		}

		arma::vec samplingBySolving(arma::vec w1, arma::vec w2) {
			//std::cout << "SAMPLING::\n\t" << w1.t() << "\t" << w2.t();
			//_roundoff(w1);
			//_roundoff(w2);
			//std::cout << "SAMPLING::\n\t" << w1.t() << "\t" << w2.t();
			//arma::vec s;
			arma::vec s;
			setupSolver();
			Constraint c1 = toConstraint(w1) & toConstraint(w2, false);
			Constraint c2 = toConstraint(w1) & toConstraint(w2, false);
			Constraint c = c1 | c2;
			if (solve(c, s) == false) {
				std::cout << "Can not find a solution for constraint: " << c << std::endl;
				_status = 1;
				return s;
			}
			_status = 0;
			return s;
		}

		arma::vec samplingMixed(arma::vec w1, arma::vec w2) {
			arma::vec s = samplingRandomly(w1, w2);
			if (_status == 0)
				return s;
			s = samplingBySolving(w1, w2);
			return s;
		}

		double (*categorizeF)(arma::vec x);
		bool add(const std::vector<double> &values, const double &y);
		bool addVec(const arma::vec &values, const double &y);
		//bool add(const std::map<std::string, double> &valuator, const double &y);
		void clear();

		friend std::ostream& operator << (std::ostream& out, QBCLearner& qbc) {
			out << std::setprecision(16);
			bool found_not_zero = false;
			arma::vec w = qbc._weight;
			arma::vec ratio_w = w / w[1]; 
			std::cout << CYAN << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
			std::cout << "R-Learn: " << ratio_w.t();
			std::cout << CYAN << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" << NORMAL;
			if (w[0] != 0) {
				found_not_zero = true;
				std::cout << YELLOW << w[0] << NORMAL;
			}

			for (size_t i = 1; i < w.size(); i++) {
				if (w[i] == 0)
					continue;
				if (w[i] <= 0) {
					if (found_not_zero == false) {
						std::cout << " -";
					} else {
						std::cout << " - ";
					}
					w[i] = -w[i];
				} else {
					if (found_not_zero) {
						std::cout << " + ";
					}
				}
				if (w[i] != 1) 
					std::cout << YELLOW << w[i] << BLUE << " * ";
				std::cout << RED <<  qbc._names[i] << NORMAL;
				found_not_zero = true;
			}
			if (found_not_zero == false)
				std::cout << YELLOW << "0";
			std::cout << " >= 0";
			return out;
		}

		/*
		   void _roundoff(arma::vec& v) {
		   Polynomial poly;
		   poly.setValues(v);
		   v = poly.roundoff();
		   }
		   */

	public:
		bool learn_linear(size_t T);
		arma::vec hit_and_run(arma::vec xpoint, arma::mat constraintMat, size_t T);
		void roundoff() {
			Polynomial poly;
			for (size_t i = 0; i < _weight.n_elem; i++)
				poly.set_coef(i, _weight.at(i));
			/*
			poly.set_coefs(_weight, _weight.n_elem, 
					[](arma::vec& x, int t) {
						return (double)x.at(t);
					}
					);
					*/
			poly.roundoff_in_place();
			for (size_t i = 0; i < _weight.n_elem; i++)
				_weight.at(i) = poly.get_coef(i);
		}
};


#endif /* __qbclearner__ */
