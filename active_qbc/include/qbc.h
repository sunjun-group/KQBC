#ifndef __qbclearner__
#define __qbclearner__

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <cmath>
#include <armadillo>
#include "color.h"
#include "polynomial.h"


const double tolerance = 1.0e-10;
const int MAX_ITERATION = 50;
const size_t qbc_learner_default_problem_size = 1 << 16;

class QBCLearner {
protected:
    std::vector<std::string> _names;
    size_t _data_occupied;
    arma::mat _data;
    arma::vec _labels;
	arma::vec _weight;
    
public:
    QBCLearner(const std::vector<std::string> &names) : _names(names)
    , _data_occupied(0L), _data(0ULL, names.size()), _labels(0ULL) {}
    ~QBCLearner() {}
    
protected:
    bool increase_problem_size();

public:
	arma::vec (*samplingF)(arma::vec w1, arma::vec w2);
	double (*categorizeF)(arma::vec x);
    bool add(const std::vector<double> &values, const double &y);
    bool addVec(const arma::vec &values, const double &y);
    //bool add(const std::map<std::string, double> &valuator, const double &y);
    void clear();
    
friend std::ostream& operator << (std::ostream& out, QBCLearner& qbc) {
	/*
		if (qbc._weight.size() <= qbc._names.size()) {
			std::cout << "The learner has not started yet.\n";
			return out;
		}
		*/
		bool found_not_zero = false;
		arma::vec w = qbc._weight;
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

public:
    bool learn_linear(size_t T);
	arma::vec hit_and_run(arma::vec xpoint, arma::mat constraintMat, size_t T);
	void roundoff() {
		Polynomial poly;
		poly.setValues(_weight);
		_weight = poly.roundoff();
	}
};

#endif /* __qbclearner__ */
