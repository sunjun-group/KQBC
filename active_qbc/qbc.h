//
//  qbclearner.h
//  project_name
//
//  Created by Li Li on 3/8/16.
//  Copyright © 2016 Lilissun. All rights reserved.
//

#ifndef __qbclearner__
#define __qbclearner__

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <cmath>

#include <armadillo>

#define RED "\e[31m"
#define GREEN "\e[32m"
#define YELLOW "\e[33m"
#define BLUE "\e[34m"
#define MAGENTA "\e[35m"
#define CYAN "\e[36m"
#define LIGHT_GRAY "\e[37m"
#define DARK_GRAY "\e[90m"
#define LIGHT_RED "\e[91m"
#define LIGHT_GREEN "\e[92m"
#define LIGHT_YELLOW "\e[93m"
#define LIGHT_BLUE "\e[94m" 
#define LIGHT_MAGENTA "\e[95m"
#define LIGHT_CYAN "\e[96m"
//case WHITE: // white
#define NORMAL "\e[0m"
#define BOLD "\e[1m"
#define UNDERLINE "\e[4m"

const size_t qbc_learner_default_problem_size = 1 << 16;

class QBCLearner {
protected:
    std::vector<std::string> _names;
    size_t _data_occupied;
    arma::mat _data;
    arma::vec _labels;
    
public:
    QBCLearner(const std::vector<std::string> &names) : _names(names)
    , _data_occupied(0L), _data(0ULL, names.size()), _labels(0ULL) {}
    ~QBCLearner() {}
    
protected:
    bool increase_problem_size();

public:
    bool add(const std::vector<double> &values, const double &y);
    //bool add(const std::map<std::string, double> &valuator, const double &y);
    void clear();
    
public:
    //LinearConstraint learn_linear(size_t T);
    void learn_linear(size_t T);
	arma::vec new_query(size_t T);
	arma::vec hit_and_run(arma::vec xpoint, arma::mat constraintMat, size_t T);
};

/*
QBCLearner l({"x", "y", "z"});
l.add({1.0, 1.0, 1.0}, 1.0);
l.add({-1.0, -1.0, -1.0}, -1.0);
LinearConstraint cons = learn_linear(10);
*/

#endif /* __qbclearner__ */
