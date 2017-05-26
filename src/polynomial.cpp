/** @file polynomial.h
 *  @brief Defines the linear polynomial format and its solution format.
 *
 *  @author Li Jiaying
 *  @bug No known bugs. 
 */

#include <cmath>
#include <cfloat>
#include <stdarg.h>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <gsl/gsl_poly.h>
#include "polynomial.h"
#include "common.h"

const double UPBOUND = pow(0.1, PRECISION);


Polynomial& Polynomial::operator=(Polynomial& rhs) {
	if (this == &rhs) { return *this; }
	nvars = rhs.nvars;
	set_power(rhs.get_power());
	values.clear();
	for (int i = 0; i < rhs.get_dimension(); i++)
		values.push_back(rhs.get_coef(i));
	return *this;
}


bool Polynomial::operator==(const Polynomial& rhs) const {
	if ((power != rhs.get_power()) || (nvars != rhs.get_nvars()))
		return false;
	for (int i = 0; i < dim; i++)
		if (values[i] != rhs.get_coef(i))
			return false;
	return true;
}

double Polynomial::get_coef(int index) const {
	assert((index>=0) && (index<dim));
	return values[index];
}

template<typename T>
bool Polynomial::set_coef(int index, T c) {
#ifdef _PRT_POLY_
	cout << "+ " << (double)c << " @coef[" << index << "]\n"; 
#endif
	/*
	   if ((index<0) || (index>=dim)) {
	   cerr << "Polynomial::set_coef tries to set coef at index " << index 
	   << ", which exceed the boundary " << dim << ".\n";
	   return false;
	   }
	   */

	if (index>=dim) {
		values.resize(index+1);
		dim = index;
	}
	values[index] = (double)c;
	return true;
}

Polynomial::Polynomial(vec v) {
	nvars = v.n_elem;
	set_power(1);
	set_anonymous_list(v.n_elem);
	set_coefs(v);
	optype = ATOM_GEQ;
}

bool Polynomial::set_coefs(vec v) {
	for (size_t i = 0; i < v.n_elem; i++)
		values[i] = v.at(i);
	return true;
}


vector<double> Polynomial::get_coefs() const {
	return values;
}

template<typename T>
bool Polynomial::set_coefs(vector<T> cs) {
	values = cs;
	return true;
}

template<typename T>
bool Polynomial::set_coefs(T cs, int size, double (* visit_function)(T& t, int i)) {
	values.resize(size);
	for (int i = 0; i < size; i++) {
		values[i] = visit_function(cs, i);
	}
	return true;
}

bool Polynomial::set_variable_name(int index, const string& name) {
	if ((index<0) || (index>=dim)) {
		cerr << "Polynomial::set_coef tries to set coef at index " << index
			<< ", which exceed the boundary " << dim << ".\n";
		return false;
	}
	vars[index] = name;
	return true;
}

string Polynomial::get_variable_name(int index) const {
	assert((index>=0) && (index<dim));
	if (index == 0)
		return "1";
	return vars[index - 1];
}

bool Polynomial::set_variable_name_list(const vector<string>& names) {
	vars.clear();
	vars = names;
	nvars = names.size();
	set_power(1);
	return true;
}

bool Polynomial::set_anonymous_list(int n) {
	vars.clear();
	for (int i = 0; i < n; i++) {
		vars.push_back("anonymous_[" + std::to_string(i) + "]");
	}
	nvars = n;
	set_power(1);
	return true;
}

vector<string> Polynomial::get_variable_name_list() const {
	return vars;
}

int Polynomial::compute_dimension() {
	dim = 0;
	//dbg_print();
	for (int i = 0; i <= power; i++) {
#ifdef _PRT_POLY_
		cout << "combination(" << nvars << ", " << i << ") = " << combination(nvars, i) << ".\n";
#endif
		dim += combination(nvars, i);
	}
	//dbg_print();
	//values.clear();
	values.resize(dim);
	return dim;
}

int Polynomial::get_dimension() const {
	return dim;
}

bool Polynomial::scale(double times) {
	if (times == 0) return false;
	for (int i = 0; i < dim; i++)
		values[i] *= times;
	return true;
}

std::string Polynomial::to_string() const {
	std::ostringstream stm;
	stm << std::setprecision(12);
	bool firstplus = false;
	for (int j = 0; j < dim; j++) {
		if (values[j] == 0) continue;
		if (firstplus == false) {
			firstplus = true;
			if (values[j] < 0) 
				stm << " -";
		} else {
			if (values[j] < 0) 
				stm << " - ";
			else 
				stm << " + ";
		}

		if (j == 0) {
			stm << std::abs(values[j]);
			continue;
		}

		if (std::abs(values[j]) != 1)
			stm << std::abs(values[j]) << " *";

		stm << " {" << get_variable_name(j) << "} ";
	}

	switch (optype) {
		case ATOM_GEQ:
			stm << " >= ";
			break;
		case ATOM_LEQ:
			stm << " <= ";
			break;
		case ATOM_GT:
			stm << " > ";
			break;
		case ATOM_LT:
			stm << " < ";
			break;
		case ATOM_EQ:
			stm << " = ";
			break;
		case ATOM_NEQ:
			stm << " != ";
			break;
		default:
			stm << RED << "Unsupported optype";
			break;
	}
	stm << "0";
	return stm.str();
}


z3::expr Polynomial::to_z3_expr(z3::context& c) const {
	std::vector<z3::expr> x;
	for (int i = 0; i < nvars; i++) {
		//x.push_back(c.real_const(vars[i].c_str()));
		x.push_back(c.int_const(vars[i].c_str()));
	}

	std::vector<z3::expr> v;
	char real[65];
	for (int i = 0; i < dim; i++) {
		snprintf(real, 64, "%2.8f", values[i]);
		v.push_back(c.real_val(real));
	}

	z3::expr expr0 = v[0];
	for (int i = 1; i < dim; i++) {
		z3::expr tmp = v[i];
		tmp = tmp * x[i-1];
		/*
		   for (int j = 0; j < nvars ; j++) {
		   int power = vparray[i][j];
		   while (power-- > 0) {
		   tmp = tmp * x[j];
		   }
		   }
		   */
		expr0 = expr0 + tmp;
	}

	//std::cout << "expr1: " << expr1 << std::endl;
	
	z3::expr hypo = expr0;
	switch (optype) {
		case ATOM_GEQ:
			hypo = expr0 >= 0;
			break;
		case ATOM_LEQ:
			hypo = expr0 <= 0;
			break;
		case ATOM_GT:
			hypo = expr0 > 0;
			break;
		case ATOM_LT:
			hypo = expr0 < 0;
			break;
		case ATOM_EQ:
			hypo = expr0 == 0;
			break;
		case ATOM_NEQ:
			hypo = expr0 != 0;
			break;
		default:
			cout << RED << "Unsupported optype";
			break;
	}
	//z3::expr hypo = expr >= 0;
	//cout << "generating expr: " << hypo << endl;

	solver s(c);
	s.add(hypo);
	dbg_print();
	
	if (s.check() == unsat) {
		cout << "UNSAT\n";
	} else {
		cout << "SAT\n";
		dbg_print();
		model m = s.get_model();
		for (int i = 0; i < nvars; i++) {
			cout << "..." << vars[i] << "**" << x[i] << "**"<< ": " << (int)m.eval(x[i]) << endl;
		}
	}

	x.clear();
	v.clear();
	return hypo;
}

bool Polynomial::z3_implies(const Polynomial& e2) const {
#ifdef __PRT_QUERY
	std::cout << BLUE << "-------------uni-Imply solving-------------\n" << NORMAL;
	std::cout << RED << to_string() << " ==> " << e2 << std::endl << NORMAL;
#endif

	z3::config cfg;
	cfg.set("auto_config", true);
	z3::context c(cfg);

	z3::expr hypo = to_z3_expr(c);
	z3::expr conc = e2.to_z3_expr(c);
	z3::expr query = implies(hypo, conc);

#ifdef __PRT_QUERY
	std::cout << "\nhypo: " << hypo << std::endl;
	std::cout << "conc: " << conc << std::endl;
	std::cout << BLUE << "Query : " << query << std::endl << NORMAL;
#endif

	z3::solver s(c);
	s.add(!query);
	switch(s.check()) {
		case sat: cout << "SAT\n"; return false;
		case unsat: cout << "UNSAT\n"; return true;
		case unknown: cout << "Unknown"; break;
	}
	return false;
}

bool Polynomial::z3_imply(const Polynomial* e1, int e1_num, const Polynomial& e2) {
#ifdef __PRT_QUERY
	std::cout << "-------------Multi-Imply solving-------------\n";
#endif
	z3::config cfg;
	cfg.set("auto_config", true);
	z3::context c(cfg);

	z3::expr hypo = e1[0].to_z3_expr(c);
	for (int i = 1; i < e1_num; i++) {
		hypo = hypo && e1[i].to_z3_expr(c);;
	}
	z3::expr conc = e2.to_z3_expr(c);

	//std::cout << "hypo: " << hypo << std::endl;
	//std::cout << "conc: " << conc << std::endl;
	z3::expr query = implies(hypo, conc);

#ifdef __PRT_QUERY
	std::cout << "Query : " << query << std::endl;
	std::cout << "Answer: ";
#endif

	z3::solver s(c);
	s.add(!query);
	z3::check_result ret = s.check();

	if (ret == unsat) {
		std::cout << "True" << std::endl;
		return true;
	} else {
		std::cout << "False" << std::endl;
		return false;
	}
}


bool Polynomial::setup_terms() {
	terms.clear();
	terms.push_back(VariableTerm::make("ConstantOne"));
	for (size_t i = 1; i < vars.size(); i++) {
		//std::cout << "_names[" << i << "] = " << _names[i] << "\n";
		terms.push_back(VariableTerm::make(vars[i]));
	}
	return true;
}


Constraint Polynomial::to_mistral_expr() const {
	//std::cout << "convert >>>w: " << w.t() << "-----" << std::flush;
	map<Term*, long int> map0;
	for (size_t i = 1; i < vars.size(); i++) {
		map0[terms[i]] = (long int)(values.at(i));
		//cout << "-- map0[" << terms[i]->to_string() << ":" << (long int)(w.at(i)) << "]\n";
	}
	Term* term0 = ArithmeticTerm::make(map0, values.at(0));
	//std::cout << "   - term0:  " << term0->to_string() << std::endl;
	Constraint c(term0, ConstantTerm::make(0), optype);
	//std::cout << "-> constraint:  " << c << std::endl;
	return c;
}


bool Polynomial::mistral_solve(Constraint& c, vec& v) {
	//std::cout << GREEN << "solve constraint:" << c << std::endl << NORMAL;
	if (c.sat() == false) {
		//std::cout << "not sat!\n";
		return false;
	}
	v.resize(nvars);
	map<Term*, SatValue> solution;
	c.get_assignment(solution);
	//std::cout << "sat!\n solution:\n";
	for (auto it = solution.begin(); it != solution.end(); ++it) {
		for (size_t i = 1; i < vars.size(); i++) {
			if (it->first == terms.at(i)) {
				//cout << "#######" << terms.at(i)->to_string() 
				//<< " --> " << it->second.value << std::endl;
				v.at(i) = it->second.value.to_double();
				break;
			}
		}
	}
	v.at(0) = 1;
	//std::cout << " =>=>=>=>=>=> " << v.t(); 
	return true;

}


/** @brief This method is used to check whether *this polynomail is similar to given polynomail e or not.
 *		   *this ~= e ???
 *
 * @param precision defines how much variance we can bare.
 *		  The default is 4, which means we can bare 0.0001 difference.
 *		  In this case 1 ~=1.00001, but 1!~=1.000011
 */
bool Polynomial::is_similar(Polynomial& e2, int precision) {
	//precision += 6;
	if (dim != e2.get_dimension()) 
		return false;

	double ratio = 0;
	int i;
	for (i = 0; i < dim; i++) {
		if ((values[i] != 0) && (e2.get_coef(i) != 0)) {
			ratio = values[i] / e2.get_coef(i);
			break;
		}
	}
	if (i >= dim)
		return false;
	//std::cout << "1[ratio=" << ratio <<"]\n";
	double b1, b2;
	b1 = ratio * (1 + pow(0.1, precision));
	b2 = ratio * (1 - pow(0.1, precision));
	for (int i = 0; i < dim; i++) {
		if ((values[i] - e2.get_coef(i) * b1) * (values[i] - e2.get_coef(i) * b2) > 0)
			return false;
	}

	/*
	   double down, up;
	   if (ratio >= 0) {
	   up = ratio * (1 + pow(0.1, precision));
	   down = ratio * (1 - pow(0.1, precision));
	   } else {
	   up = ratio * (1 - pow(0.1, precision));
	   down = ratio * (1 + pow(0.1, precision));
	   }
	//std::cout << "[" << down << ", " << ratio << ", " << up << "]";
	for (int i = 0; i < dim; i++) {
	if (e2.get_coef(i) >= 0) {
	if ((values[i] < e2.get_coef(i) * down) || (values[i] > e2.get_coef(i) * up))
	return false;
	} else {
	if ((values[i] < e2.get_coef(i) * up) || (values[i] > e2.get_coef(i) * down))
	return false;
	}
	}
	*/
	return true;
}



Polynomial* Polynomial::roundoff_in_place() {
	Polynomial poly;
	this->roundoff(poly);
	*this = poly;
	return this;
}

bool Polynomial::already_roundoff() {
	for (int i = 0; i < dim; i++)
		if (int(values[i]) != values[i]) 
			return false;
	return true;
}

int Polynomial::roundoff(Polynomial& e) {
#ifdef _PRT_
	std::cout << "ROUND OFF " << *this << " --> ";
#endif
	if (already_roundoff() == true) {
		e = *this;
		return 0;
	}
	double max = 0;
	for (int i = 0; i < dim; i++) {
		if (std::abs(values[i]) > max) {
			max = std::abs(values[i]);
		}
	}

	double min = max;
	for (int i = 1; i < dim; i++) {
		if (std::abs(values[i]) == 0) continue;
		if (std::abs(values[i]) * pow(10, PRECISION) < max) {
			values[i] = 0;
			continue;
		}
		if (std::abs(values[i]) < min) {
			min = std::abs(values[i]);
		}
	}


#ifdef __PRT_POLYNOMIAL
	std::cout << GREEN << "Before roundoff: " << to_string();
	std::cout << " min=" << min << std::endl;
#endif

	e = *this;
	e.scale(1.0/min);
	double scale_up = 2;
	//e.values[0] = values[0]/min;
	while(scale_up <= 100) {
		if (e.roundoffable() == true)
			break;
		e.scale((1.0 * scale_up)/(scale_up-1));
		scale_up++;
	}
	if (scale_up > 100) {
		for (int i = 0; i < dim; i++) {
			roundoff_double(e.values[i], e.values[i]);
		}
	}

	//int poly_gcd = ngcd<double>(e.get_coefs());
	int poly_gcd = ngcd(e.get_coefs());
	e.values[0] = floor(e.values[0] / poly_gcd);
	if (poly_gcd > 1) {
		for (int i = 1; i < dim; i++) {
			e.values[i] = e.values[i] / poly_gcd;
		}
	}
#ifdef __PRT_POLYNOMIAL
	std::cout << "\tAfter roundoff: " << e << NORMAL << std::endl;
#endif
	//std::cout << e << std::endl;
	return 0;
}

bool Polynomial::roundoff_double(double x, double& roundx) {
	double xx = x;
	if (std::abs(xx) <= UPBOUND) {
		roundx = 0;
		return true;
	}
	roundx = nearbyint(xx);
	if ((roundx  >= xx * (1 - UPBOUND) && roundx  <= xx * (1 + UPBOUND))
			|| (roundx  <= xx * (1 - UPBOUND) && roundx  >= xx * (1 + UPBOUND))) {
		return true;
	}
	return false;
}

bool Polynomial::roundoffable() {
	std::vector<double> roundoffed;
	double tmp;
	for (int i = 0; i < dim; i++) {
		if (roundoff_double(values[i], tmp) == false) {
			roundoffed.clear();
			return false;
		}
		roundoffed.push_back(tmp);
	}
	set_coefs(roundoffed);
	roundoffed.clear();
	return true;
}


#include <cstdlib>

int test_polynomial() {
	cout << "testing polynomial class.\n";
	srand(time(NULL));
	//dbg_print();
	/*
	   Polynomial p0;
	   dbg_print();
	   p0.set_variable_name_list({"x0", "x1", "x2"});
	   for (int j = 0; j < p0.get_nvars(); j++) {
	   dbg_print();
	   p0.set_coef(j, rand() %1000 / 9.9);
	   cout << "p0.coef(" << j << ") = " << p0.get_coef(j) << endl;
	   }
	   cout << "-------------------------------------------------\n";
	   cout << p0.to_string() << endl;
	   p0.roundoff_in_place();
	   cout << p0.to_string() << endl;
	   return 0;
	   */

	Polynomial p[3];
	p[0].set_variable_name_list({"x0", "x1", "x2"});
	p[1].set_anonymous_list(4);
	p[2].set_variable_name_list({"X", "Y"});

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < p[i].get_dimension(); j++) {
			//dbg_print();
			p[i].set_coef(j, rand() % 1000 / (rand() % 99 / 11.0));
			//cout << "p[" << i << "].coef(" << j << ") = " << p[i].get_coef(j) << endl;
		}
	}

	for (int i = 0; i < 3; i++) {
		cout << BLUE << "------------------------------------------------------------------------------------" 
			<< "--------------------------------------------------------------------------------\n" << NORMAL;
		cout << RED << "p[" << i << "]:(orginal)--|--> " << NORMAL;
		cout << p[i].to_string() << endl;
		p[i].roundoff_in_place();
		cout << RED << "p[" << i << "]:(roundoff)-|--> " << NORMAL;
		cout << p[i].to_string() << endl;
	}

	return 0;
}

