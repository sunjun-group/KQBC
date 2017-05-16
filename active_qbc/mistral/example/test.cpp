
#include <iostream>
#include "CNode.h"
#include "Constraint.h"
#include "term.h"


/*
 * Compile as: g++ -I.. -I../cnode -I../solver -I../numeric-lib -I../term/ -std=c++0x example.cpp ../build/libmistral.a ../build/parser/libparser.a -o exmp -lgmp
 * The, run ./my_project
 */

using namespace std;
#include <string>
#include <map>
int main(int c, char** argv)
{
	Term* a = VariableTerm::make("a");
	Term* b = VariableTerm::make("b");

	cout << "-----------------test------------------------" << endl;
	map<Term*, long int> map1;
	map<Term*, long int> map2;
	map1[a] = 3;
	map1[b] = 4;
	map2[a] = 1;
	map2[b] = 1;
	Term* term1 = ArithmeticTerm::make(map1, 20);
	Term* term2 = ArithmeticTerm::make(map2, 31);
	Constraint con1 (term1, ConstantTerm::make(8), ATOM_GT);
	Constraint con2 (term2, ConstantTerm::make(3.2), ATOM_LEQ);
	Constraint final_cons = con1 & con2;
	cout << "Result of sat(): " << (final_cons.sat() ? " yes " : " no " ) << endl;
	cout << "Simplified Constraint: " << final_cons << endl;

	map<Term*, SatValue> solutions;
	final_cons.get_assignment(solutions);
	cout << "Solutions:" << endl;
	for (auto it = solutions.begin(); it != solutions.end(); ++it) {
		if (it->first == a) cout << "Found a!\n";
		if (it->first == b) cout << "Found b!\n";
		cout << "  >" << it->first->to_string() << " = "<< it->second.value << endl;
	}
	return 0;
}
