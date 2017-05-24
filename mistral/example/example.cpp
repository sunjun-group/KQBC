 #include <iostream>
 #include "CNode.h"
 #include "Constraint.h"
 #include "term.h"

 using namespace std;
 #include <string>
 #include <map>
 
 /*
 * Compile as: g++ -I.. -I../cnode -I../solver -I../numeric-lib -I../term/ -std=c++0x example.cpp ../build/libmistral.a ../build/parser/libparser.a -o exmp -lgmp
 * The, run ./my_project
 */
 
 int main(int c, char** argv)
 {
   
   Term* a = VariableTerm::make("a");
   Term* b = VariableTerm::make("b");
   map<Term*, long int> elems;
   elems[a] = 3;
   elems[b] = 7;
   Term* t4 = ArithmeticTerm::make(elems, 0);
   Term* x = VariableTerm::make("x");
   cout << "Term t4 is: " << t4->to_string() << endl;
   
   
   Constraint c1(t4, x, ATOM_GT);
   cout << "c1: " << c1 << endl;
   
   Constraint c2(a, ConstantTerm::make(5), ATOM_EQ);
   cout << "c2: " << c2 << endl;
   
   c2 &=c1;
   cout << "c2: " << c2 << endl;
   
   assert(a->get_term_type() == VARIABLE_TERM);
   VariableTerm* vt = static_cast<VariableTerm*>(a);
   c2.eliminate_evar(vt);
   
   cout << "c2 after eliminating a: " << c2 << endl;
   
   
   //replace b by 88;
   c2.replace_term(b, ConstantTerm::make(88));
   
   cout << "c2 after replacing b with 88: " << c2 << endl;
   
   
   //Some satifiability queries
   Constraint c3(x, ConstantTerm::make(7), ATOM_EQ);
   
   cout << "c3:: " << c3 << endl;
   Constraint new_c = c3 & c2;
   cout << "new_c: " << new_c << endl;
   
   cout << "Is this constraint SAT? (use sat_discard() to check this.)" << (new_c.sat_discard() ? " yes " : " no " ) << endl;
   cout << "~~~~~~~~~~~~~~~~~~~~~ " << endl;
   cout << "Alternatively, use sat() to check satisfiability & bring constraint to simplified form " << endl;
   cout << "(See paper http://www.stanford.edu/~isil/sas2010.pdf). Use this if you reuse the constraint or want it simplified." << endl;
   cout << "~~~~~~~~~~~~~~~~~~~~~ " << endl;
   cout << "Result of sat(): " << (new_c.sat() ? " yes " : " no " ) << endl;
   cout << "Simplified Constraint: " << new_c << endl;
   //new_c.get_assignment(solutions);
   
   cout << "-----------------new test------------------------" << endl;
   map<Term*, long int> map1;
   map<Term*, long int> map2;
   map1[a] = 3;
   map1[b] = 4;
   map2[a] = 1;
   map2[b] = 1;
   Term* term1 = ArithmeticTerm::make(map1, 0);
   Term* term2 = ArithmeticTerm::make(map2, 0);
   Constraint con1 (term1, ConstantTerm::make(8), ATOM_GT);
   Constraint con2 (term2, ConstantTerm::make(2), ATOM_LEQ);
   Constraint final_cons = con1 & con2;
   map<Term*, SatValue> solutions;
   cout << "Result of sat(): " << (final_cons.sat() ? " yes " : " no " ) << endl;
   cout << "Simplified Constraint: " << final_cons << endl;
   final_cons.get_assignment(solutions);
   cout << "Solutions:" << endl;
   for (auto it = solutions.begin(); it != solutions.end(); ++it)
	   cout << "  >" << it->first->to_string() << " = "<< it->second.value << endl;
 }
