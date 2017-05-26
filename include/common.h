#ifndef _COMMON_H_
#define _COMMON_H_
#include <vector>
#include "color.h"
#include "dbg.h"
using namespace std;

#define PRECISION 2

int gcd(int a, int b);

/*
template<typename T>
int ngcd(vector<T> v);
*/

int ngcd(vector<double> v);

int combination(int n, int m);

#endif
