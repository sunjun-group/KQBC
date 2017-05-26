//#include <iostream>
#include <cmath>
#include "common.h"
using namespace std;

int gcd(int a, int b) {
	if (b==1 || a == b)
		return b;
	if (a==0 || b==0)
		return (a+b)>0? (a+b):1;
	if (a<b) {
		int tmp = a;
		a = b;
		b = tmp;
	}
	return gcd(b, a%b);
}

/*
template<typename T>
int ngcd(vector<T> v) {
	int n = v.size();
	for (int i = 0; i < n; i++) {
		if ((int)v[i] - v[i] != 0)
			return 1;
	}

	int coffgcd = std::abs((int)v[0]);
	for (int i = 1; i < n; i++) {
		if (coffgcd == 1)
			break;
		coffgcd = gcd(coffgcd, abs((int)v[i]));
	}
	return coffgcd;
}
*/

int ngcd(vector<double> v) {
	int n = v.size();
	for (int i = 0; i < n; i++) {
		if ((int)v[i] - v[i] != 0)
			return 1;
	}

	int coffgcd = std::abs((int)v[0]);
	for (int i = 1; i < n; i++) {
		if (coffgcd == 1)
			break;
		coffgcd = gcd(coffgcd, abs((int)v[i]));
	}
	return coffgcd;
}


int combination(int n, int m) {
	/*
#ifdef _PRT_
	std::cout << "combination(" << n << ", " << m << ")\n";
#endif
*/
	if (m == 0)
		return 1;
	if (m == 1)
		return n;
	if (n == m)
		return 1;
	return combination(n-1, m-1) + combination(n-1, m);
}
