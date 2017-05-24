#!/bin/bash

if [ $# -ge 1 ]; then
	g++ -I.. -I../cnode -I../solver -I../numeric-lib -I../term/ -std=c++11 $1 ../build/libmistral.a ../build/parser/libparser.a -o exmp -lgmp
else
	g++ -I.. -I../cnode -I../solver -I../numeric-lib -I../term/ -std=c++11 example.cpp ../build/libmistral.a ../build/parser/libparser.a -o exmp -lgmp
fi
