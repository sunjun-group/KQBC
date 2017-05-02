# build kqbc c++ version

+ install armadillo first!.

+ compile kqbc with following commands:
    + g++ qbc.cpp -o qbc -larmadillo -std=c++11
-------------------------------------------------------------

+ new compile commands:
	+ g++ src/main.cpp src/qbc.cpp -o qbc -larmadillo -std=c++11 -Iinclude
	+ OR 
		+ mkdir -p build
		+ cd build
		+ cmake ..
		+ make
