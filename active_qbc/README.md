# build kqbc c++ version

- install armadillo first!.
```
sudo apt-get install liblapack-dev
sudo apt-get install libblas-dev
sudo apt-get install libboost-dev
sudo apt-get install libarmadillo-dev
```
	- useful link: [install armadillo](http://www.uio.no/studier/emner/matnat/fys/FYS4411/v13/guides/installing-armadillo/)

- build kqbc
```
mkdir -p build
cd build
cmake ..
make
```

- Run
```
cd build
./qbc 1 # test qbc with embeded parameters
./qbc
	
```

-------------------------------------------------------------
- compile kqbc with following commands:
    - g++ qbc.cpp -o qbc -larmadillo -std=c++11
- g++ src/main.cpp src/qbc.cpp -o qbc -larmadillo -std=c++11 -Iinclude
