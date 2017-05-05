# build kqbc c++ version

- install armadillo first!.
```
sudo apt-get install liblapack-dev
sudo apt-get install libblas-dev
sudo apt-get install libboost-dev
sudo apt-get install libarmadillo-dev
```
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
./qbc 1
	
```

-------------------------------------------------------------
- compile kqbc with following commands:
    - g++ qbc.cpp -o qbc -larmadillo -std=c++11
- g++ src/main.cpp src/qbc.cpp -o qbc -larmadillo -std=c++11 -Iinclude
