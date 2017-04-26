# KQBC
Python implementation for KQBC

## Linear learning
* Converge with 1000 inputs


## build
include with right path, and the -Wl,-rpath.

### ubuntu
[//]: # (g++ qbc.cpp -o qbc -I/usr/local/MATLAB/R2016b/extern/include -leng -lmat -lmx -lmex -lut -L/usr/local/MATLAB/R2016b/bin/glnxa64 -Wl,-rpath=/usr/local/MATLAB/R2016b/bin/glnxa64)
g++ qbc.cpp -o qbc -I/usr/local/MATLAB/R2016b/extern/include -leng -lmx -L/usr/local/MATLAB/R2016b/bin/glnxa64 -Wl,-rpath=/usr/local/MATLAB/R2016b/bin/glnxa64

### mac
[//]: # (export MatlabRoot="/Applications/MATLAB_R2016a.app")
[//]: # (export PATH=$MatlabRoot/bin:$PATH)
[//]: # (export DYLD_LIBRARY_PATH=$MatlabRoot/bin/maci64:$MatlabRoot/sys/os/maci64:$DYLD_LIBRARY_PATH)
[//]: # (g++ qbc.cpp -o qbc -I$MatlabRoot/extern/include/ -L$MatlabRoot/bin/maci64 -leng -lmx)

export PATH=/Applications/MATLAB_R2016a.app/bin:$PATH
export DYLD_LIBRARY_PATH=/Applications/MATLAB_R2016a.app/bin/maci64:/Applications/MATLAB_R2016a.app/sys/os/maci64:$DYLD_LIBRARY_PATH
g++ qbc.cpp -o qbc -I/Applications/MATLAB_R2016a.app/extern/include/ -L/Applications/MATLAB_R2016a.app/bin/maci64 -leng -lmx
