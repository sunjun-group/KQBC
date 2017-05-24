# (c) Isil Dillig (isil@cs.stanford.edu), Thomas Dillig (tdillig@cs.stanford.edu)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

set (CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -mtune=native -march=native -ftree-vectorize -funroll-loops -fstrict-aliasing  -fno-stack-protector -ffast-math -mfpmath=sse -msse3 -Wall -fmessage-length=0 -g ")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++0x -O2 -mtune=native -march=native -ftree-vectorize -funroll-loops -fstrict-aliasing  -fno-stack-protector -ffast-math -mfpmath=sse -msse3 -Wall -fmessage-length=0 -g ")

#set(CMAKE_CXX_FLAGS "-std=c++0x -g")
#set(CMAKE_CXX_FLAGS_DEBUG "-std=c++0x -g")



file(GLOB src "example.cpp")

include_directories(. .. ../cnode ../numeric-lib ../solver ../term) #-I*****
#link_directories(../build ../build/parser)
link_libraries(${PROJECT_SOURCE_DIR}/../build/libmistral.a ${PROJECT_SOURCE_DIR}/../build/parser/libparser.a gmp)
#link_libraries(../build/libmistral.a ../build/parser/libparser.a)
#link_libraries(/home/lijiaying/Research/GitHub/KQBC/active_qbc/mistral/build/libmistral.a /home/lijiaying/Research/GitHub/KQBC/active_qbc/mistral/build/parser/libparser.a)
add_executable(example ${src})
target_link_libraries(example gmp)

#add_subdirectory(..)
