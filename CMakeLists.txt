cmake_minimum_required (VERSION 2.8)
#add_definitions(-D__SELECTIVE_SAMPLING_ENABLED)
project(QBC)

#add_definitions (-D_DBG_)
#add_definitions (-D_RAND_)
add_definitions (-D_Z3_)
#add_definitions (-D_MISTRAL_)
#add_definitions (-D_PRT_)
#add_definitions (-D__PRT_DEBUG)
#add_definitions (-D_BOUND_SOLVE_)
add_definitions (-D_LOG_)


if(UNIX)
	set(CMAKE_CXX_COMPILER             "g++")
	#set(CMAKE_CXX_FLAGS                "-Wall -g")
	#set(CMAKE_CXX_FLAGS                "-Wall -g -lgsl -lgslcblas")
	set(CMAKE_CXX_FLAGS                "-Wall -std=c++11") #" mistral/build/libmistral.a mistral/build/parser/libparser.a")
	set(CMAKE_CXX_FLAGS_DEBUG          "-g")
	set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
endif(UNIX)


file(GLOB HEADER "include/*.h")
source_group("Header Files" FILES ${HEADERS}) 

file(GLOB SRC "src/*.cpp")
source_group("Source Files" FILES ${SRC}) 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
find_package(Z3 REQUIRED)
find_package(GSL REQUIRED)



# add include dir and source dir into project
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
#AUX_SOURCE_DIRECTORY(test DIR_TEST)

#configure_file(config.h.in ../include/config.h)
include_directories (include)
include_directories(/usr/local/include/mistral)
#include_directories(/usr/local/include/nk)

#include_directories (mistral)
#include_directories (mistral/cnode)
#include_directories (mistral/solver)
#include_directories (mistral/numeric-lib)
#include_directories (mistral/term)

link_directories(/usr/local/lib/mistral)
#find_library(MISTR libmistral.a mistral/build mistral/build/parser NO_DEFAULT_PATH)
#find_library(PARSER libparser.a mistral/build mistral/build/parser NO_DEFAULT_PATH)
#link_libraries(${MISTR})
#link_libraries(${PARSER})

#add_executable(qbc src/main.cpp ${DIR_SRCS} ${HEADER})
link_libraries(libmistral.a libparser.a)
link_libraries(armadillo gmp) # nk)
link_libraries(${Z3_LIBRARY} ${GSL_LIBRARIES})
add_executable(qbc ${DIR_SRCS}) # ${HEADER})
#target_link_libraries(qbc mistral/build/libmistral.a)
#target_link_libraries(qbc mistral/build/parser/libparser.a)
