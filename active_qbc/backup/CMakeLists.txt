cmake_minimum_required (VERSION 2.8)
#add_definitions(-D__SELECTIVE_SAMPLING_ENABLED)
project(QBC)

#add_definitions (-D__PRT)
#add_definitions (-D__PRT_DEBUG)


if(UNIX)
	set(CMAKE_CXX_COMPILER             "g++")
	#set(CMAKE_CXX_FLAGS                "-Wall -g")
	#set(CMAKE_CXX_FLAGS                "-Wall -g -lgsl -lgslcblas")
	set(CMAKE_CXX_FLAGS                "-Wall -std=c++11 -larmadillo")
	set(CMAKE_CXX_FLAGS_DEBUG          "-g")
	set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
endif(UNIX)


file(GLOB HEADER "include/*.h")
source_group("Header Files" FILES ${HEADERS}) 

file(GLOB SRC "src/*.cpp")
source_group("Source Files" FILES ${SRC}) 

# add include dir and source dir into project
#configure_file(config.h.in ../include/config.h)
include_directories (include)
#include_directories(.)
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
#AUX_SOURCE_DIRECTORY(test DIR_TEST)

#add_executable(qbc src/main.cpp ${DIR_SRCS} ${HEADER})
add_executable(qbc ${DIR_SRCS} ${HEADER})
