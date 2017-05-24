# (c) Isil Dillig (isil@cs.stanford.edu), Thomas Dillig (tdillig@cs.stanford.edu)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

set (CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -mtune=native -march=native -ftree-vectorize -funroll-loops -fstrict-aliasing  -fno-stack-protector -ffast-math -mfpmath=sse -msse3 -Wall -fmessage-length=0 -g ")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++0x -O2 -mtune=native -march=native -ftree-vectorize -funroll-loops -fstrict-aliasing  -fno-stack-protector -ffast-math -mfpmath=sse -msse3 -Wall -fmessage-length=0 -g ")

#set(CMAKE_CXX_FLAGS "-std=c++0x -g")
#set(CMAKE_CXX_FLAGS_DEBUG "-std=c++0x -g")



file(GLOB MY_SRC "*.cpp" "cnode/*.cpp" "elimination/*.cpp" "numeric-lib/*.cpp" "sat-solver/*.cpp" "solver/*.cpp" "term/*.cpp")

include_directories(. ./cnode ./numeric-lib ./elimination ./sat-solver ./solver ./term ./parser ./smtparser)
add_library(mistral ${MY_SRC})
TARGET_LINK_LIBRARIES(mistral parser smtparser)


file(GLOB HEADER ./*.h ./cnode/*.h ./numeric-lib/*.h ./elimination/*.h ./sat-solver/*.h ./solver/*.h ./term/*.h ./parser/*.h ./smtparser/*.h)
install(FILES ${HEADER} DESTINATION include/mistral)
install(TARGETS mistral ARCHIVE DESTINATION lib/mistral)


add_subdirectory(parser)
add_subdirectory(smtparser)
add_subdirectory(ui)

