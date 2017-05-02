cc=g++
target=qbc
object=main.o qbc.o

ALL: $(target)


$(target): $(object)
	g++ $(object) -o $(target) -larmadillo

main.o: include/oracle.h include/qbc.h
	g++ $< -o $@ -std=c++11

qbc.o: include/oracle.h include/qbc.h
	g++ $< -o $@ -std=c++11
