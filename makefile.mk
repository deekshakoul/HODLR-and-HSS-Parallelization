CC		=g++
CFLAGS	= -c  -fopenmp -Wall -march=native -O3 -funroll-loops -ffast-math -ffinite-math-only -I header/ 
LDFLAGS	=-fopenmp
SOURCES	=./examples/testHODLR.cpp #./examples/KDTree.cpp
OBJECTS	=$(SOURCES:.cpp=.o)
EXECUTABLE	=./exec/HODLR_Test

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o$@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.out ./examples/*.o ./exec/*

tar:
	tar -zcvf HODLR.tar.gz ./makefile.mk ./exec ./src ./header ./examples ./README.md ./LICENSE.md
