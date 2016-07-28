

FLAGS=-std=c11 -O3

all: antenas antenas_debug

antenas: antenas.c cputils.h
	mpicc $(FLAGS)  antenas.c   -o antenas 

antenas_debug: antenas.c cputils.h
	mpicc $(FLAGS) -DDEBUG  antenas.c -o antenas_debug 

clean:
	rm -f antenas antenas_debug *.o *~
