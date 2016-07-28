FLAGS= -O3

all: antenas antenas_debug

antenas: antenas.cu
	nvcc $(FLAGS)  antenas.cu   -o antenas 

antenas_debug: antenas.cu
	nvcc $(FLAGS) -DDEBUG  antenas.cu -o antenas_debug 

clean:
	rm -f antenas antenas_debug *.o *~
