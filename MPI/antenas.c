/**
 * Parallel computing (2015-2016 course)
 *
 * Antennas setup
 * MPI implementation
 *
 * @author Hector Del Campo Pando
 * @author Alberto Gutierrez Perez
 */

// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>


// Include para las utilidades de computación paralela
#include "cputils.h"
#include <mpi.h>

/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
} Antena;

/**
 * Estructura para cada nodo con informacion sobre sus limites
 */
typedef struct {
	int begin_row;
	int last_row;
	int rank;
	int size;
} Nodo;

/**
 * Estructura para guardar la informacion del numero maximo global
 */
typedef struct {
	int value;
	int pos;
} Max_data;

/**
 * Macros para acceder al maximo y su posicion
 */
#define valor(max) max.value
#define pos(max) max.pos

/**
 * Macro para acceder a las posiciones del mapa
 */
#define m(y,x) mapa[ (y * cols) + x ]
#define posicion(y,x) (y * cols) + x
#define row(n)	((int)n) / cols
#define col(n)	n % cols

/**
 * Macro para la funcion manhattan
 */
#define manhattan(a, i, j) (abs(a.x -j) + abs(a.y - i)) * (abs(a.x -j) + abs(a.y - i))
#define nueva_antena(n)	{row(n), col(n)}

/**
 * Macros para ayudar a los calculos de los nodos
 */
#define num_iteraciones (int)(rows / size)

/**
 * Operacion personalizada para el maximo
 */
MPI_Op myOp;

/**
 * Función de ayuda para imprimir el mapa
 */
void print_mapa(int * mapa, int rows, int cols, Antena * a){


	if(rows > 50 || cols > 30){
		printf("Mapa muy grande para imprimir\n");
		return;
	};

	#define ANSI_COLOR_RED     "\x1b[31m"
	#define ANSI_COLOR_GREEN   "\x1b[32m"
	#define ANSI_COLOR_RESET   "\x1b[0m"

	printf("Mapa [%d,%d]\n",rows,cols);
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			int val = m(i,j);

			if(val == 0){
				if(a != NULL && a->x == j && a->y == i){
					printf( ANSI_COLOR_RED "   A"  ANSI_COLOR_RESET);
				} else { 
					printf( ANSI_COLOR_GREEN "   A"  ANSI_COLOR_RESET);
				}
			} else {
				printf("%4d",val);
			}
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar(int * mapa, int rows, int cols, Antena antena, Nodo nodo){

	m(antena.y,antena.x) = 0;
	int nuevadist = 0;

	for(int i=nodo.begin_row; i<=nodo.last_row; i++){
		for(int j=antena.x; j<cols; j++){

			nuevadist = manhattan(antena,i,j);

			if(nuevadist > m(i,j)){
				break;
			}

			m(i,j) = nuevadist;

		} // j
	} // i

	for(int i=nodo.begin_row; i<=nodo.last_row; i++){
		for(int j=antena.x; j>-1; j--){

			nuevadist = manhattan(antena,i,j);

			if(nuevadist > m(i,j)){
				break;
			}
				m(i,j) = nuevadist;
		}//j
	}//i

}



/**
 * Calcular la distancia máxima en el mapa
 */
Max_data calcular_max(int * mapa, int rows, int cols, Nodo nodo){
	
	Max_data max = {0,0};
	Max_data max_global = {0,0};

	//Cada nodo busca el maximo de su trozo
	for(int i=nodo.begin_row; i<=nodo.last_row; i++){
		for(int j=0; j<cols; j++){
			if(m(i,j) > valor(max)){
				valor(max) = m(i,j);			
				pos(max) = posicion(i,j);
			}
		} // j
	} // i

	MPI_Reduce(&max, &max_global, 1, MPI_2INT, myOp, 0, MPI_COMM_WORLD);
	MPI_Bcast(&max_global, 1, MPI_2INT, 0, MPI_COMM_WORLD);

	return max_global;
}

/**
 * Funcion personalizada para encontrar el maximo con un reduce
 */
void max_min(void *in, void *inout, int *len, MPI_Datatype *type){

	for(int i=0; i < *len; i++){
		if(valor(((Max_data*)inout)[i]) <= valor(((Max_data*)in)[i]) &&
			pos(((Max_data*)inout)[i]) > pos(((Max_data*)in)[i])){
			valor(((Max_data*)inout)[i]) = valor(((Max_data*)in)[i]);
			pos(((Max_data*)inout)[i]) = pos(((Max_data*)in)[i]);
		}
	}
}

/**
 * Función principal
 */
int main(int nargs, char ** vargs){

	int rank, size;
	MPI_Init(&nargs, &vargs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//
	// 1. LEER DATOS DE ENTRADA
	//

	// Comprobar número de argumentos
	if(nargs < 7){
		fprintf(stderr,"Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",vargs[0]);
		return -1;
	}

	// Leer los argumentos de entrada
	int rows = atoi(vargs[1]);
	int cols = atoi(vargs[2]);
	int distMax = atoi(vargs[3]);
	int nAntenas = atoi(vargs[4]);

	if(nAntenas<1 || nargs != (nAntenas*2+5)){
		fprintf(stderr,"Error en la lista de antenas\n");
		return -1;
	}


	// Mensaje
	if(rank == 0){
		printf("Calculando el número de antenas necesarias para cubrir un mapa de"
		   " (%d x %d)\ncon una distancia máxima no superior a %d "
		   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);
	}

	// Reservar memoria para las antenas
	Antena * antenas = malloc(sizeof(Antena) * (size_t) nAntenas);
	if(!antenas){
		fprintf(stderr,"Error al reservar memoria para las antenas inicales\n");
		return -1;
	}	
	
	// Leer antenas
	for(int i=0; i<nAntenas; i++){
		antenas[i].x = atoi(vargs[5+i*2]);
		antenas[i].y = atoi(vargs[6+i*2]);

		if(antenas[i].y<0 || antenas[i].y>=rows || antenas[i].x<0 || antenas[i].x>=cols ){
			fprintf(stderr,"Antena #%d está fuera del mapa\n",i);
			return -1;
		}
	}


	//
	// 2. INICIACIÓN
	//

	// Medir el tiempo
	MPI_Barrier(MPI_COMM_WORLD);
	double tiempo = MPI_Wtime();

	int minIndex= num_iteraciones*rank;
	int maxIndex = ((rank+1)*num_iteraciones)-1;
	int dif = rows - (num_iteraciones*size);
	if(rank < dif ){
		minIndex += rank;
		maxIndex += rank+1;
	}else{
		minIndex += dif;
		maxIndex += dif;
	}
	
	//Indicar a cada nodo cuanto trabajo ha de realizar
	Nodo nodo = {minIndex, maxIndex, rank, size};
	
	// Crear el mapa
	int * mapa = malloc((size_t) (rows*cols) * sizeof(int) );


#ifdef DEBUG_NODO
	printf("Procesador:%d indice:%d indice final:%d\n", rank, nodo.begin_row, nodo.last_row);
#endif


	// Iniciar el mapa con el valor MAX INT
	for(int i=nodo.begin_row; i<=nodo.last_row; i++){
		for(int j=0; j<cols; j++){
			m(i,j) = INT_MAX;
		}
	}

	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		actualizar(mapa,rows,cols,antenas[i],nodo);
	}

	// Debug
#ifdef DEBUG
	print_mapa(mapa,rows,cols,NULL);
#endif


	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	// Contador de antenas
	int nuevas = 0;

	//Declaracion de todo lo necesario para encontrar el max
	Max_data max;
	MPI_Op_create(max_min, 1, &myOp);
	
	while(1){

		// Calcular el máximo
		max = calcular_max(mapa, rows, cols, nodo);

		// Salimos si ya hemos cumplido el maximo
		if (valor(max) <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualización del mapa
		Antena antena = nueva_antena(pos(max));
		actualizar(mapa,rows,cols,antena,nodo);

	}

	// Debug
#ifdef DEBUG
	print_mapa(mapa,rows,cols,NULL);
#endif

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	MPI_Barrier(MPI_COMM_WORLD);
	tiempo = MPI_Wtime() - tiempo;

	MPI_Op_free(&myOp);

	// Salida
	if(rank == 0){
		printf("Result: %d\n",nuevas);
		printf("Time: %f\n",tiempo);
	}

	MPI_Finalize();

	return 0;
}



