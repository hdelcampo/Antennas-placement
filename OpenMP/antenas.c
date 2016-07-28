/**
 * Parallel computing (2015-2016 course)
 *
 * Antennas setup
 * OpenMP implementation
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


/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
} Antena;

typedef struct {
	int position;
	int value;
} max_data;


/**
 * Macro para acceder a las posiciones del mapa
 */
#define m(y,x) mapa[ (y * cols) + x ]

/**
 * Macros para acceder a posiciones
 */
#define position(y,x) (y * cols) + x
#define row(n)	((int)n) / cols
#define col(n)	n % cols
#define manhattan(antena, i, j) (abs(antena.x-j)+abs(antena.y-i)) * (abs(antena.x-j)+abs(antena.y-i))
#define nueva_antena(data)	{row(data.position), col(data.position)};

#define CHUNK 12
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
void actualizar(int *mapa, int rows, int cols, Antena antena){

	m(antena.y,antena.x) = 0;
	int j,i, nuevadist, half = rows/2;

	#pragma omp parallel default(none) shared(rows,cols,mapa,antena, half) private(i,j,nuevadist)
	{
	#pragma omp for nowait schedule(static, CHUNK)
	for(i=0; i<half;i++){
		j = antena.x;
		if(manhattan(antena,i,j) > m(i,j)) continue;
		while(j<cols){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i, j)){
				break;
			}
			m(i,j) = nuevadist;
			j++;
		}//j
	}//i
	#pragma omp for nowait schedule(static, CHUNK)
	for(i=0; i<half;i++){
		j = antena.x;
		if(manhattan(antena,i,j) > m(i,j)) continue;
		while(j>-1){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i, j)){
				break;
			}
			m(i,j) = nuevadist;
			j--;
		}//j
	}//i
	#pragma omp for nowait schedule(static, CHUNK)
	for(i=half; i<rows;i++){
		j = antena.x;
		if(manhattan(antena,i,j) > m(i,j)) continue;
		while(j<cols){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i, j)){
				break;
			}
			m(i,j) = nuevadist;
			j++;
		}//j
	}//i
	#pragma omp for nowait schedule(static, CHUNK)
	for(i=half; i<rows;i++){
		j = antena.x;
		if(manhattan(antena,i,j) > m(i,j)) continue;
		while(j>-1){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i, j)){
				break;
			}
			m(i,j) = nuevadist;
			j--;
		}//j
	}//i
	}//parallel
}

max_data calcular_max(int *mapa, int rows, int cols){
	int max = INT_MIN,
		i,j;
	int minPosition = rows*cols - 1;
	int *maxs = malloc(sizeof(int) * rows);

	#pragma omp parallel default(none) shared(mapa, rows, cols, maxs) private(i,j) reduction(max:max)
	{
	#pragma omp for schedule(static,CHUNK)
	for(i = 0; i < rows; i++){
		maxs[i]=0;
	}

	#pragma omp for nowait schedule(static, CHUNK)
	for(i = 0;i<rows;i++){
		for(j=0;j<cols;j++){
			if(m(i,j) > max){
				max = m(i,j);
				maxs[i] = j;
			}
		}
	}
	}

	for(i=0;i<rows;i++){
		if(m(i,maxs[i]) == max){
			minPosition = position(i, maxs[i]);
			break;
		}
	}

	free(maxs);
	max_data data = {minPosition, max};
	return data;
}

/**
 * Función principal
 */
int main(int nargs, char ** vargs){


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

	if(nAntenas<1 || nargs != ((nAntenas<<1)+5)){
		fprintf(stderr,"Error en la lista de antenas\n");
		return -1;
	}


	// Mensaje
	printf("Calculando el número de antenas necesarias para cubrir un mapa de"
		   " (%d x %d)\ncon una distancia máxima no superior a %d "
		   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);

	// Reservar memoria para las antenas
	Antena * antenas = malloc(sizeof(Antena) * (size_t) nAntenas);
	if(!antenas){
		fprintf(stderr,"Error al reservar memoria para las antenas inicales\n");
		return -1;
	}	

	int i;
	
	// Leer antenas
	for(i=0; i<nAntenas; i++){
		antenas[i].x = atoi(vargs[5+(i<<1)]);
		antenas[i].y = atoi(vargs[6+(i<<1)]);

		if(antenas[i].y<0 || antenas[i].y>=rows || antenas[i].x<0 || antenas[i].x>=cols ){
			fprintf(stderr,"Antena #%d está fuera del mapa\n",i);
			return -1;
		}
	}


	//
	// 2. INICIACIÓN
	//
	int size = rows * cols;
	// Medir el tiempo
	double tiempo = cp_Wtime();
	// Crear el mapa
	int * mapa = malloc((size_t) (size) * sizeof(int) );

	// Iniciar el mapa con el valor MAX INT
	#pragma omp parallel default(none) shared(size, mapa) private(i)
	{
	#pragma omp for schedule(static, CHUNK)
	for(i=0;i<size;i++){
		mapa[i] = INT_MAX;
	}
	}
	// Colocar las antenas iniciales
	for(i=0; i<nAntenas; i++){
		actualizar(mapa, rows, cols, antenas[i]);
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
	max_data max;
	
	while(1){
		// Calcular el máximo
		max = calcular_max(mapa, rows, cols);
		// Salimos si ya hemos cumplido el maximo
		if (max.value <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualización del mapa
		Antena antena = nueva_antena(max);
		actualizar(mapa,rows,cols,antena);
	}

	// Debug
#ifdef DEBUG
	print_mapa(mapa,rows,cols,NULL);
#endif

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;	

	// Salida
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);

	return 0;
}
