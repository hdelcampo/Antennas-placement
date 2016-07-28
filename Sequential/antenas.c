/**
 * Parallel computing (2015-2016 course)
 *
 * Antennas setup
 * Original implementation
 *
 * @author Javier
 */


// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>


// Include para las utilidades de computaciÃ³n paralela
#include "cputils.h"


/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
} Antena;


/**
 * Macro para acceder a las posiciones del mapa
 */
#define m(y,x) mapa[ (y * cols) + x ]


/**
 * FunciÃ³n de ayuda para imprimir el mapa
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
 * Distancia de una antena a un punto (y,x)
 * @note Es el cuadrado de la distancia para tener mÃ¡s carga
 */
int manhattan(Antena a, int y, int x){

	int dist = abs(a.x -x) + abs(a.y - y);
	return dist * dist;
}



/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar(int * mapa, int rows, int cols, Antena antena){

	m(antena.y,antena.x) = 0;

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			int nuevadist = manhattan(antena,i,j);

			if(nuevadist < m(i,j)){
				m(i,j) = nuevadist;
			}

		} // j
	} // i
}



/**
 * Calcular la distancia mÃ¡xima en el mapa
 */
int calcular_max(int * mapa, int rows, int cols){

	int max = 0;

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			if(m(i,j)>max){
				max = m(i,j);			
			}

		} // j
	} // i

	return max;
}


/**
 * Calcular la posiciÃ³n de la nueva antena
 */
Antena nueva_antena(int * mapa, int rows, int cols, int min){

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			if(m(i,j)==min){

				Antena antena = {i,j};
				return antena;
			}

		} // j
	} // i
}



/**
 * FunciÃ³n principal
 */
int main(int nargs, char ** vargs){


	//
	// 1. LEER DATOS DE ENTRADA
	//

	// Comprobar nÃºmero de argumentos
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
	printf("Calculando el nÃºmero de antenas necesarias para cubrir un mapa de"
		   " (%d x %d)\ncon una distancia mÃ¡xima no superior a %d "
		   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);

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
			fprintf(stderr,"Antena #%d estÃ¡ fuera del mapa\n",i);
			return -1;
		}
	}


	//
	// 2. INICIACIÃ“N
	//

	// Medir el tiempo
	double tiempo = cp_Wtime();

	// Crear el mapa
	int * mapa = malloc((size_t) (rows*cols) * sizeof(int) );

	// Iniciar el mapa con el valor MAX INT
	for(int i=0; i<(rows*cols); i++){
		mapa[i] = INT_MAX;
	}

	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		actualizar(mapa,rows,cols,antenas[i]);
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
	
	while(1){

		// Calcular el mÃ¡ximo
		int max = calcular_max(mapa, rows, cols);

		// Salimos si ya hemos cumplido el maximo
		if (max <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualizaciÃ³n del mapa
		Antena antena = nueva_antena(mapa, rows, cols, max);
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
