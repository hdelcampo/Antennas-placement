/**
 * Parallel computing (2015-2016 course)
 *
 * Antennas setup
 * CUDA implementation
 *
 * @author Hector Del Campo Pando
 * @author Alberto Gutierrez Perez
 */

// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>


// Include para las utilidades de computación paralela
#include <time.h>
#include <cuda.h>
#include <math.h>

/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
} Antena;

/**
 * Estructura para guardar la informacion del numero maximo global
 */
typedef struct {
	int max;
	int pos;
} Max_data;


/**
 * Macros para acceder al maximo y su posicion
 */
#define valor(m) m->max
#define pos(max) max->pos

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

#define posicion_thread() (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.x + threadIdx.y * blockDim.x)

/**
 * Macros para la reduccion
 */
#define MAX_THREADS 1024
#define NUM_THREADS_BLOCK 256
#define NUM_BLOCKS ((int)rows*cols/NUM_THREADS_BLOCK + 1)

__global__ void iniciarMapa(int *mapa, int rows, int cols){
	int posicion = posicion_thread();
	if(posicion < rows*cols)
		mapa[posicion] = INT_MAX;
}


template <unsigned int blockSize>
__global__ void max_kernel(int *entrada_max, int *entrada_pos, int *salida_max, int *salida_pos, int size){
	extern __shared__ int s[];

	unsigned int nHilos = blockDim.x;
	unsigned int id = threadIdx.x;	//id thread en bloque
	unsigned int idBloque = blockIdx.x;	//id del bloque en grid
	unsigned int posicion = id + idBloque*nHilos;	//Posicion en el mapa

	int *maximos = (int*)s;
	int *posiciones = (int*)&maximos[nHilos];

	int myMax = INT_MIN, pos = INT_MAX;

	if(entrada_pos == NULL){
		while(posicion < size){
			if(entrada_max[posicion] > myMax){
				myMax = entrada_max[posicion];
				pos = posicion;
			}
			posicion+=(nHilos*gridDim.x);
		}
	}else{
		while(posicion < size){
			if(entrada_max[posicion] > myMax){
				myMax = entrada_max[posicion];
				pos = entrada_pos[posicion];
			}
			posicion+=(nHilos*gridDim.x);
		}
	}

	maximos[id] = myMax;
	posiciones[id] = pos;
	__syncthreads();



	if(blockSize>= 1024){
		if( id < 512)
			if(maximos[id + 512] > maximos[id] || (maximos[id + 512] == maximos[id] && posiciones[id+512] < posiciones[id])){
				maximos[id] = maximos[id+512];
				posiciones[id] = posiciones[id+512];
			}
			__syncthreads();
	}
	if(blockSize>= 512){
		if( id < 256)
			if(maximos[id + 256] > maximos[id] || (maximos[id + 256] == maximos[id] && posiciones[id+256] < posiciones[id])){
				maximos[id] = maximos[id+256];
				posiciones[id] = posiciones[id+256];
			}
			__syncthreads();
	}
	if(blockSize>= 256){
		if( id < 128)
			if(maximos[id + 128] > maximos[id] || (maximos[id + 128] == maximos[id] && posiciones[id+128] < posiciones[id])){
				maximos[id] = maximos[id+128];
				posiciones[id] = posiciones[id+128];
			}			
		__syncthreads();
	}
	if(blockSize>= 128){
		if( id < 64)
			if(maximos[id + 64] > maximos[id] || (maximos[id + 64] == maximos[id] && posiciones[id+64] < posiciones[id])){
				maximos[id] = maximos[id+64];
				posiciones[id] = posiciones[id+64];
			}
			__syncthreads();
	}

	if(id < 32){
		if(blockSize>= 64){	if((maximos[id + 32] > maximos[id] || (maximos[id + 32] == maximos[id] && posiciones[id+32] < posiciones[id]))){maximos[id] = maximos[id+32]; posiciones[id] = posiciones[id+32];}}
		if(blockSize>=32){	if((maximos[id + 16] > maximos[id] || (maximos[id + 16] == maximos[id] && posiciones[id+16] < posiciones[id]))){maximos[id] = maximos[id+16]; posiciones[id] = posiciones[id+16];}}
		if(blockSize>=16){	if((maximos[id + 8] > maximos[id] || (maximos[id + 8] == maximos[id] && posiciones[id+8] < posiciones[id]))){maximos[id] = maximos[id+8]; posiciones[id] = posiciones[id+8];}}
		if(blockSize>=8){	if((maximos[id + 4] > maximos[id] || (maximos[id + 4] == maximos[id] && posiciones[id+4] < posiciones[id]))){maximos[id] = maximos[id+4]; posiciones[id] = posiciones[id+4];}}
		if(blockSize>=4){	if((maximos[id + 2] > maximos[id] || (maximos[id + 2] == maximos[id] && posiciones[id+2] < posiciones[id]))){maximos[id] = maximos[id+2]; posiciones[id] = posiciones[id+2];}}
		if(blockSize>=2){	if((maximos[id + 1] > maximos[id] || (maximos[id + 1] == maximos[id] && posiciones[id+1] < posiciones[id]))){maximos[id] = maximos[id+1]; posiciones[id] = posiciones[id+1];}}
	}

	if(id == 0){
		salida_max[idBloque] = maximos[0];
		salida_pos[idBloque] = posiciones[0];
	}

}

__global__ void actualizar_kernel( int *mapa, Antena antena, int rows, int cols){
	m(antena.y, antena.x) = 0;
	int nuevadist;
	unsigned int desplHor = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int desplVer = threadIdx.y + blockIdx.y*blockDim.y;
	int j;

	for(int i = antena.y - desplVer; i >= 0; i-=blockDim.y){
		j = antena.x + desplHor;
		nuevadist = manhattan(antena,i,j);
		if(nuevadist > m(i,j))	break;
		for(; j < cols; j+=blockDim.x){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i,j))	break;
			m(i,j) = nuevadist;
		}
	}

	for(int i = antena.y + desplVer; i < rows; i+=blockDim.y){
		j = antena.x - desplHor;
		nuevadist = manhattan(antena,i,j);
		if(nuevadist > m(i,j))	break;
		for(; j >= 0; j-=blockDim.x){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i,j))	break;
			m(i,j) = nuevadist;
		}
	}

	for(int i = antena.y + desplVer; i < rows; i+=blockDim.y){
		j = antena.x + desplHor;
		nuevadist = manhattan(antena,i,j);
		if(nuevadist > m(i,j))	break;
		for(; j < cols; j+=blockDim.x){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i,j))	break;
			m(i,j) = nuevadist;
		}
	}

	for(int i = antena.y - desplVer; i >= 0; i-=blockDim.y){
		j = antena.x - desplHor;
		nuevadist = manhattan(antena,i,j);
		if(nuevadist > m(i,j))	break;
		for(; j >= 0; j-=blockDim.x){
			nuevadist = manhattan(antena,i,j);
			if(nuevadist > m(i,j))	break;
			m(i,j) = nuevadist;
		}
	}
}

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
 * Funcion depuracion para CUDA
 */
void print_mapa_cuda(int *mapa, int rows, int cols){
		int *m = (int*)malloc(sizeof(int)*rows*cols);
		cudaMemcpy(m, mapa, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost);
		print_mapa(m,rows,cols,NULL);
		free(m);
		getchar();
}

/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar(int *mapa, Antena antena, int rows, int cols){
	dim3 hilos(6,6);
	dim3 bloques(32);
	actualizar_kernel<<<bloques, hilos>>>(mapa,antena,rows,cols);
}



/**
 * Calcular la distancia máxima en el mapa
 */
Max_data calcular_max(int * mapa, int *maximos, int *posiciones, int rows, int cols){
	int posicion;
	int maximo;
	int aux = NUM_BLOCKS, nBloques = NUM_BLOCKS;

	//reduccion por bloques
	max_kernel<NUM_THREADS_BLOCK><<<nBloques, NUM_THREADS_BLOCK, sizeof(int)*NUM_THREADS_BLOCK*2>>>(mapa, NULL, maximos, posiciones,rows*cols);

	//reduccion de los bloques
	nBloques = (nBloques - 1)/NUM_THREADS_BLOCK + 1;

	while(aux != 1){
		max_kernel<NUM_THREADS_BLOCK><<<nBloques, NUM_THREADS_BLOCK, sizeof(int)*2*NUM_THREADS_BLOCK>>>(maximos, posiciones, maximos, posiciones, aux);
		aux = nBloques;
		nBloques = (nBloques - 1)/NUM_THREADS_BLOCK + 1;
	}
	//Fin reduccion bloques

	cudaMemcpy(&maximo, &maximos[0], sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&posicion, &posiciones[0], sizeof(int), cudaMemcpyDeviceToHost);

	Max_data max_global = {maximo, posicion};
	return max_global;
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

	if(nAntenas<1 || nargs != (nAntenas*2+5)){
		fprintf(stderr,"Error en la lista de antenas\n");
		return -1;
	}


	// Mensaje
	printf("Calculando el número de antenas necesarias para cubrir un mapa de"
	   " (%d x %d)\ncon una distancia máxima no superior a %d "
	   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);

	// Reservar memoria para las antenas
	Antena *antenas = (Antena*)malloc(sizeof(Antena) * (size_t) nAntenas);
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
	clock_t reloj = clock();
	double tiempo;
	cudaSetDevice(0);

	// Crear el mapa
	int * mapa;
	cudaMalloc((void**) &mapa , (rows*cols) * sizeof(int) );

	// Iniciar el mapa con el valor MAX INT
	iniciarMapa<<<NUM_BLOCKS, NUM_THREADS_BLOCK>>>(mapa,rows,cols);

	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		actualizar(mapa, antenas[i], rows, cols);
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
	Max_data max;

	// Variables para CUDA
	int nBloques = NUM_BLOCKS;
	
	int *posiciones;
	int *maximos;
	cudaMalloc((void**) &posiciones, nBloques * sizeof(int));
	cudaMalloc((void**) &maximos, nBloques * sizeof(int));

	while(1){

		// Calcular el máximo
		max = calcular_max(mapa,maximos,posiciones,rows,cols);

		// Salimos si ya hemos cumplido el maximo
		if (max.max <= distMax) break;	

		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualización del mapa
		Antena antena = nueva_antena(max.pos);
		actualizar(mapa,antena,rows,cols);

	}

	reloj = clock() - reloj;

	// Debug
#ifdef DEBUG
	print_mapa(mapa,rows,cols,NULL);
#endif
	
	//Liberar recursos en el host
	cudaFree(mapa);
	cudaFree(posiciones);
	cudaFree(maximos);

	cudaDeviceReset();

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = (double)reloj / CLOCKS_PER_SEC ;

	// Salida
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);


	return 0;
}




