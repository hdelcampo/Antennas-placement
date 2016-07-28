/**
 * Computación Paralela
 * Funciones para las prácticas
 * 
 * @author Javier Fresno
 * @version 1.3 (curso 1415)
 * @version 1.4 (curso 1516)
 *
 */
#ifndef _CPUTILS_
#define _CPUTILS_

// Includes
#include <sys/time.h>


/*
 * FUNCIONES
 */

/**
 * Función que devuelve el tiempo
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, (void *) 0);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}



// Este año no se necesitan las funciones de fichero


#endif
