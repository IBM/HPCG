/*
 * ibm.h
 *
 *  Created on: Jun 17, 2015
 *      Author: acm
 */

#if defined(__HAVE_HPM) || defined(__HAVE_MAIN_HPM)
#ifdef __cplusplus
extern "C" {
#endif
    void HPM_Start(char *);
    void HPM_Stop(char *);
    void summary_start();
    void summary_stop();
#ifdef __cplusplus
}
#endif
#endif

#if defined(__bgq__)
extern
#ifdef __cplusplus
"builtin"
#endif
void __alignx (int n, const void *addr);
#endif

#if defined(__bgq__) || defined(__PPC64__)
#include <builtins.h>
//#include "essl.h"
#endif

#if defined(__bgq__)
#include <spi/include/l1p/sprefetch.h>
#include <spi/include/l1p/pprefetch.h>
//#include "pprefetch.h"
#endif

// TO ENABLE THE CONSECUTIVE MEMORY ALLOCATION
#define __HAVE_NEW_DEFAULT_MEM_ALLOC

// TO ENABLE ELLPACK FORMAT
// NOTE: This is a format with the diagonal always as first element and possibly inverse of the diagonal as last element
// NOTE: The version without this is broken
#define __HAVE_ELLPACK_FORMAT

#if defined(__HAVE_ELLPACK_FORMAT)
// A VERSION WITHOUT STORING THE DIAGONAL
//#define __HAVE_ELLPACK_FORMAT_WITHOUTDIAG
#endif

// A VERSION WITH ASYNCHRONOUS SYMGS
#define __HAVE_ASYNC_SYMGS
