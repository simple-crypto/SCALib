#include <math.h>
#ifndef ERROR_FACTOR
    #define ERROR_FACTOR 1E-4
#endif
#ifndef MIN_AND
    #define MIN_AND 4096
#endif
#ifndef MAX_IT
    #define MAX_IT (0x02ffff)
#endif
#ifndef ZERO
    #define ZERO 1E-10 
#endif
#define TILE (ZERO)
#define ROL16(a, offset) ((offset != 0) ? (((a) << offset) ^ ((a) >> (16-offset)))&0x0000ffff : a&0x0000ffff)

#define max(x,y) ((x) >= (y)) ? (x):(y)
#define min(x,y) ((x) <= (y)) ? (x):(y)

#define index(x,y,len) (((x)*(len))+(y))
#define P10(x) (pow(10,x))
