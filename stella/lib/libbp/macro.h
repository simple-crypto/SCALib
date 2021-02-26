#include <math.h>
#ifndef ZERO
    #define ZERO 1E-10 
#endif
#define TILE (ZERO)
#define ROL16(a, offset) ((offset != 0) ? (((a) << offset) ^ ((a) >> (16-offset)))&0x0000ffff : a&0x0000ffff)

#define max(x,y) ((x) >= (y)) ? (x):(y)
#define min(x,y) ((x) <= (y)) ? (x):(y)

#define index(x,y,len) (((x)*(len))+(y))
#define P10(x) (pow(10,x))
