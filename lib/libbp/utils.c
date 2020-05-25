#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "macro.h"
#include <assert.h>
/*
 * pre: in is a distribution
 *      len of a
 * post:
 *      return the sum over all the elements of the distri
 */
proba_t sum(const proba_t *in,uint32_t len){
    uint32_t i;
    proba_t sum = 0;
    for(i=0;i<len;i++){
        sum = in[i];
    }
    return sum;
}

/*
 *  pre: in is a distribution of size len
 *  post: out is the log of the input distriubution
 */
void apply_log10(lproba_t * out,const proba_t *in,uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = log10(in[i]);
    }
}


/*
 * pre: in is a lproba_t of size len
 * post: out is the distribution of in
 */
void apply_P10(proba_t * out,const lproba_t *in,uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = P10(in[i]);
    }
}



/*
 * pre: in is a distribution of size len
 * post: the min of the distribution is set to val
 */
void tile(proba_t *out, const proba_t *in, proba_t val, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = ((in[i])<(val)) ? val:in[i];
    }
}

/*
 * add a cst to the distribution out of size len
 */
void add_cst(proba_t * out,proba_t cst,uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = out[i] +cst;
    }
}

void add_cst_dest(proba_t *out, const proba_t *in, proba_t val, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = in[i]+val;
    }
}
void abs_vec(proba_t *out, const proba_t *in, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = abs((in[i]));
    }
}
/*
 * div by a cst the distribution out of size len
 */
void div_cst(proba_t * out,const proba_t *in,proba_t cst,uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = (in[i])/cst;
    }
}

/*
 * return the minimum of the distribution in
 */
proba_t get_min(const proba_t *in,uint32_t len){
    proba_t min;
    uint32_t i;
    min = DBL_MAX;
    for(i=0;i<len;i++){
        min = min<in[i] ? min:in[i];
    }
    return min;
}

/*
 * add distributions in1 and in2 of size len
 * out is the output vector 
 */
void add_vec(proba_t *out, const proba_t *in1, const proba_t *in2, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = in1[i] + in2[i];
    }
}
/*
 * multiply to vectors and store the result in out
 */
void mult_vec(proba_t *out, const proba_t *in1, const proba_t *in2, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = in1[i] * in2[i];
    }
}

void div_vec(proba_t *out, const proba_t *in1, const proba_t *in2, uint32_t len){
    uint32_t i;
    proba_t a,b;
    for(i=0;i<len;i++){
        out[i] = in1[i] / in2[i];
    }
}
/*
 * substract in2 to in1 as well as the cst 
 */
void sub_vec(proba_t *out, const proba_t *in1, const proba_t *in2, const proba_t cst,uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        out[i] = in1[i] - in2[i] - (cst);
    }
}

/*
 * return the max of the distribution in of size len
 */
proba_t get_max(const proba_t *in,uint32_t len){
    proba_t max;
    uint32_t i;
    max = -DBL_MAX;
    for(i=0;i<len;i++){
        max = max>in[i] ? max:in[i];
    }
    return max;
}
