#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "macro.h"
#include <assert.h>
void arange(uint32_t *tab,uint32_t begin, uint32_t end, uint32_t step){
    uint32_t val;
    for(val=begin;val<end;val+=step){
        *tab = val;
        tab++;
    }
}
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
        sum += *in;
        in++;
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
        *out = log10(*in);
        out++;in++;
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
    sanity_check(out,out,len);
}



/*
 * pre: in is a distribution of size len
 * post: the min of the distribution is set to val
 */
void tile(proba_t *out, const proba_t *in, proba_t val, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        *out = ((*in)<(val)) ? val:*in;
        out++;in++;
    }
}

/*
 * add a cst to the distribution out of size len
 */
void add_cst(proba_t * out,proba_t cst,uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        *out = (*out) +cst;
        out++;
    }
}

void add_cst_dest(proba_t *out, const proba_t *in, proba_t val, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        *out = (*in)+val;
        out++;in++;
    }
}
void abs_vec(proba_t *out, const proba_t *in, uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        *out = abs((*in));
        out++;in++;
    }
}
/*
 * div by a cst the distribution out of size len
 */
void div_cst(proba_t * out,const proba_t *in,proba_t cst,uint32_t len){
    uint32_t i;
    for(i=0;i<len;i++){
        *out = (*in)/cst;
        out++;in++;
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
        min = min<*in ? min:*in;
        in++;
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
        *out = *in1+ *in2;
        out++;in1++;in2++;
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
        *out = (*in1) - (*in2) - (cst);
        out++;in1++;in2++;
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
        max = max>*in ? max:*in;
        in++;
    }
    return max;
}

uint32_t cum_at_index(const proba_t *distr, uint32_t *indexes, const proba_t thr, uint32_t len){
    int32_t i;
    proba_t acc=0;
    for(i=len-1;i>0;i--){
        acc += distr[indexes[i]];
        if(acc > thr)
            return i+1;
    }
    return 1;
}


void sanity_check(proba_t *out,const proba_t *in,uint32_t len){
}
