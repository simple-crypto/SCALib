#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "macro.h"
#include "utils.h"
#include "graph.h"
#include "graph_utils.h"
#include <assert.h>

extern uint32_t Nk;
void xor_ex(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO){
    uint32_t i0,i1,o;
    for(i0=0;i0<Nk;i0++){
        for(i1=0;i1<Nk;i1++){
            o = i1 ^ i0;
            msg[index(0,o,Nk)] += distri0[i0] * distri1[i1];
            msg[index(1,i0,Nk)]+= distri1[i1] * distriO[o];
            msg[index(2,i1,Nk)]+= distri0[i0] * distriO[o];
        }
    }
}
void and_ex(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO){
    uint32_t i0,i1,o;
    for(i0=0;i0<Nk;i0++){
        for(i1=0;i1<Nk;i1++){
            o = i1 & i0;
            msg[index(0,o,Nk)] += distri0[i0] * distri1[i1];
            msg[index(1,i0,Nk)]+= distri1[i1] * distriO[o];
            msg[index(2,i1,Nk)]+= distri0[i0] * distriO[o];
        }
    }
}

/*
 *
 * XOR GATE
 *
 */

/*
 * Compute the walch hadamgard distribution of a distribution
 * pre: a is the distributioa

 *      len of the distribution
 * post: a is the wht of the input
 */
void fwht(proba_t *a,const uint32_t len){
    uint32_t h,i,j,step;
    proba_t x,y;
    h = 1;
    while(h<len){
        step = h*2;
        for(i=0;i<len;i+=step){
            for(j=i;j<(i+h);j++){
                x = a[j];
                y = a[j+h];
                a[j] = x+y;
                a[j+h] = x-y;
            }
        }
        h = step;
    }
}


void xor_fwht(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO){
    // running fwht on all the distri
    fwht(distri0,Nk);
    fwht(distri1,Nk);
    fwht(distriO,Nk);

    // computing the conv in freq domain
    mult_vec(&msg[index(0,0,Nk)],distri0,distri1,Nk);
    mult_vec(&msg[index(1,0,Nk)],distri1,distriO,Nk);
    mult_vec(&msg[index(2,0,Nk)],distri0,distriO,Nk);

    // back to distri domain + constant scaling (in the next scaling)
    fwht(&msg[index(0,0,Nk)],Nk);
    fwht(&msg[index(1,0,Nk)],Nk);
    fwht(&msg[index(2,0,Nk)],Nk);
}


/*
 * pre: in is a distri of size len
 * post: out is the normalized distri
 *
 *  returns the normalization cst
 */
proba_t normalize_vec(proba_t *out, const proba_t *in,uint32_t len,uint32_t tile_flag){
    proba_t norm;
    int32_t i;
    norm = 0;
    for(i=0;i<len;i++){
        norm += in[i];
    }
    for(i=0;i<len;i++){
        out[i] = (in[i]) / norm;
    }

    if(tile_flag == 0){
	    return norm;
    }else{
        tile(out,out,TILE,len);
        return normalize_vec(out,out,len,0);
    }
}
