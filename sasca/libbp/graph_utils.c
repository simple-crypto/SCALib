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

int cmpfunc_inv(const void *a, const void *b, void * tab){
    uint32_t id_a,id_b;
    proba_t *proba_to_sort = (proba_t *) tab;

    id_a = (uint32_t) *(uint32_t *)a;
    id_b = (uint32_t) *(uint32_t *)b;

    //printf("%.20f and %.20f \n",proba_to_sort[id_a],proba_to_sort[id_b]);
    if(proba_to_sort[id_a]>proba_to_sort[id_b])
        return -1;
    else if(proba_to_sort[id_a] < proba_to_sort[id_b])
        return 1;
    else
        return 0;
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

void hard_thr(proba_t *distri,proba_t thr,uint32_t len){
    uint32_t *index,i,lim;
    index = (uint32_t *) malloc(sizeof(uint32_t)*len);
    arange(index,0,len,1);
    qsort_r(index,len,sizeof(uint32_t),cmpfunc_inv,(void*)distri);
    lim = cum_at_index(distri,index,thr,len);
    for(i=(lim+1);i<len;i++)
        distri[index[i]] = distri[index[lim]]*1E-20;
    free(index);
}

/*
 * pre: in is a distri of size len
 * post: out is the normalized distri
 *
 *  returns the normalization cst
 */
proba_t normalize_vec(proba_t *out, const proba_t *in,uint32_t len,uint32_t tile_flag){
    proba_t norm;
    uint32_t *index;
    int32_t i,lim;

    index = (uint32_t *) malloc(sizeof(uint32_t)*len);
    arange(index,0,len,1);
    norm = 0;
    qsort_r(index,len,sizeof(uint32_t),cmpfunc_inv,(void*)in);
    tile(out,in,TILE,len); 
    for(i=(len-1);i>=0;i--){
        norm += out[index[i]];
    }
    if(norm<=0){
        printf("Norm 0 %f \n",norm);
        for(i=(len-1);i>=0;i--)
            out[i] = 1.0/Nk;
    }
    else{
        for(i=(len-1);i>=0;i--){
            out[i] /=norm;
        }
    }
    
    free(index);
    if(tile_flag == 0){
	    return norm;
    }else{
        tile(out,out,TILE,len);
        return normalize_vec(out,out,len,0);
    }
}
