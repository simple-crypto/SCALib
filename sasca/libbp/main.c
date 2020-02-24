#include "libbp.h"
#include "utils.h"
#include "graph_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#define LEN 650771884 //620771884
#define NVNODES 271
#define NFNODES 246
int main(){
    FILE *fp;

    uint32_t *buff;
    double *out;
    buff = (uint32_t *) malloc(LEN);
    if(buff == NULL)
        exit(EXIT_FAILURE);
    out = (double*) malloc(sizeof(double)*65536*NVNODES);

    fp = fopen("save_buff.dat","rb");
    fread(buff,LEN,1,fp);

    func(buff,NVNODES,NFNODES,(uint32_t)1,(uint32_t)1,out);

    fclose(fp);
    free(buff);
    free(out);
}
