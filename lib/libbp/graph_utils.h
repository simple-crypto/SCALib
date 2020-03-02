#include "graph.h"
#include <stdint.h>

void and_ex(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO);
void and_indep(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO);
proba_t normalize_vec(proba_t *out, const proba_t *in,uint32_t len,uint32_t tile_flag);
void init_table();
void free_table();
void and_fast(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO,
        uint32_t *index0, uint32_t *index1, uint32_t *indexO);
void xor_fwht(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO);
void and_fast_partial(proba_t *msg,proba_t *distri0,proba_t *distri1,proba_t *distriO,
        uint32_t *index0, uint32_t *index1, uint32_t *indexO);
void delta_node(proba_t *msg,proba_t *distri0,proba_t *distri1, proba_t *distriO);
void hard_thr(proba_t *distri,proba_t thr,uint32_t len);
