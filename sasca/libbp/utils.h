#include <stdint.h>
#include <float.h>
#include "graph.h"

void sanity_check(proba_t *out,const proba_t *in,uint32_t len);
proba_t sum(const proba_t *in,uint32_t len);
void apply_P10(proba_t * out,const proba_t *in,uint32_t len);
void tile(proba_t *out, const proba_t *in, proba_t val, uint32_t len);
void add_cst(proba_t * out,proba_t cst,uint32_t len);
void div_cst(proba_t * out,const proba_t *in,proba_t cst,uint32_t len);
void add_vec(proba_t *out, const proba_t *in1, const proba_t *in2, uint32_t len);
void mult_vec(proba_t *out, const proba_t *in1, const proba_t *in2, uint32_t len);
void div_vec(proba_t *out, const proba_t *in1, const proba_t *in2, uint32_t len);
void sub_vec(proba_t *out, const proba_t *in1, const proba_t *in2, const proba_t cst,uint32_t len);


void arange(uint32_t *tab,uint32_t begin, uint32_t end, uint32_t step);
void add_cst_dest(proba_t *out, const proba_t *in, proba_t val, uint32_t len);
void apply_log10(proba_t * out,const proba_t *in,uint32_t len);

uint32_t cum_at_index(const proba_t *distr, uint32_t *indexes, const proba_t thr, uint32_t len);
proba_t get_max(const proba_t *in,uint32_t len);
proba_t get_min(const proba_t *in,uint32_t len);
void abs_vec(proba_t *out, const proba_t *in, uint32_t len);
