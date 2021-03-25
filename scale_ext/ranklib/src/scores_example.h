#ifndef SCORES_EXAMPLE_H
#define SCORES_EXAMPLE_H

#include "hel_util.h"

extern double global_score_10[NB_SUBKEY_INIT][NB_KEY_VALUE_INIT];
extern double global_score_21[NB_SUBKEY_INIT][NB_KEY_VALUE_INIT];
extern double global_score_25[NB_SUBKEY_INIT][NB_KEY_VALUE_INIT];
extern double global_score_29[NB_SUBKEY_INIT][NB_KEY_VALUE_INIT];
extern double global_score_34[NB_SUBKEY_INIT][NB_KEY_VALUE_INIT];
extern double global_score_39[NB_SUBKEY_INIT][NB_KEY_VALUE_INIT];

double** get_scores_from_example(int rank);
void free_scores_example(double** scores);

#endif
