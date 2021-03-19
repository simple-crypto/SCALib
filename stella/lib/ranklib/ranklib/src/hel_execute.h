#ifndef HEL_EXECUTE_H
#define HEL_EXECUTE_H

#include "hel_enum.h"

int hel_execute_procedure(hel_param_t* param, hel_result_t* result);

hel_result_t* hel_execute_rank(int merge_value, int nb_bins, double** score_mat_init,int* key_init);

hel_result_t* hel_execute_enum(int merge_value, int nb_bins, double** score_mat_init, int* key_init, NTL::ZZ bound_start, NTL::ZZ bound_end,int test_key_bool, unsigned char** pt_ct, int enumerate_to_real_key_rank_bool, int up_to_bound_bool);

#endif
