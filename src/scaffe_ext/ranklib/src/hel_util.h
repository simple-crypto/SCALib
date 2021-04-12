#ifndef HEL_UTIL_H
#define HEL_UTIL_H



#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <fstream>
#include <stdint.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "hel_struct.h"

#define NB_SUBKEY_INIT 16
#define NB_KEY_VALUE_INIT 256


hel_int_list_t* hel_int_list_init();
hel_int_list_t* hel_int_list_add(int val, hel_int_list_t* list);
void hel_free_int_list(hel_int_list_t* list);

double hel_min_max_mat(double** mat, int s1, int* s2, double* max_val);

double** merge_mat_score(double** score_mat_init, hel_preprocessing_t* preprocess);

void merge_key_score( hel_data_t* data, hel_preprocessing_t* preprocess, double** score_mat_init ,int* key_init);

#endif
