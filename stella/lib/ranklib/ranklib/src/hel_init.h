#ifndef HEL_INIT_H
#define HEL_INIT_H

#include "hel_util.h"


extern "C" {

//preprocessing
hel_preprocessing_t* hel_alloc_preprocessing();
int hel_init_preprocessing( hel_preprocessing_t* preprocessing, int merge_value);
void hel_free_preprocessing( hel_preprocessing_t* preprocessing );


//histogram
hel_histo_t* hel_alloc_histo();
int hel_init_histo(hel_histo_t* histo, int nb_bins);
void hel_free_histo(hel_histo_t* histo);


//data
hel_data_t* hel_alloc_data();
int hel_init_data(hel_data_t* data, hel_preprocessing_t* preprocessing, double** score_mat_init, int* key);
void hel_free_data( hel_data_t* data, hel_preprocessing_t* preprocessing);


//enum param
hel_enum_input_t* hel_alloc_enum_input();
int hel_init_enum_input(hel_enum_input_t* enum_input, hel_preprocessing_t* preprocessing, hel_data_t* data, hel_histo_t* histo , NTL::ZZ bound_start, NTL::ZZ bound_end, int test_key_boolean ,unsigned char** pt_ct, int enumerate_to_real_key_rank, int up_to_bound_bool);
void hel_free_enum_input(hel_enum_input_t* enum_input, hel_histo_t* histo, hel_preprocessing_t* preprocessing);




//param
hel_param_t* hel_alloc_param();
int hel_init_param(hel_param_t* param, hel_algo_mode_t algo_mode, int merge_value, int nb_bins, double** score_mat_init, int* key_init, NTL::ZZ bound_start, NTL::ZZ bound_end , int test_key_boolean, unsigned char** pt_ct,int enumerate_to_real_key_rank, int up_to_bound_bool);
void hel_free_param(hel_param_t* param);


//result
hel_real_key_info_t* hel_alloc_real_key_info();
int hel_init_real_key_info(hel_real_key_info_t* real_key_info);
void hel_free_real_key_info( hel_real_key_info_t* real_key_info);

hel_enum_info_t* hel_alloc_enum_info();
int hel_init_enum_info(hel_enum_info_t* enum_info);
void hel_free_enum_info( hel_enum_info_t* enum_info);

hel_result_t* hel_alloc_result();
int hel_init_result(hel_result_t* result);
void hel_free_result(hel_result_t* result);


//accessor
NTL::ZZ hel_result_get_estimation_rank(hel_result_t* result);
NTL::ZZ hel_result_get_estimation_rank_min(hel_result_t* result);
NTL::ZZ hel_result_get_estimation_rank_max(hel_result_t* result);
double hel_result_get_estimation_time(hel_result_t* result);
NTL::ZZ hel_result_get_enum_rank(hel_result_t* result);
NTL::ZZ hel_result_get_enum_rank_min(hel_result_t* result);
NTL::ZZ hel_result_get_enum_rank_max(hel_result_t* result);
double hel_result_get_enum_time_preprocessing(hel_result_t* result);
double hel_result_get_enum_time(hel_result_t* result);
int hel_result_is_key_found(hel_result_t* result);

}

#endif
