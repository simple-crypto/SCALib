#ifndef HEL_ENUM_H
#define HEL_ENUM_H

#include "hel_histo.h"
#include "aes.h"


int*** convert_list_into_array( hel_int_list_t*** key_list, int nb_bins, int updated_nb_subkey);

int** get_binary_hist_from_big_hist(NTL::ZZX* hists, int* big_hist_size,int* convolution_order, int updated_nb_subkey);

int get_index_bin_bound(NTL::ZZX* hist, NTL::ZZ bound,NTL::ZZ* nb_total_elem,int boolean_end);

void get_enum_info( hel_enum_info_t* enum_info, hel_enum_input_t* enum_input , hel_preprocessing_t* preprocessing, hel_histo_t* histo,hel_real_key_info_t* real_key_info);

int set_enum_input(hel_enum_input_t* enum_input,  hel_enum_info_t* enum_info , hel_preprocessing_t* hel_preprocessing, hel_histo_t* histo);

int test_key_equality(unsigned char* ct1, unsigned char* ct2, int size);

int print_file_key_original(int** current_key_facto,int test_key_boolean, int updated_nb_subkey, int merge_value, unsigned char** pt_ct);

int print_file_key_up_to_key(int** current_key_facto,int enumerate_to_real_key_rank, int updated_nb_subkey, int merge_value, unsigned char** pt_ct, int* real_key);

void decompose_bin(int current_small_hist, int current_index, hel_preprocessing_t* preprocessing, hel_enum_input_t* enum_input, int* found_key);

void start_recursive_enumeration( hel_preprocessing_t* preprocessing, hel_histo_t* histo, hel_enum_input_t* enum_input, hel_enum_info_t* enum_info);

#endif
