#ifndef HEL_STRUCT_H
#define HEL_STRUCT_H

#include <NTL/ZZXFactoring.h>
#include <NTL/RR.h>

extern "C" {

struct hel_int_list_t{

	int val;
	struct hel_int_list_t* next;

};

typedef struct hel_int_list_t hel_int_list_t;



typedef enum{

	RANK,
	ENUM

}hel_algo_mode_t;

typedef struct{

	int updated_nb_subkey; //number of subkeys lists after merging
	int* updated_nb_key_value; //number of subkeys value per lists after merging
 	int merge_value;

}hel_preprocessing_t;



typedef struct{

	int nb_bins;
	NTL::RR* width;
	NTL::ZZX* hists;

	int hists_alloc_bool;

}hel_histo_t;

typedef struct{

	double** log_probas;
	int* real_key;
	double* log_probas_real_key;
	double shift;

	int log_probas_alloc_bool;
	int real_key_alloc_bool;
	int log_probas_real_key_alloc_bool;

}hel_data_t;

typedef struct{

	int** binary_hists;
	int* binary_hists_size;
	int*** key_list_bin;
	hel_int_list_t*** key_list_bin2;
	int** index_list;
	int* convolution_order;
	int** key_factorization;
	int bin_to_start;
	int bin_to_end;
	NTL::ZZ* bound_start;
	NTL::ZZ* bound_end;
	int test_key_boolean;
	unsigned char** pt_ct;
	int enumerate_to_real_key_rank; //if set to 1, enumeration is launched only if this rank is less than the one specified in "bound"
	int* real_key;
	int up_to_bound_bool;

	int binary_hists_alloc_bool;
	int binary_hists_size_alloc_bool;
	int key_list_bin_alloc_bool;
	int key_list_bin2_alloc_bool;
	int index_list_alloc_bool;
	int convolution_order_alloc_bool;
	int key_factorization_alloc_bool;
	int pt_ct_alloc_bool;
	int real_key_alloc_bool;

}hel_enum_input_t;


typedef struct{

	int bin_real_key;  //bin where the real key is found
	NTL::ZZ* bound_real_key; //associated value of key number

	int bin_bound_min; //bin of the min bound of the real key
	NTL::ZZ* bound_min; //associated value of key number

	int bin_bound_max; //bin of the max bound of the real key
	NTL::ZZ* bound_max; //associated value of key number

	double rank_estimation_time;

}hel_real_key_info_t;


typedef struct{

	int bin_enum_start; //bin where the enumeration starts according to the user's provided bound
	NTL::ZZ* nb_key_enum_start; //associated sumed number of key from the last bin to this bin

	int bin_enum_end; //bin where the enumeration ends according to the user's provided bound
	NTL::ZZ* nb_key_enum_end; //associated value of key number

	int bin_found_key; //bin where the real key is found (if found)
	NTL::ZZ* nb_key_enum_found_key; //associated sumed number of key from the last bin to this bin

	int bin_found_key_bound_min; //bin associated to the min bound of the real key (if found)
	NTL::ZZ* nb_key_enum_found_key_bound_min; //associated sumed number of key from the last bin to this bin

	int bin_found_key_bound_max; //bin associated to the max bound of the real key (if found)
	NTL::ZZ* nb_key_enum_found_key_bound_max; //associated sumed number of key from the last bin to this bin

	int bound_bin_enum_start; //bin associated to the min bound where the enumeration starts according to the user's provided bound
	NTL::ZZ* bound_nb_key_enum_start; //associated sumed number of key from the last bin to this bin

	int bound_bin_enum_end; //bin associated to the max bound where the enumeration ends according to the user's provided bound
	NTL::ZZ* bound_nb_key_enum_end; //associated sumed number of key from the last bin to this bin

	double preprocessing_time;
	double enum_time;

    int found_key_boolean;


}hel_enum_info_t;

typedef struct{

	hel_enum_info_t* enum_info;
	hel_real_key_info_t* real_key_info;

}hel_result_t;


typedef struct{

	hel_algo_mode_t algo_mode;
	hel_preprocessing_t* preprocessing;
	hel_histo_t* histo;
	hel_data_t* data;
	hel_enum_input_t* enum_input;

}hel_param_t;

}

#endif
