#include "hel_init.h"

using namespace std;
using namespace NTL;

//preprocessing
hel_preprocessing_t* hel_alloc_preprocessing(){

	hel_preprocessing_t* preprocessing = NULL;
	preprocessing = (hel_preprocessing_t*) malloc( sizeof(hel_preprocessing_t));
	preprocessing->updated_nb_key_value = NULL;


	return preprocessing;
}



int hel_init_preprocessing( hel_preprocessing_t* preprocessing, int merge_value){

	int error = 0;

	if ( (merge_value < 1) || (merge_value > 3) ){
		error = 1;
		fprintf(stderr,"Error in preprocessing initialization, merge value not supported (%d)\n",merge_value);
	}

	else{
		preprocessing->merge_value = merge_value;
	}

	return error;

}


void hel_free_preprocessing( hel_preprocessing_t* preprocessing ){

	if (preprocessing != NULL){
		free(preprocessing->updated_nb_key_value);
		free(preprocessing);
	}

}








//histogram
hel_histo_t* hel_alloc_histo(){

	hel_histo_t* histo = NULL;
	histo = (hel_histo_t*) malloc(sizeof(hel_histo_t));
	histo->hists = NULL;
	histo->hists_alloc_bool = 0;
	histo->width = new RR[1];

	return histo;

}

int hel_init_histo(hel_histo_t* histo, int nb_bins){

	int error = 0;

	if (nb_bins < 1){
		error = 1;
		fprintf(stderr,"Error in histogram initialization, incorrect bin number (%d)\n",nb_bins);
	}
	else{
		histo->nb_bins = nb_bins;
	}

	return error;

}


void hel_free_histo( hel_histo_t* histo){

	if (histo != NULL){
		delete[] histo->width;
		if(histo->hists_alloc_bool){
			delete[] histo->hists;
		}

		free(histo);
	}
}





//data
hel_data_t* hel_alloc_data(){

	hel_data_t* data = NULL;
	data = (hel_data_t*) malloc(sizeof(hel_data_t));

	data->log_probas = NULL;
	data->real_key = NULL;
	data->log_probas_real_key = NULL;

	data->log_probas_alloc_bool = 0;
	data->real_key_alloc_bool = 0;
	data->log_probas_real_key_alloc_bool = 0;

	return data;
}


int hel_init_data(hel_data_t* data, hel_preprocessing_t* preprocessing, double** score_mat_init, int* key_init){

	int error = 0;

	if ( score_mat_init == NULL){
		error  = 1;
		fprintf(stderr, "Error, log_probas not initialized\n");
	}
	else{

		data->log_probas = merge_mat_score( score_mat_init, preprocessing);

		data->log_probas_alloc_bool = 1;
	}

	if ( key_init != NULL){


		merge_key_score(data,  preprocessing, score_mat_init,key_init);
		data->log_probas_real_key_alloc_bool = 1;
		data->real_key_alloc_bool = 1;
	}


	return error;

}


void hel_free_data( hel_data_t* data, hel_preprocessing_t* preprocessing){

	int i;

	if (data != NULL){
		if ( data->log_probas_alloc_bool){
			for (i = 0; i < preprocessing->updated_nb_subkey ; i++){
				free(data->log_probas[i]);
			}
			free(data->log_probas);
		}

		if (data->real_key_alloc_bool){
			free(data->real_key);
		}

		if (data->log_probas_real_key_alloc_bool){
			free(data->log_probas_real_key);
		}


		free(data);
	}
}




//enum param

hel_enum_input_t* hel_alloc_enum_input(){

	hel_enum_input_t* enum_input = NULL;

	enum_input = (hel_enum_input_t*) malloc(sizeof(hel_enum_input_t));

	enum_input->binary_hists = NULL;
	enum_input->binary_hists_size = NULL;
	enum_input->key_list_bin = NULL;
	enum_input->key_list_bin2 = NULL;
	enum_input->index_list = NULL;
	enum_input->convolution_order = NULL;
	enum_input->key_factorization = NULL;
	enum_input->pt_ct = NULL;
	enum_input->real_key = NULL;

	enum_input->binary_hists_alloc_bool = 0;
	enum_input->binary_hists_size_alloc_bool = 0;
	enum_input->key_list_bin_alloc_bool = 0;
	enum_input->key_list_bin2_alloc_bool = 0;
	enum_input->index_list_alloc_bool = 0;
	enum_input->convolution_order_alloc_bool = 0;
	enum_input->key_factorization_alloc_bool = 0;
	enum_input->pt_ct_alloc_bool = 0;
	enum_input->real_key_alloc_bool = 0;

	enum_input->bound_start = new ZZ[1];
	enum_input->bound_end = new ZZ[1];

	return enum_input;

}

int hel_init_enum_input(hel_enum_input_t* enum_input, hel_preprocessing_t* preprocessing, hel_data_t* data, hel_histo_t* histo , ZZ bound_start, ZZ bound_end, int test_key_boolean ,unsigned char** pt_ct, int enumerate_to_real_key_rank, int up_to_bound_bool){

	int i,j;

	int error = 0;


	enum_input->key_list_bin2 = (hel_int_list_t***) malloc(preprocessing->updated_nb_subkey*sizeof(hel_int_list_t**));
	for (i = 0;  i < preprocessing->updated_nb_subkey ; i++){
		enum_input->key_list_bin2[i] = (hel_int_list_t**) malloc( histo->nb_bins*sizeof(hel_int_list_t*));
	}
	enum_input->key_list_bin2_alloc_bool = 1;



	enum_input->index_list = (int**) malloc(preprocessing->updated_nb_subkey*sizeof(int*));
	for (i = 0;  i < preprocessing->updated_nb_subkey ; i++){
		enum_input->index_list[i] = (int*) malloc((histo->nb_bins+1)*sizeof(int));
	}
	enum_input->index_list_alloc_bool = 1;

	enum_input->convolution_order = (int*) malloc(preprocessing->updated_nb_subkey* sizeof(int));
	enum_input->convolution_order_alloc_bool = 1;


	enum_input->key_factorization = (int**) malloc(preprocessing->updated_nb_subkey*sizeof(int*));
	for (i = 0;i  < preprocessing->updated_nb_subkey ; i++){
		enum_input-> key_factorization[i] = (int*) malloc( (preprocessing->updated_nb_key_value[i]+1)*sizeof(int));
	}
	enum_input->key_factorization_alloc_bool = 1;

	*(enum_input->bound_start) = bound_start;
	*(enum_input->bound_end) = bound_end;


	enum_input->test_key_boolean = test_key_boolean;

	if ( test_key_boolean && ( pt_ct == NULL) ){
		fprintf(stderr,"Error: user asked for test on the fly but no pt/ct are provided\n");
	}

	else{

		enum_input->pt_ct = (unsigned char**) malloc(4*sizeof(unsigned char*));
		for (i = 0 ; i < 4 ; i++){
			enum_input->pt_ct[i] = (unsigned char*) malloc( NB_SUBKEY_INIT*sizeof(unsigned char));
			for (j = 0 ; j < NB_SUBKEY_INIT ; j++){
				enum_input->pt_ct[i][j] = pt_ct[i][j];
			}
		}
		enum_input->pt_ct_alloc_bool = 1;

	}


	if (  !data->log_probas_real_key_alloc_bool && enumerate_to_real_key_rank ){
		error = 1;

		fprintf(stderr,"Error, user asked to enumerate up to the real key, but no real key is specified");


	}
	else{
		enum_input->enumerate_to_real_key_rank = enumerate_to_real_key_rank;
		enum_input->real_key = (int*) malloc(preprocessing->updated_nb_subkey*sizeof(int));
		for (i = 0; i<  preprocessing->updated_nb_subkey ; i++){
			enum_input->real_key[i] = data->real_key[i];
		}
		enum_input->real_key_alloc_bool = 1;
	}

	enum_input->up_to_bound_bool = up_to_bound_bool;
	if (enumerate_to_real_key_rank && up_to_bound_bool){
		fprintf(stdout,"Warning, up_to_bound_bool is activated without effect since enumerate_to_real_key is activated\n");
	}

	return error;

}


void hel_free_enum_input(hel_enum_input_t* enum_input, hel_histo_t* histo, hel_preprocessing_t* preprocessing){

	int i,j;

	if (enum_input != NULL){


		if (enum_input->binary_hists_alloc_bool){
			for (i = 0; i < preprocessing->updated_nb_subkey ; i++){
				free(enum_input->binary_hists[i]);
			}
			free(enum_input->binary_hists);
		}

		if(enum_input->binary_hists_size_alloc_bool){
			free(enum_input->binary_hists_size);
		}

		if(enum_input->key_list_bin_alloc_bool){
			for (i = 0 ; i < preprocessing->updated_nb_subkey ; i++){

				for (j = 1; j < enum_input->index_list[i][0] + 1; j++){

					free( enum_input->key_list_bin[i][enum_input->index_list[i][j]]);
				}

				free(enum_input->key_list_bin[i]);
			}

			free(enum_input->key_list_bin);

		}

		if(enum_input->key_list_bin2_alloc_bool){
			for (i = 0 ; i < preprocessing->updated_nb_subkey ; i++){

				for (j = 0; j < histo->nb_bins ; j++){

					hel_free_int_list( enum_input->key_list_bin2[i][j]);
				}

				free(enum_input->key_list_bin2[i]);
			}

			free(enum_input->key_list_bin2);
		}

		if(enum_input->index_list_alloc_bool){
			for (i = 0 ; i < preprocessing->updated_nb_subkey ; i++){
				free(enum_input->index_list[i]);
			}
			free(enum_input->index_list);
		}

		if (enum_input->convolution_order_alloc_bool){
			free(enum_input->convolution_order);
		}

		if (enum_input->key_factorization_alloc_bool){
			for (i = 0 ; i < preprocessing->updated_nb_subkey ; i++){
				free(enum_input->key_factorization[i]);
			}
			free(enum_input->key_factorization);
		}


		delete[] enum_input->bound_start;
		delete[] enum_input->bound_end;

		if (enum_input->pt_ct_alloc_bool){
			for (i = 0 ; i < 4 ; i++){
				free(enum_input->pt_ct[i]);
			}
			free(enum_input->pt_ct);
		}

		if (enum_input->real_key_alloc_bool){
			free(enum_input->real_key);
		}

		free(enum_input);
	}

}









//param struct

hel_param_t* hel_alloc_param(){
	hel_param_t* param = NULL;

	param = (hel_param_t*) malloc(sizeof(hel_param_t));

	param->preprocessing = hel_alloc_preprocessing();
	param->histo = hel_alloc_histo();
	param->data = hel_alloc_data();
	param->enum_input = hel_alloc_enum_input();


	return param;
}



int hel_init_param(hel_param_t* param, hel_algo_mode_t algo_mode, int merge_value, int nb_bins, double** score_mat_init, int* key_init, ZZ bound_start, ZZ bound_end , int test_key_boolean, unsigned char** pt_ct,int enumerate_to_real_key_rank, int up_to_bound_bool){

	int error = 0;
	param->algo_mode = algo_mode;


	error = hel_init_preprocessing(  param->preprocessing,  merge_value);

	error |= hel_init_histo(param->histo,  nb_bins);

	if(!error){

		error |= hel_init_data( param->data,  param->preprocessing ,  score_mat_init,  key_init);

		if(!error && (algo_mode == ENUM) ){

			error = hel_init_enum_input( param->enum_input, param->preprocessing, param->data , param->histo, bound_start, bound_end, test_key_boolean, pt_ct, enumerate_to_real_key_rank, up_to_bound_bool);

		}
	}

	return error;

}

void hel_free_param(hel_param_t* param){

	if(param != NULL){
		hel_free_enum_input(param->enum_input,param->histo,param->preprocessing);
		hel_free_data(param->data,param->preprocessing);
		hel_free_histo(param->histo);
		hel_free_preprocessing(param->preprocessing);
		free(param);
	}

}




//result
hel_real_key_info_t* hel_alloc_real_key_info(){

	hel_real_key_info_t* real_key_info = (hel_real_key_info_t*) malloc(sizeof(hel_real_key_info_t));

	real_key_info->bound_real_key = new ZZ[1];
	real_key_info->bound_min = new ZZ[1];
	real_key_info->bound_max = new ZZ[1];

	return real_key_info;
}

int hel_init_real_key_info(hel_real_key_info_t* real_key_info){

	int error = 0;

	real_key_info->bin_real_key = -1;
	*(real_key_info->bound_real_key) = -1;

	real_key_info->bin_bound_min = -1;
	*(real_key_info->bound_min) = -1;

	real_key_info->bin_bound_max = -1;
	*(real_key_info->bound_max) = -1;

	real_key_info->rank_estimation_time = -1.;

	return error;
}

void hel_free_real_key_info( hel_real_key_info_t* real_key_info){

	if (real_key_info != NULL){
		delete[] real_key_info->bound_real_key;
		delete[] real_key_info->bound_min;
		delete[] real_key_info->bound_max;

		free(real_key_info);
	}

}





hel_enum_info_t* hel_alloc_enum_info(){

	hel_enum_info_t* enum_info = (hel_enum_info_t*) malloc(sizeof(hel_enum_info_t));

	enum_info->nb_key_enum_start = new ZZ[1];
	enum_info->nb_key_enum_end = new ZZ[1];
	enum_info->nb_key_enum_found_key = new ZZ[1];
	enum_info->nb_key_enum_found_key_bound_min = new ZZ[1];
	enum_info->nb_key_enum_found_key_bound_max = new ZZ[1];
	enum_info->bound_nb_key_enum_start = new ZZ[1];
	enum_info->bound_nb_key_enum_end = new ZZ[1];

    enum_info->found_key_boolean = 0;

	return enum_info;
}

int hel_init_enum_info(hel_enum_info_t* enum_info){

	int error = 0;

	enum_info->bin_enum_start = -1;
	*(enum_info->nb_key_enum_start) = -1;

	enum_info->bin_enum_end = -1;
	*(enum_info->nb_key_enum_end) = -1;

	enum_info->bin_found_key = -1;
	*(enum_info->nb_key_enum_found_key) = -1;

	enum_info->bin_found_key_bound_min = -1;
	*(enum_info->nb_key_enum_found_key_bound_min) = -1;

	enum_info->bin_found_key_bound_max = -1;
	*(enum_info->nb_key_enum_found_key_bound_max) = -1;

	enum_info->bound_bin_enum_start = -1;
	*(enum_info->bound_nb_key_enum_start) = -1;

	enum_info->bound_bin_enum_end = -1;
	*(enum_info->bound_nb_key_enum_end) = -1;

	enum_info->preprocessing_time = -1.;
	enum_info->enum_time = -1.;

	return error;
}


void hel_free_enum_info( hel_enum_info_t* enum_info){

	if ( enum_info != NULL){
		delete[] enum_info->nb_key_enum_start;
		delete[] enum_info->nb_key_enum_end;
		delete[] enum_info->nb_key_enum_found_key;
		delete[] enum_info->nb_key_enum_found_key_bound_min;
		delete[] enum_info->nb_key_enum_found_key_bound_max;
		delete[] enum_info->bound_nb_key_enum_start;
		delete[] enum_info->bound_nb_key_enum_end;

		free(enum_info);
	}

}




hel_result_t* hel_alloc_result(){

	hel_result_t* result = NULL;

	result = (hel_result_t*) malloc(sizeof(hel_result_t));

	result->real_key_info = hel_alloc_real_key_info();
	result->enum_info = hel_alloc_enum_info();

	return result;

}



int hel_init_result(hel_result_t* result){

	int error = 0;

	hel_init_real_key_info(result->real_key_info);
	hel_init_enum_info(result->enum_info);

	return error;
}

void hel_free_result(hel_result_t* result){

	if (result != NULL){

		hel_free_real_key_info(result->real_key_info);
		hel_free_enum_info(result->enum_info);
		free(result);

	}

}




//results accessor

//rank estim
ZZ hel_result_get_estimation_rank(hel_result_t* result){

	if ( *(result->real_key_info->bound_real_key) == conv<ZZ>(-1) ){
		fprintf(stderr,"Warning: this result was not launched with rank estimation\n");
	}
	return *(result->real_key_info->bound_real_key);

}

ZZ hel_result_get_estimation_rank_min(hel_result_t* result){

	if ( *(result->real_key_info->bound_min) == conv<ZZ>(-1) ){
		fprintf(stderr,"Warning: this result was not launched with rank estimation\n");
	}
	return *(result->real_key_info->bound_min);

}

ZZ hel_result_get_estimation_rank_max(hel_result_t* result){

	if ( *(result->real_key_info->bound_max) == conv<ZZ>(-1) ){
		fprintf(stderr,"Warning: this result was not launched with rank estimation\n");
	}
	return *(result->real_key_info->bound_max);
}

double hel_result_get_estimation_time(hel_result_t* result){

	if ( result->real_key_info->rank_estimation_time == -1. ){
		fprintf(stderr,"Warning: this result was not launched with rank estimation\n");
	}
	return result->real_key_info->rank_estimation_time;

}


//enum
ZZ hel_result_get_enum_rank(hel_result_t* result){

	if ( *(result->enum_info->nb_key_enum_found_key) == conv<ZZ>(-1) ){
		fprintf(stderr,"Warning: this result was not launched with key enumeration or the key has not been found\n");
	}
	return *(result->enum_info->nb_key_enum_found_key);
}

ZZ hel_result_get_enum_rank_min(hel_result_t* result){

	if ( *(result->enum_info->nb_key_enum_found_key_bound_min) == conv<ZZ>(-1) ){
		fprintf(stderr,"Warning: this result was not launched with key enumeration or the key has not been found\n");
	}
	return *(result->enum_info->nb_key_enum_found_key_bound_min);
}

ZZ hel_result_get_enum_rank_max(hel_result_t* result){

	if ( *(result->enum_info->nb_key_enum_found_key_bound_max) == conv<ZZ>(-1) ){
		fprintf(stderr,"Warning: this result was not launched with key enumeration or the key has not been found\n");
	}
	return *(result->enum_info->nb_key_enum_found_key_bound_max);
}

double hel_result_get_enum_time_preprocessing(hel_result_t* result){

	if ( result->enum_info->preprocessing_time == -1. ){
		fprintf(stderr,"Warning: this result was not launched with key enumeration or the key has not been found\n");
	}
	return result->enum_info->preprocessing_time;
}

double hel_result_get_enum_time(hel_result_t* result){

	if ( result->enum_info->enum_time == -1. ){
		fprintf(stderr,"Warning: this result was not launched with key enumeration or the key has not been found\n");
	}
	return result->enum_info->enum_time;
}

int hel_result_is_key_found(hel_result_t* result){

	return result->enum_info->found_key_boolean;
}


