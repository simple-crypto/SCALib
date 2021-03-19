#include "hel_util.h"

using namespace std;
using namespace NTL;



hel_int_list_t* hel_int_list_init(){ //init to 0 val (we need that !)

	hel_int_list_t* root = (hel_int_list_t*) malloc( sizeof(hel_int_list_t));
	root->val = 0;
	root->next = NULL;

	return root;

}

hel_int_list_t* hel_int_list_add(int val, hel_int_list_t* list){

	hel_int_list_t* to_add =(hel_int_list_t*) malloc( sizeof(hel_int_list_t));
	to_add->val = val;
	to_add->next = NULL;

	list->next = to_add;

	return to_add;

}


void hel_free_int_list(hel_int_list_t* list){

	hel_int_list_t* next;
	
	while(  list != NULL){
		next = list->next;
		free(list);
		list = next;

	}

}



double hel_min_max_mat(double** mat, int s1, int* s2, double* max_val){

	int i,j;
	double res = mat[0][0];
	*max_val = mat[0][0];

	for (i = 0 ; i < s1 ; i++){
		for (j = 0; j < s2[i] ; j++){
			if (mat[i][j] < res){
				res = mat[i][j];
			}
			if (mat[i][j] > *max_val){
				*max_val = mat[i][j];
			}
		}
	}

	return res;
}



double** merge_mat_score(double** score_mat_init, hel_preprocessing_t* preprocess){

	double** res;
	int i,j,k;


	switch(preprocess->merge_value){

		case 1: // no merge

			preprocess->updated_nb_subkey = NB_SUBKEY_INIT;
			preprocess->updated_nb_key_value = (int*) malloc(preprocess->updated_nb_subkey*sizeof(int));

			res = (double**) malloc (preprocess->updated_nb_subkey* sizeof(double*));
			for (i = 0 ; i < preprocess->updated_nb_subkey ; i++){
				preprocess->updated_nb_key_value[i] = NB_KEY_VALUE_INIT;
				res[i] = (double*) malloc( preprocess->updated_nb_key_value[i] * sizeof(double));
			}

			for (i = 0; i < preprocess->updated_nb_subkey ; i++){
				for (j = 0; j < preprocess->updated_nb_key_value[i] ; j++){
					res[i][j] = score_mat_init[i][j];
				}
			}
			break;

		case 2:

			if ( NB_SUBKEY_INIT%2 ){ //case where the number of lists cannot be divided by 2 (i.e. NB_SUBKEY_INIT is odd)
			//we know that the last list will contain only NB_KEY_VALUE_INIT elements


				preprocess->updated_nb_subkey = NB_SUBKEY_INIT/2+1;
				preprocess->updated_nb_key_value = (int*) malloc(preprocess->updated_nb_subkey*sizeof(int));

				res = (double**) malloc( preprocess->updated_nb_subkey* sizeof(double*));
				for (i = 0 ; i < preprocess->updated_nb_subkey-1 ; i++){
					preprocess->updated_nb_key_value[i] = NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT;
					res[i] = (double*) malloc(preprocess->updated_nb_key_value[i] * sizeof(double));
				}
				preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1] = NB_KEY_VALUE_INIT;
				res[preprocess->updated_nb_subkey-1] = (double*) malloc(preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1] * sizeof(double));

				for (i = 0; i < NB_SUBKEY_INIT-1 ; i+=2){
					for (j = 0; j < preprocess->updated_nb_key_value[i/2] ; j++){
						res[i/2][j] = score_mat_init[i][j/NB_KEY_VALUE_INIT]+ score_mat_init[i+1][j%NB_KEY_VALUE_INIT];
					}
				}
				for (j = 0; j  < preprocess->updated_nb_key_value[NB_SUBKEY_INIT/2] ; j++){
					res[NB_SUBKEY_INIT/2][j] = score_mat_init[NB_SUBKEY_INIT-1][j];
				}
			}

			else{ // case where it can be divided by 2
				preprocess->updated_nb_subkey = NB_SUBKEY_INIT/2;
				preprocess->updated_nb_key_value = (int*) malloc(preprocess->updated_nb_subkey*sizeof(int));

				res = (double**) malloc( preprocess->updated_nb_subkey* sizeof(double*));
				for (i = 0 ; i < preprocess->updated_nb_subkey ; i++){
					preprocess->updated_nb_key_value[i] = NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT;
					res[i] = (double*) malloc(preprocess->updated_nb_key_value[i] * sizeof(double));
				}

				for (i = 0; i < NB_SUBKEY_INIT ; i+=2){
					for (j = 0; j < preprocess->updated_nb_key_value[i/2] ; j++){
						res[i/2][j] = score_mat_init[i][j/NB_KEY_VALUE_INIT]+ score_mat_init[i+1][j%NB_KEY_VALUE_INIT];
					}
				}
			}


			break;

		case 3:

			if ( NB_SUBKEY_INIT%3){
				preprocess->updated_nb_subkey = NB_SUBKEY_INIT/3+1; //assume 16 sbox at the origin... so, 5x2^24 + 1x2^8
				preprocess->updated_nb_key_value = (int*) malloc(preprocess->updated_nb_subkey*sizeof(int));

				res = (double**) malloc( preprocess->updated_nb_subkey* sizeof(double*));
				for (i = 0 ; i < preprocess->updated_nb_subkey-1 ; i++){
					preprocess->updated_nb_key_value[i] = NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT;
					res[i] = (double*) malloc( preprocess->updated_nb_key_value[i] * sizeof(double));
				}
				if ( (NB_SUBKEY_INIT%3) == 1 ){
					preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1]	= NB_KEY_VALUE_INIT;
					res[preprocess->updated_nb_subkey-1] = (double*)  malloc( preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1]*sizeof(double) );
				}
				else{ // then equal to 2 mod 3
					preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1]	= NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT;
					res[preprocess->updated_nb_subkey-1] = (double*)  malloc( preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1]*sizeof(double) );
				}

				for (i = 0; i < NB_SUBKEY_INIT-1 ; i+=3){
					for (j = 0; j < preprocess->updated_nb_key_value[i/3] ; j++){
						res[i/3][j] = score_mat_init[i][j/(NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT)] + score_mat_init[i+1][(j/NB_KEY_VALUE_INIT)%NB_KEY_VALUE_INIT] + score_mat_init[i+2][j%NB_KEY_VALUE_INIT];
					}
				}

				if ( (NB_SUBKEY_INIT%3) == 1 ){
					for (i = 0; i < preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1] ; i++){
						res[NB_SUBKEY_INIT/3][i] = score_mat_init[NB_SUBKEY_INIT-1][i];
					}
				}
				else{
					for (i = 0; i < preprocess->updated_nb_key_value[preprocess->updated_nb_subkey-1] ; i++){
						res[NB_SUBKEY_INIT/3][i] = score_mat_init[NB_SUBKEY_INIT-2][i/NB_KEY_VALUE_INIT] + score_mat_init[NB_SUBKEY_INIT-1][j%NB_KEY_VALUE_INIT];
					}
				}
			}


			else{//it divides
				preprocess->updated_nb_subkey = NB_SUBKEY_INIT/3;
				preprocess->updated_nb_key_value = (int*) malloc(preprocess->updated_nb_subkey*sizeof(int));

				res = (double**) malloc( preprocess->updated_nb_subkey* sizeof(double*));
				for (i = 0 ; i < preprocess->updated_nb_subkey ; i++){
					preprocess->updated_nb_key_value[i] = NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT;
					res[i] = (double*) malloc(preprocess->updated_nb_key_value[i] * sizeof(double));
				}

				for (i = 0; i < NB_SUBKEY_INIT ; i+=3){
					for (j = 0; j < preprocess->updated_nb_key_value[i/3] ; j++){
						res[i/3][j] =  score_mat_init[i][j/(NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT)] + score_mat_init[i+1][(j/NB_KEY_VALUE_INIT)%NB_KEY_VALUE_INIT] + score_mat_init[i+2][j%NB_KEY_VALUE_INIT];
					}
				}
			}

			break;

		default:
			printf("merge value not supported\n");
			break;

	}

	return res;
}

void merge_key_score( hel_data_t* data, hel_preprocessing_t* preprocess, double** score_mat_init ,int* key_init){

	int i;

	switch(preprocess->merge_value){

		case 1: //no merge
			data->log_probas_real_key = (double*) malloc( preprocess->updated_nb_subkey* sizeof(double));
			data->real_key = (int*) malloc( preprocess->updated_nb_subkey* sizeof(double));
			for (i = 0 ; i < preprocess->updated_nb_subkey ; i++){
				data->real_key[i] = key_init[i];
				data->log_probas_real_key[i] = score_mat_init[i][key_init[i]];
			}
			break;

		case 2:

			if ( NB_SUBKEY_INIT%2 ){
				data->log_probas_real_key = (double*) malloc( preprocess->updated_nb_subkey* sizeof(double));
				data->real_key = (int*) malloc( preprocess->updated_nb_subkey* sizeof(double));
				for (i = 0 ; i < NB_SUBKEY_INIT-1 ; i+=2){
					data->real_key[i/2] = key_init[i]*NB_KEY_VALUE_INIT + key_init[i+1];
					data->log_probas_real_key[i/2] = score_mat_init[i][key_init[i]] + score_mat_init[i+1][key_init[i+1]];
				}
				data->real_key[preprocess->updated_nb_subkey-1] = key_init[NB_SUBKEY_INIT-1];
				data->log_probas_real_key[preprocess->updated_nb_subkey-1] = score_mat_init[NB_SUBKEY_INIT-1][key_init[NB_SUBKEY_INIT-1]];
			}

			else{
				data->log_probas_real_key = (double*) malloc( preprocess->updated_nb_subkey* sizeof(double));
				data->real_key = (int*) malloc( preprocess->updated_nb_subkey* sizeof(double));
				for (i = 0 ; i < NB_SUBKEY_INIT ; i+=2){
					data->real_key[i/2] = key_init[i]*NB_KEY_VALUE_INIT + key_init[i+1];
					data->log_probas_real_key[i/2] = score_mat_init[i][key_init[i]] + score_mat_init[i+1][key_init[i+1]];
				}
			}
			break;

		case 3:

			if ( NB_SUBKEY_INIT%3){
				data->log_probas_real_key = (double*) malloc(preprocess->updated_nb_subkey*sizeof(double));
				data->real_key = (int*) malloc( preprocess->updated_nb_subkey* sizeof(double));
				for (i = 0 ; i < NB_SUBKEY_INIT-1 ; i+=3){
					data->real_key[i/3] = key_init[i]*NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT + key_init[i+1]*NB_KEY_VALUE_INIT+key_init[i+2];
					data->log_probas_real_key[i/3] = score_mat_init[i][key_init[i]] + score_mat_init[i+1][key_init[i+1]] + score_mat_init[i+2][key_init[i+2]];
				}

				if ( (NB_SUBKEY_INIT%3 == 1)){
					data->real_key[preprocess->updated_nb_subkey-1] = key_init[NB_SUBKEY_INIT-1];
					data->log_probas_real_key[preprocess->updated_nb_subkey-1] = score_mat_init[NB_SUBKEY_INIT-1][key_init[NB_SUBKEY_INIT-1]];
				}
				else{
					data->real_key[preprocess->updated_nb_subkey-1] = key_init[NB_SUBKEY_INIT-2]*NB_KEY_VALUE_INIT*key_init[NB_SUBKEY_INIT-1];
					data->log_probas_real_key[preprocess->updated_nb_subkey-1] = score_mat_init[NB_SUBKEY_INIT-2][key_init[NB_SUBKEY_INIT-2]] + score_mat_init[NB_SUBKEY_INIT-1][key_init[NB_SUBKEY_INIT-1]];
				}
			}

			else{
				data->log_probas_real_key = (double*) malloc( preprocess->updated_nb_subkey* sizeof(double));
				data->real_key = (int*) malloc( preprocess->updated_nb_subkey* sizeof(double));
				for (i = 0 ; i < NB_SUBKEY_INIT ; i+=3){
					data->real_key[i/3] = key_init[i]*NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT + key_init[i+1]*NB_KEY_VALUE_INIT + key_init[i+2];
					data->log_probas_real_key[i/3] = score_mat_init[i][key_init[i]] + score_mat_init[i+1][key_init[i+1]] + score_mat_init[i+2][key_init[i+2]];
				}
			}

			break;

		default:
			printf("merge value not supported\n");
			break;

	}


}

