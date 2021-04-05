#include "hel_histo.h"

using namespace std;
using namespace NTL;


//function to find the bin associated to the real key
int find_real_key_bin(hel_histo_t* histo, hel_data_t* data, hel_preprocessing_t* preprocessing){

	int i;
	RR tmp_score;
	int ret = 0;
	int tmp;

	for (i = 0; i < preprocessing->updated_nb_subkey ; i++){


		tmp_score = conv<RR>(data->log_probas_real_key[i] + data->shift);

		if ( tmp_score < (*(histo->width)) ){
			tmp = 0;
		}

		else{
			tmp = (int) conv<long>(tmp_score/ (*(histo->width)) );
		}

		if(tmp == histo->nb_bins){
			tmp--; //shouldnt happen if the range max has been slightly uped
		}

		ret +=tmp;
	}

	return ret;

}



//compute an histogram from log_probas
//if the algo mode is ENUM, also compute some preprocessing used for enumeration
void hel_compute_single_histogram(hel_algo_mode_t algo_mode, ZZX* hist, int subkey_number, hel_data_t* data, hel_histo_t* histo, hel_enum_input_t* enum_input, hel_preprocessing_t* preprocess){

	int i;

	hel_int_list_t** key_list_bin2_current;

	int target_bin;
	int target_index;
	int already_in;

	int ret_bin_subkey= -1;

	RR tmp_score;


	ZZ one_zz = conv<ZZ>(1);

	if (algo_mode == ENUM){

		 key_list_bin2_current = (hel_int_list_t**) malloc(histo->nb_bins * sizeof(hel_int_list_t*));

		for (i = 0 ; i < histo->nb_bins ; i++){ //all empty at the begining

			enum_input->index_list[subkey_number][0] = 0;

			enum_input->key_list_bin2[subkey_number][i] = hel_int_list_init();

			key_list_bin2_current[i] = enum_input->key_list_bin2[subkey_number][i]; //current = init

		}
	}




	int nb_test = preprocess->updated_nb_key_value[subkey_number];

	for ( i = 0; i < nb_test ; i++){
		tmp_score = conv<RR>(data->log_probas[subkey_number][i]);

		if (tmp_score < (*(histo->width)) ){
			target_bin = 0;
		}

		else{

			target_bin = (int) conv<long>(tmp_score/(*(histo->width)));

		}

		if (target_bin == histo->nb_bins) //shouldnt happen if the range max has been slightly uped
			target_bin = histo->nb_bins-1;




		add( ((*hist)[target_bin]) , ((*hist)[target_bin]) ,one_zz);

		if (algo_mode == ENUM){

			if ( enum_input->key_list_bin2[subkey_number][target_bin]->val == 0){ //if this bin was not selected yet
				//we add it to the non empty index list
				 enum_input->index_list[subkey_number][0]++;
				 enum_input->index_list[subkey_number][ enum_input->index_list[subkey_number][0]] = target_bin;

			}


			key_list_bin2_current[target_bin] = hel_int_list_add(i,key_list_bin2_current[target_bin]);
			//add elem and update current
			 enum_input->key_list_bin2[subkey_number][target_bin]->val++; //increment number

		}

	}
	if(algo_mode == ENUM){
		free(key_list_bin2_current);
	}



}












/*
	swap the column of a matrice target[n][*]
	each column might have a different size
*/
void swap_column(int** target, int index_col_1, int index_col_2){

	int* tmp = target[index_col_1];
	target[index_col_1] = target[index_col_2];
	target[index_col_2] = tmp;


}



//for qsort decreasing order int array
int cmp_array( const void* a, const void *b){

	int ret = -1;

	if ( *((int*) a) < *((int*)b) ){
		ret = 1;
	}

	else if ( *((int*) a) == *((int*)b) ){
		ret = 0;
	}

	return ret;
}


void search_convo_order(int** index, int* convo_order, int updated_nb_subkey){

	int* tmp_sort_nb_bin = (int*) malloc( updated_nb_subkey*sizeof(int));


	int i,j;
	int tmp;

	int max_index;
	int max_val;

	for (i = 0; i < updated_nb_subkey ; i++){
		convo_order[i] = i;
		tmp_sort_nb_bin[i] = index[i][0];
	}



	//naiv sort (updated_nb_subkey is typically small enough)
	for (i = 0; i < updated_nb_subkey-1; i++){

		max_index = i;
		max_val = tmp_sort_nb_bin[i];

		for (j = i+1 ; j < updated_nb_subkey ; j++){

			if ( tmp_sort_nb_bin[j] < max_val ){
				max_val = tmp_sort_nb_bin[j];
				max_index = j;
			}
		}

		tmp_sort_nb_bin[max_index] = tmp_sort_nb_bin[i];
		tmp_sort_nb_bin[i] = max_val;

		tmp = convo_order[max_index];
		convo_order[max_index] = convo_order[i];
		convo_order[i] = tmp;

	}



	free(tmp_sort_nb_bin);


}




//compute all the histograms and their convolution in function of the log probas inputs
ZZX* compute_histograms_procedure(hel_algo_mode_t algo_mode, hel_preprocessing_t* preprocessing, hel_data_t* data, hel_histo_t* histo, hel_enum_input_t* enum_input){

	double min_val, max_val;
	//int bin_real_key = 0;
	int i,j;

    //shift the log proba to positiv domain
	min_val = hel_min_max_mat(data->log_probas,  preprocessing->updated_nb_subkey , preprocessing->updated_nb_key_value, &max_val); //get the min and max log_probas
	for (i = 0 ; i < preprocessing->updated_nb_subkey; i++){
		for (j = 0 ; j < preprocessing->updated_nb_key_value[i] ; j++){
			data->log_probas[i][j] -= min_val;
		}
	}

	data->shift = -min_val;
	max_val -= min_val;
	//now min_val = 0

	RR rr_max = conv<RR>(max_val);
	RR rr_nb_bins = conv<RR>(histo->nb_bins);
	*(histo->width) = (rr_max + rr_max/rr_nb_bins) / rr_nb_bins; //small margin !

	ZZX* hists = new ZZX[preprocessing->updated_nb_subkey*2 -1];
	histo->hists_alloc_bool = 1;


	//compute the initial histograms
	for (i = 0;  i < preprocessing->updated_nb_subkey ; i++){
		hists[i].SetLength(histo->nb_bins);
		hel_compute_single_histogram( algo_mode, &(hists[i]), i, data, histo,  enum_input,  preprocessing);
	}

    //convolution + some preprocessing for enumeration
	if(algo_mode == ENUM){
		search_convo_order(enum_input->index_list,enum_input->convolution_order, preprocessing->updated_nb_subkey);

		//sort the index list of the non empty bins in the decreasing order
		for ( i = 0 ; i  < preprocessing->updated_nb_subkey ; i++){
			qsort(enum_input->index_list[i]+1, enum_input->index_list[i][0], sizeof(int), cmp_array);
		}

		hists[preprocessing->updated_nb_subkey] = hists[enum_input->convolution_order[0]]*hists[enum_input->convolution_order[1]]; //first convolution

		for( i = 2 ; i < preprocessing->updated_nb_subkey ; i++){
			hists[preprocessing->updated_nb_subkey+i-1] = hists[preprocessing->updated_nb_subkey+i-2]*hists[enum_input->convolution_order[i]]; //next convolutions
		}
	}

    //convolution
	else{

		hists[preprocessing->updated_nb_subkey] = hists[0]*hists[1]; //first convolution

		for( i = 2 ; i < preprocessing->updated_nb_subkey ; i++){

			hists[preprocessing->updated_nb_subkey+i-1] = hists[preprocessing->updated_nb_subkey+i-2]*hists[i]; //next convolutions

		}

	}

	return hists;
}






int hel_compute_histogram(hel_algo_mode_t algo_mode, hel_preprocessing_t* preprocessing, hel_histo_t* histo, hel_data_t* data, hel_enum_input_t* enum_input){

	int error = 0;

	if (algo_mode == RANK){
		if ( !data->log_probas_real_key_alloc_bool){
			error = 1;
			fprintf(stderr,"Error, enumeration cannot be launch without real key log_probas\n");
		}
	}

	if (!error){
		histo->hists = compute_histograms_procedure(algo_mode, preprocessing, data,  histo,  enum_input  );

	}

	return error;
}




//get info from histogram depending on params
void get_real_key_info(hel_real_key_info_t* real_key_info, hel_preprocessing_t* preprocessing, hel_histo_t* histo, hel_data_t* data ){

	int i;


	*(real_key_info->bound_real_key) = 0;
	*(real_key_info->bound_max) = 0;
	*(real_key_info->bound_min) = 0;

	int last_bin_index = (int) deg(histo->hists[preprocessing->updated_nb_subkey*2-2]);

	real_key_info->bin_real_key = find_real_key_bin(histo, data,  preprocessing);

	real_key_info->bin_bound_max = real_key_info->bin_real_key - preprocessing->updated_nb_subkey/2;
	if (real_key_info->bin_bound_max < 0){
		real_key_info->bin_bound_max = 0;
	}

	real_key_info->bin_bound_min = real_key_info->bin_real_key + preprocessing->updated_nb_subkey/2;
	if (real_key_info->bin_bound_min > last_bin_index){
		real_key_info->bin_bound_min = last_bin_index;
	}



	for (i = last_bin_index ; i > real_key_info->bin_bound_min ; i--){
		add( *(real_key_info->bound_min) , *(real_key_info->bound_min) , histo->hists[preprocessing->updated_nb_subkey*2-2][i]);
	}


	*(real_key_info->bound_real_key) = *(real_key_info->bound_min);
	if ( *(real_key_info->bound_min) == 0){
		*(real_key_info->bound_min) = 1;
	}

	for (i = real_key_info->bin_bound_min; i >= real_key_info->bin_real_key ; i--){
		add( *(real_key_info->bound_real_key) , *(real_key_info->bound_real_key) ,histo->hists[preprocessing->updated_nb_subkey*2-2][i]);
	}


	*(real_key_info->bound_max) = *(real_key_info->bound_real_key);


	for (i = real_key_info->bin_real_key-1 ; i >= real_key_info->bin_bound_max ; i--){
		add( *(real_key_info->bound_max) , *(real_key_info->bound_max) ,histo->hists[preprocessing->updated_nb_subkey*2-2][i]);
	}


}

