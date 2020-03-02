#include "hel_execute.h"



int hel_execute_procedure(hel_param_t* param, hel_result_t* result){

	int error = 0;

	clock_t t1,t2;
	t1 = clock();

	error = hel_compute_histogram( param->algo_mode, param->preprocessing, param->histo,  param->data, param->enum_input);

	if (!error){

		if(param->data->log_probas_real_key_alloc_bool){
			get_real_key_info( result->real_key_info , param->preprocessing,  param->histo, param->data );
		}


		if(param->algo_mode == ENUM){


			get_enum_info( result->enum_info, param->enum_input, param->preprocessing,  param->histo , result->real_key_info );

			error |= set_enum_input( param->enum_input,  result->enum_info , param->preprocessing, param->histo);

			t2 = clock();
			result->enum_info->preprocessing_time += ((double)(t2 - t1)) / ((double) CLOCKS_PER_SEC);

			if (!error){

				t1 = clock();
				start_recursive_enumeration( param->preprocessing,  param->histo, param->enum_input, result->enum_info);
				t2 = clock();
				result->enum_info->enum_time = ((double)(t2 - t1)) / ((double) CLOCKS_PER_SEC);
			}
		}

	}


	return error;


}


hel_result_t* hel_execute_rank(int merge_value, int nb_bins, double** score_mat_init, int* key_init){

	hel_param_t* param = NULL;
	hel_result_t* result = NULL;
	int error = 0;

	clock_t t1,t2;
	t1 = clock();

	param = hel_alloc_param();
	error = hel_init_param( param, RANK, merge_value,  nb_bins,  score_mat_init, key_init, conv<ZZ>(0), conv<ZZ>(0), 0, NULL,0,0);

	if(!error){

		result = hel_alloc_result();
		error |= hel_init_result(result);

		if(!error){

			error |= hel_execute_procedure(param,  result);
		}
		else{
			fprintf(stderr,"Procedure aborted\n");
		}
	}
	else{
		fprintf(stderr,"Error in result initialization\n");
	}

	t2 = clock();
	result->real_key_info->rank_estimation_time = ((double)(t2 - t1)) / ((double) CLOCKS_PER_SEC);

	hel_free_param(param);

	return result;
}



hel_result_t* hel_execute_enum(int merge_value, int nb_bins, double** score_mat_init, int* key_init, ZZ bound_start, ZZ bound_end,int test_key_bool, unsigned char** pt_ct, int enumerate_to_real_key_rank_bool, int up_to_bound_bool){

	hel_param_t* param = NULL;
	hel_result_t* result = NULL;
	int error = 0;

	clock_t t1,t2;
	t1 = clock();

	fprintf(stdout,"Starting preprocessing\n");
	param = hel_alloc_param();
	error = hel_init_param( param, ENUM, merge_value,  nb_bins,  score_mat_init, key_init, bound_start, bound_end, test_key_bool, pt_ct, enumerate_to_real_key_rank_bool, up_to_bound_bool);

	if(!error){

		result = hel_alloc_result();
		error |= hel_init_result(result);

		if(!error){
			t2 = clock();
			result->enum_info->preprocessing_time = ((double)(t2 - t1)) / ((double) CLOCKS_PER_SEC);

			error |= hel_execute_procedure(param,  result);
		}
		else{
			fprintf(stderr,"Procedure aborted\n");
		}
	}
	else{
		fprintf(stderr,"Error in result initialization\n");
	}

	fprintf(stdout,"Clearing memory\n");
	hel_free_param(param);

	return result;

}


