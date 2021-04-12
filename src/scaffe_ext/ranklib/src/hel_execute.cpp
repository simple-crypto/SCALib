#include "hel_execute.h"
using namespace std;
using namespace NTL;



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

#if 0
extern "C" double** merge_mat_score_key(double** score_mat_init, int merge_value, int* updated_nb_subkey, int** updated_nb_key_value, int* key_init, int** real_key, double** log_probas_real_key);
extern "C" void get_real_key_info2(const hel_param_t *param, hel_result_t* result);
extern "C" void hel_compute_histograms_procedure2(hel_param_t* param);
extern "C" int hel_my_rank(int merge_value, int nb_bins, double** score_mat_init, int* key_init, hel_result_t** rresult){
	if (nb_bins < 1){
            fprintf(stderr,"Error in histogram initialization, incorrect bin number (%d)\n",nb_bins);
            return 1;
	}
	if ( (merge_value < 1) || (merge_value > 3) ){
		fprintf(stderr,"Error in preprocessing initialization, merge value not supported (%d)\n",merge_value);
                return 1;
	}
	int error = 0;
        hel_result_t* result;
	hel_param_t* param = hel_alloc_param();
        param->algo_mode = RANK;
        param->preprocessing->merge_value = merge_value;
        param->histo->nb_bins = nb_bins;
        param->data->log_probas = merge_mat_score_key( score_mat_init, merge_value, &param->preprocessing->updated_nb_subkey, &param->preprocessing->updated_nb_key_value, key_init, &param->data->real_key, &param->data->log_probas_real_key);
        param->data->log_probas_alloc_bool = 1;
        param->data->log_probas_real_key_alloc_bool = 1;
        param->data->real_key_alloc_bool = 1;

	//error = hel_compute_histogram(RANK, param->preprocessing, param->histo,  param->data, param->enum_input);
        //if (error) goto END;
        //param->histo->hists = compute_histograms_procedure(RANK, param->preprocessing, param->data,  param->histo,  param->enum_input  );
        hel_compute_histograms_procedure2(param);

	result = hel_alloc_result();
        *rresult = result;
        error = hel_init_result(result);
        if (error) goto END;
        //get_real_key_info( result->real_key_info , param->preprocessing,  param->histo, param->data );
        get_real_key_info2(param, result);
END:
	hel_free_param(param);
        return error;
}
#endif

hel_result_t* hel_execute_rank(int merge_value, int nb_bins, double** score_mat_init, int* key_init){

	hel_param_t* param = NULL;
	hel_result_t* result = NULL;
	int error = 0;

	clock_t t1,t2;
	t1 = clock();

	fprintf(stdout,"Starting preprocessing\n");
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

	fprintf(stdout,"Clearing memory\n");
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


