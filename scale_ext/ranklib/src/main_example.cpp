#include "hel_execute.h"
#include "scores_example.h"

using namespace std;
using namespace NTL;

//example
//serial version of the code

/*
	GETTING THE INPUT LOG PROBABILITIES:

    some log probabilities examples are hardcoded in score_example_dat.cpp
	They have been produced from template attacks on simulated AES-128 SBOX leakages.
    their names can be found in scores_example.h of the form : global_score_n
    n is the approximate rank of the real key for these log probas
    the real AES-128 key is (in decimal) {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    one can use these log probas by calling the function "double* get_scores_from_example(int rank)" defined in score_example.cpp
    it returns a double** corresponding to the log proba matrix of the example having a key rank of (approximatively) n.

    n can take the following values: 10, 21, 25, 29, 34, 39.

    One can also load the results of his own attacks (the results must be log probabilities, not probabilities, so don't forget to convert them to log before !)

	------------------------------------------

	CUSTOM KEY TESTING:

	If one want to use his own key testing (a different AES implementation, another cipher, some parallel key testing or sending the factorized key to a third party...)
	=> fill the empty space in the ELSE condition of function "print_file_key_original" in "hel_enum.cpp"
	=> set test_key to 0 so it doesn't use the current default AES-128 implementation that test keys on the fly using our (very not optimized) AES-128 optimization (and thus recombine the factorized subkeys)
	=> don't forget to set to_real_key to 0 as well.

	------------------------------------------

	CUSTEM NUMBER OF SUBKEY AND SUBKEYS LENGTH:

	if one want to attack something different than a AES-128 like case (i.e. something else than 16 lists of 8 bits subkeys).
	Then you must change the following defines defined in "hel_util.h"

	#define NB_SUBKEY_INIT => number of subkeys lists
	#define NB_KEY_dval_to_printUE_INIT => number of subkeys per lists

	the code will automaticaly do the right preprocessing according to the merging value.

	HOWEVER, the current tests that are implemented are for 16 lists of 256 elements.
	In this case, one must implement as well his own key testing to support this new setting. (See CUSTOM KEY SETTING above)

*/

int main(){


	int i;


	double** log_proba = NULL; //will contain the initial log probas
	int* real_key = NULL; //will contain the real key (optional for key enumeration)
	hel_result_t* result = NULL; //will contain the results of either rank estimation or key enumeration

	unsigned char kk[16]  = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	//real key of the simulated results

	unsigned char pt1[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	unsigned char pt2[16] = {255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
	unsigned char ct1[16] = {198,161,59,55,135,143,91,130,111,79,129,98,161,200,216,121};
	unsigned char cc2[16] = {60,68,31,50,206,7,130,35,100,215,162,153,14,80,187,19};
	//associated plaintexts/ciphertexts

	unsigned char** texts = (unsigned char**) malloc(4*sizeof(unsigned char*));
	for (i = 0 ; i < 4 ; i++){
		texts[i] = (unsigned char*) malloc(16*sizeof(unsigned char));
	}
	for (i = 0; i < 16 ; i++){
		texts[0][i] = pt1[i];
		texts[1][i] = pt2[i];
		texts[2][i] = ct1[i];
		texts[3][i] = cc2[i];
	}
	//load the plaintext ciphertext into a single unsigned char**
	//the form is the following => texts[0] contains the first plaintext, and tests[1] the associated ciphertext
	//texts[2] contains the 2nd plaintext and test[3] the associated ciphertext


	real_key =  (int*) malloc(16*sizeof(int));
	for ( i = 0; i < 16 ; i++){
		real_key[i] = (int) kk[i];
	}
	//load the real subkey dval_to_printues


	log_proba = get_scores_from_example(21);
	//load log probas from examples where the real key depth is around 2^21
	//function defined in score_example.cpp
	//parameter: the approximated rank of the real key.
	//6 possibles log probas are provided with a key depth of (approximatively) 10,21,25,29,34 and 39.

	//one can instead load its own attack result (and the associated true subkeys if needed and the associated plaintexts/ciphertexts for key testing on the fly


	//STARTING DECLARATION OF INPUT PARAMETERS
	int nb_bin;
	int merge;
	ZZ bound_start;
	ZZ bound_stop;
	int to_bound;
	int test_key;
	int to_real_key;
	//ENDING DECLARATION OF INPUT PARAMETERS
	

	//STARTING DECLARATION OF OUTPUT PARAMETERS
	ZZ rank_estim_rounded,rank_estim_min,rank_estim_max;
	double time_rank;
	double time_preprocessing;
	double time_enum;
	ZZ rank_enum_rounded,rank_enum_min,rank_enum_max;
	//ENDING DECLARATION OF OUTPUT PARAMETERS

	ZZ two;
	two = 2; //used to compute the bound_start and bound_end using two^(integer)
	//note that the bounds can be any number (not especially a power of two)

	RR rval_to_print; //to convert some ZZ to RR
	RR rr_log2 = log(conv<RR>(two)); //log2 as RR

	//EXAMPLE FOR RANK ESTIMATION

	nb_bin = 2048;
	merge = 2;
	//setting rank estimation parameters

	cout << "results rank estimation" << endl;
	cout << "nb_bin = " << nb_bin << endl;
	cout << "merge = " << merge << endl;

	result = hel_execute_rank(merge, nb_bin, log_proba, real_key);

	rank_estim_rounded = hel_result_get_estimation_rank(result);
	rank_estim_min = hel_result_get_estimation_rank_min(result);
	rank_estim_max = hel_result_get_estimation_rank_max(result);
	time_rank = hel_result_get_estimation_time(result);
	//these result accessors are in hel_init.cpp/h

	
	rval_to_print = conv<RR>(rank_estim_min);
        cout << "min: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_estim_rounded);
        cout << "actual rounded: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_estim_max);
        cout << "max: 2^" << log(rval_to_print)/rr_log2 <<endl;
        cout << "time enum: " << time_enum << " seconds" << endl;
	cout << endl << endl;
	hel_free_result(result);






	//EXAMPLES FOR ENUMERATION
	//WE SHOW THE SAME EXAMPLES AS IN THE PAPER

	//(1) => enumerate from "bound_start=2^10" up to "bound_stop=2^40" by testing keys with aes (stops if the real key is found before)
    //THE PROVIDED AES IS NOT OPTIMIZED AT ALL

	nb_bin = 2048;
	merge = 1;
	bound_start = power(two,10);
	bound_stop = power(two,40);
	test_key = 1;
	to_bound = 1;
	to_real_key = 0;
	//setting enumeration parameters

	cout << "results key enumeration (example 1)" << endl;
	cout << "nb_bin = " << nb_bin << endl;
	cout << "merge = " << merge << endl;
	cout << "bound_start = 2^" << log2(conv<double>(bound_start)) << endl;
	cout << "bound_end = 2^" << log2(conv<double>(bound_stop)) << endl;
 	cout << "test_key = " << test_key << endl;
	cout << "to_bound = " << to_bound << endl;
	cout << "to_real_key = " << to_real_key << endl;

	result = hel_execute_enum(merge,nb_bin,log_proba, real_key, bound_start,bound_stop , test_key, texts,to_real_key,  to_bound); //launch enumeration

    if ( hel_result_is_key_found(result) ){ //print results if the key is found
        rank_enum_rounded = hel_result_get_enum_rank(result);
        rank_enum_min = hel_result_get_enum_rank_min(result);
        rank_enum_max = hel_result_get_enum_rank_max(result);
        time_enum = hel_result_get_enum_time(result);
		//these result accessors are in hel_init.cpp/h

        rval_to_print = conv<RR>(rank_enum_min);
        cout << "min: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_rounded);
        cout << "actual rounded: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_max);
        cout << "max: 2^" << log(rval_to_print)/rr_log2 <<endl;
        cout << "time enum: " << time_enum << " seconds" << endl;
    }
    else{
        cout << "key not found" << endl;
    }
    time_preprocessing = hel_result_get_enum_time_preprocessing(result);
    cout << "time preprocessing: " << time_preprocessing << " seconds" << endl;
    cout << endl << endl;
	hel_free_result(result);



	//(2) => enumerate up to "bound_stop" by testing keys
	//the testing part have to be implemented by the user in this case
	//in the "else" part of the function "print_file_key_original" of  the file "hel_enum.cpp
	//Useless to launch it if the user's testing part is not implemented
	/*

	nb_bin = 2048;
	merge = 1;
	test_key = 0;
	to_bound = 0;
	to_real_key = 0;
	bound_start = power(two,0);
	bound_stop = power(two,40);
	//setting enumeration parameters

	cout << "results key enumeration (example 2)" << endl;
	cout << "nb_bin = " << nb_bin << endl;
	cout << "merge = " << merge << endl;
	cout << "bound_start = " << log2(conv<double>(bound_start)) << endl;
	cout << "bound_end = " << log2(conv<double>(bound_stop)) << endl;
 	cout << "test_key = " << test_key << endl;
	cout << "to_bound = " << to_bound << endl;
	cout << "to_real_key = " << to_real_key << endl;

	result = hel_execute_enum(merge,nb_bin,log_proba, real_key, bound_start,bound_stop , test_key, texts,to_real_key,  to_bound); //launch enumeration

    if ( hel_result_is_key_found(result) ){ //print results if the key is found
        rank_enum_rounded = hel_result_get_enum_rank(result);
        rank_enum_min = hel_result_get_enum_rank_min(result);
        rank_enum_max = hel_result_get_enum_rank_max(result);
        time_enum = hel_result_get_enum_time(result);
		//these result accessors are in hel_init.cpp/h

        rval_to_print = conv<RR>(rank_enum_min);
        cout << "min: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_rounded);
        cout << "actual rounded: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_max);
        cout << "max: 2^" << log(rval_to_print)/rr_log2 <<endl;
        cout << "time enum: " << time_enum << " seconds" << endl;
    }
    else{
        cout << "key not found" << endl;
    }
    time_preprocessing = hel_result_get_enum_time_preprocessing(result);

	cout << "time preprocessing: " << time_preprocessing << " seconds" << endl;
	cout << endl << endl;
	hel_free_result(result);
	*/


	//(3) => enumerate up to the real key if it is ranked less than "bound_stop".
	//keys are not tested with AES and are kept factorized

	nb_bin = 2048;
	merge = 1;
	test_key = 0; //has no effect with "to_real_key > 0"
	to_bound = 0; //has no effect with "to_real_key > 0"
	to_real_key = 1;
	bound_start = power(two,0);
	bound_stop = power(two,32);
	//setting enumeration parameters

	cout << "results key enumeration (example 3)" << endl;
	cout << "nb_bin = " << nb_bin << endl;
	cout << "merge = " << merge << endl;
	cout << "bound_start = 2^" << log2(conv<double>(bound_start)) << endl;
	cout << "bound_end = 2^" << log2(conv<double>(bound_stop)) << endl;
 	cout << "test_key = " << test_key << endl;
	cout << "to_bound = " << to_bound << endl;
	cout << "to_real_key = " << to_real_key << endl;

	result = hel_execute_enum(merge,nb_bin,log_proba, real_key, bound_start,bound_stop , test_key, texts,to_real_key,  to_bound); //launch enumeration

    if ( hel_result_is_key_found(result) ){ //print results if the key is found
        rank_enum_rounded = hel_result_get_enum_rank(result);
        rank_enum_min = hel_result_get_enum_rank_min(result);
        rank_enum_max = hel_result_get_enum_rank_max(result);
        time_enum = hel_result_get_enum_time(result);
		//these result accessors are in hel_init.cpp/h

        rval_to_print = conv<RR>(rank_enum_min);
        cout << "min: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_rounded);
        cout << "actual rounded: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_max);
        cout << "max: 2^" << log(rval_to_print)/rr_log2 <<endl;
        cout << "time enum: " << time_enum << " seconds" << endl;
    }
    else{
        cout << "key not found" << endl;
    }
	time_preprocessing = hel_result_get_enum_time_preprocessing(result);
	cout << "time preprocessing: " << time_preprocessing << " seconds" << endl;
	cout << endl << endl;
	hel_free_result(result);




	//(4) => enumerate up to the real key if it is ranked less than "bound_stop".
	//keys are not tested with AES and defactorized

	nb_bin = 2048;
	merge = 1;
	test_key = 0; //has no effect with "to_real_key > 0"
	to_bound = 0; //has no effect with "to_real_key > 0"
	to_real_key = 2;
	bound_start = power(two,0);
	bound_stop = power(two,32);
	//setting enumeration parameters

	cout << "results key enumeration (example 4)" << endl;
	cout << "nb_bin = " << nb_bin << endl;
	cout << "merge = " << merge << endl;
	cout << "bound_start = 2^" << log2(conv<double>(bound_start)) << endl;
	cout << "bound_end = 2^" << log2(conv<double>(bound_stop)) << endl;
 	cout << "test_key = " << test_key << endl;
	cout << "to_bound = " << to_bound << endl;
	cout << "to_real_key = " << to_real_key << endl;
	
	result = hel_execute_enum(merge,nb_bin,log_proba, real_key, bound_start,bound_stop , test_key, texts,to_real_key,  to_bound); //launch enumeration

    if ( hel_result_is_key_found(result) ){ //print results if the key is found
        rank_enum_rounded = hel_result_get_enum_rank(result);
        rank_enum_min = hel_result_get_enum_rank_min(result);
        rank_enum_max = hel_result_get_enum_rank_max(result);
        time_enum = hel_result_get_enum_time(result);
		//these result accessors are in hel_init.cpp/h

        rval_to_print = conv<RR>(rank_enum_min);
        cout << "min: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_rounded);
        cout << "actual rounded: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_max);
        cout << "max: 2^" << log(rval_to_print)/rr_log2 <<endl;
        cout << "time enum: " << time_enum << " seconds" << endl;
    }
    else{
        cout << "key not found" << endl;
    }
	time_preprocessing = hel_result_get_enum_time_preprocessing(result);
	cout << "time preprocessing: " << time_preprocessing << " seconds" << endl;
	cout << endl << endl;
	hel_free_result(result);



	//(5) => enumerate up to the real key if it is ranked less than "bound_stop".
	//keys are tested with AES and (thus) defactorized

	nb_bin = 2048;
	merge = 1;
	test_key = 0; //has no effect with "to_real_key > 0"
	to_bound = 0; //has no effect with "to_real_key > 0"
	to_real_key = 3;
	bound_start = power(two,0);
	bound_stop = power(two,32);
	//setting enumeration parameters

	cout << "results key enumeration (example 5)" << endl;
	cout << "nb_bin = " << nb_bin << endl;
	cout << "merge = " << merge << endl;
	cout << "bound_start = 2^" << log2(conv<double>(bound_start)) << endl;
	cout << "bound_end = 2^" << log2(conv<double>(bound_stop)) << endl;
 	cout << "test_key = " << test_key << endl;
	cout << "to_bound = " << to_bound << endl;
	cout << "to_real_key = " << to_real_key << endl;

	result = hel_execute_enum(merge,nb_bin,log_proba, real_key, bound_start,bound_stop , test_key, texts,to_real_key,  to_bound); //launch enumeration

	if ( hel_result_is_key_found(result) ){ //print results if the key is found
        rank_enum_rounded = hel_result_get_enum_rank(result);
        rank_enum_min = hel_result_get_enum_rank_min(result);
        rank_enum_max = hel_result_get_enum_rank_max(result);
        time_enum = hel_result_get_enum_time(result);
		//these result accessors are in hel_init.cpp/h

        rval_to_print = conv<RR>(rank_enum_min);
        cout << "min: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_rounded);
        cout << "actual rounded: 2^" << log(rval_to_print)/rr_log2 <<endl;
        rval_to_print = conv<RR>(rank_enum_max);
        cout << "max: 2^" << log(rval_to_print)/rr_log2 <<endl;
        cout << "time enum: " << time_enum << " seconds" << endl;
	}
    else{
        cout << "key not found" << endl;
    }
	time_preprocessing = hel_result_get_enum_time_preprocessing(result);
	cout << "time preprocessing: " << time_preprocessing << " seconds" << endl;
	cout << endl << endl;
	hel_free_result(result);




	free_scores_example(log_proba);
	free(real_key);
	for (i = 0; i < 4 ; i++){
		free(texts[i]);
	}
	free(texts);


	return 0;
}


