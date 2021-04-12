#include "hel_enum.h"
using namespace std;
using namespace NTL;



/*
    convert subkeys lists into arrays (for a faster accessing during enumeration)
*/
int*** convert_list_into_array( hel_int_list_t*** key_list, int nb_bins, int updated_nb_subkey){

	int*** ret = (int***) malloc( updated_nb_subkey*sizeof(int**));

	int i,j,k;

	hel_int_list_t* curr;
	int nb_elem;

	for (i = 0 ; i < updated_nb_subkey ; i++){

		ret[i] = (int**) malloc( nb_bins*sizeof(int*));

		for (j = 0; j < nb_bins ; j++){

			nb_elem = key_list[i][j]->val;

			if ( nb_elem != 0){

				ret[i][j] = (int*) malloc( (nb_elem+1)*sizeof(int) );
				curr = key_list[i][j];

				for ( k = 0 ; k < nb_elem+1 ; k++){
					ret[i][j][k] = curr->val;
					curr = curr->next;
				}

			}
		}

	}
	return ret;
}


//convert all the ZZX hists used for the recursive iteration into  int* binary histograms which contains 0 if bin is empty, 1 otherwise.
//these binary hist will be used to check if the bin is empty (faster than checking with ZZX hists)
int** get_binary_hist_from_big_hist(ZZX* hists, int* big_hist_size,int* convolution_order, int updated_nb_subkey){

	int i,j;

	int** binary_hists = (int**) malloc( updated_nb_subkey*sizeof(int*));

	ZZ zz_zero;
	zz_zero = 0;

	for (i = updated_nb_subkey-1 ; i >= 1; i--){
		binary_hists[i] = (int*) calloc( deg(hists[updated_nb_subkey+i-1]) +1, sizeof(int) );
		big_hist_size[i] = deg(hists[updated_nb_subkey+i-1]) +1;

		for ( j = 0 ; j < big_hist_size[i] ; j++){
			if (hists[updated_nb_subkey+i-1][j] > zz_zero ){
				binary_hists[i][j] = 1; //get the non zero bins
			}
		}

	}
	binary_hists[0] = (int*) calloc(  deg(hists[convolution_order[1]]) +1 ,sizeof(int));
	big_hist_size[0] = deg(hists[convolution_order[1]]) +1;

	for ( j = 0 ; j < big_hist_size[0] ; j++){
		if (hists[convolution_order[1]][j] > zz_zero ){
			binary_hists[0][j] = 1; //get the non zero bins
		}
	}


	return binary_hists;

}



//this function find the bin associated to the bound (i.e. a number of key to enumerate)
//it simply count the number of key in the bins from the last until we cap the bound
//if boolean_end is set to 1, it means we check the bound for the ending bin of enumeration (and thus we remove one more bin)
int get_index_bin_bound(ZZX* hist, ZZ bound,ZZ* nb_total_elem,int boolean_end){

	ZZ nb_elem_next;
	nb_elem_next = 0;
	*nb_total_elem = 0;

	int bin_index = (int) deg(*hist);

	while ( (nb_elem_next <= bound) && bin_index >= 0){

		nb_elem_next += (*hist)[bin_index];


		if (nb_elem_next <= bound){
			*nb_total_elem = nb_elem_next;

		}

		else
			break;


		bin_index--;


	}

	if(boolean_end){ //this is the ending bin, not starting => remove 1
		bin_index--;
		*nb_total_elem += (*hist)[bin_index];
	}




	return bin_index;
}




/*
    fill up the enum_info_t from the user input
*/
void get_enum_info( hel_enum_info_t* enum_info, hel_enum_input_t* enum_input , hel_preprocessing_t* preprocessing, hel_histo_t* histo,hel_real_key_info_t* real_key_info){

	int i;


	if(enum_input->enumerate_to_real_key_rank){

		enum_info->bin_enum_end = real_key_info->bin_real_key-1;
		*(enum_info->nb_key_enum_end) = *(real_key_info->bound_real_key);

		enum_info->bin_enum_start = deg(histo->hists[preprocessing->updated_nb_subkey*2-2]);
		*(enum_info->nb_key_enum_start) = 1;

		enum_info->bound_bin_enum_start = -1; //not needed in this case
		enum_info->bound_bin_enum_end = -1; //not needed in this case

		*(enum_info->bound_nb_key_enum_start) = -1;
		*(enum_info->bound_nb_key_enum_end) = -1;

	}


	else{
		enum_info->bin_enum_start = get_index_bin_bound( &(histo->hists[preprocessing->updated_nb_subkey*2-2]), *(enum_input->bound_start), enum_info->nb_key_enum_start , 0 );

		if ( *(enum_info->nb_key_enum_start) == 0){
			*(enum_info->nb_key_enum_start) = 1;
		}

		enum_info->bin_enum_end = get_index_bin_bound( &(histo->hists[preprocessing->updated_nb_subkey*2-2]), *(enum_input->bound_end), enum_info->nb_key_enum_end , 1 );

		enum_info->bound_bin_enum_end = enum_info->bin_enum_end - preprocessing->updated_nb_subkey/2;
		if ( enum_info->bound_bin_enum_start < 0 ){
			enum_info->bound_bin_enum_start = 0;
		}


		*(enum_info->bound_nb_key_enum_end)  = *(enum_info->nb_key_enum_end);
		for(i = enum_info->bin_enum_end -1 ; i >= enum_info->bound_bin_enum_end ; i--){
			add( *(enum_info->bound_nb_key_enum_end) , *(enum_info->bound_nb_key_enum_end) , histo->hists[preprocessing->updated_nb_subkey*2-2][i] );
		}


		enum_info->bound_bin_enum_start = enum_info->bin_enum_start + preprocessing->updated_nb_subkey/2;
		if ( enum_info->bound_bin_enum_start > deg(histo->hists[preprocessing->updated_nb_subkey*2-2]) ){
			enum_info->bound_bin_enum_start = deg(histo->hists[preprocessing->updated_nb_subkey*2-2]);
		}

		*(enum_info->bound_nb_key_enum_start)  = *(enum_info->nb_key_enum_start);
		for(i = enum_info->bin_enum_start ; i < enum_info->bound_bin_enum_start ; i++){
			sub( *(enum_info->bound_nb_key_enum_start) , *(enum_info->bound_nb_key_enum_start) , histo->hists[preprocessing->updated_nb_subkey*2-2][i] );
		}

		if ( *(enum_info->bound_nb_key_enum_start) == 0 ){
			*(enum_info->bound_nb_key_enum_start) = 1;
		}

	}


}



/*
    fill up the enum_input_t in function of the user input
*/
int set_enum_input(hel_enum_input_t* enum_input, hel_enum_info_t* enum_info , hel_preprocessing_t* preprocessing, hel_histo_t* histo){

	int error = 0;



	if(enum_input->enumerate_to_real_key_rank && ( *(enum_input->bound_end) < *(enum_info->nb_key_enum_end) )  ){
		error = 1; //if we want to enumerate up to the real key but its rank is superior than the bound, we have error
		fprintf(stderr,"Error, cannot enumerate up to the real key (rank larger than specified bound).\n");



	}

	else{
		enum_input->key_list_bin = convert_list_into_array(  enum_input->key_list_bin2, histo->nb_bins , preprocessing->updated_nb_subkey);
		enum_input->key_list_bin_alloc_bool = 1;

		enum_input->binary_hists_size = (int*) malloc( preprocessing->updated_nb_subkey * sizeof(int));
		enum_input->binary_hists_size_alloc_bool = 1;

		enum_input->binary_hists = get_binary_hist_from_big_hist( histo->hists, enum_input->binary_hists_size, enum_input->convolution_order, preprocessing->updated_nb_subkey);
		enum_input->binary_hists_alloc_bool = 1;

		if( enum_input->enumerate_to_real_key_rank || !enum_input->up_to_bound_bool ){
			enum_input->bin_to_start = enum_info->bin_enum_start;
			enum_input->bin_to_end = enum_info->bin_enum_end;

			if( enum_input->enumerate_to_real_key_rank){
				*(enum_input->bound_start) = 1;
			}
		}
		else{
			enum_input->bin_to_start = enum_info->bound_bin_enum_start;
			enum_input->bin_to_end = enum_info->bound_bin_enum_end;
		}
	}

	return error;
}



//return 1 if the two arrays of unsigned char are equal, 0 otherwise
int test_key_equality(unsigned char* ct1, unsigned char* ct2, int size){

	int i;
	int ret_bool = 1;

	for (i = 0; i < size ; i++){
		if (ct1[i] != ct2[i]){
			ret_bool =0;
			break;
		}
	}

	return ret_bool;
}


//NORMAL key testing
//CAREFUL => the AES128 we provide along the library is not optimized at all.
//for custom case, the user have to implement it in the ELSE part
//return 1 is key is found in the current facto
int print_file_key_original(int** current_key_facto,int test_key_boolean, int updated_nb_subkey, int merge_value, unsigned char** pt_ct){


	int i,j;
	int* indexes;
	//for unfactorization, indexes is used to know the task to do.
	//indexes = [0,...0] means we haven't unfactorized any key yet.
	//it ends when indexes = [ current_key_facto[0][0], current_key_facto[1][0], ... , current_key_facto[updated_nb_subkey][0] ]

	int found_key = 0;
	int current_index;
	int incremente_to_do;
	unsigned char* ct;
	unsigned char* key;

	if (test_key_boolean){ //test key using the provided AES (not optimized)
							//suitable for AES 128 case

		indexes = (int*) calloc( updated_nb_subkey,sizeof(int));
		current_index = 0;
		incremente_to_do = 1;
		ct = (unsigned char*) malloc(NB_SUBKEY_INIT*sizeof(unsigned char));
		key = (unsigned char*) malloc(NB_SUBKEY_INIT*sizeof(unsigned char));

		while (current_index < updated_nb_subkey){

			if (found_key){
				break;
			}
			incremente_to_do = 1;

			while(incremente_to_do){
				switch(merge_value){

					case 1:
						for (i = 0 ; i < NB_SUBKEY_INIT ; i++){
							key[i] = (unsigned char) (current_key_facto[i][indexes[i]+1]);
						}
						break;

					case 2:
						for (i = 0 ; i < NB_SUBKEY_INIT/2 ; i++){
							key[2*i] = (unsigned char) (current_key_facto[i][indexes[i]+1]/NB_KEY_VALUE_INIT);
							key[2*i+1] = (unsigned char) (current_key_facto[i][indexes[i]+1]%NB_KEY_VALUE_INIT);
						}
						break;

					case 3:
						for (i = 0 ; i < NB_SUBKEY_INIT/3 ; i++){
							key[3*i] = (unsigned char) (current_key_facto[i][indexes[i]+1]/(NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT));
							key[3*i+1] = (unsigned char) ((current_key_facto[i][indexes[i]+1]/NB_KEY_VALUE_INIT)%NB_KEY_VALUE_INIT);
							key[3*i+2] = (unsigned char) (current_key_facto[i][indexes[i]+1]%NB_KEY_VALUE_INIT);
						}
						key[NB_SUBKEY_INIT-1] = (unsigned char) (current_key_facto[updated_nb_subkey-1][indexes[updated_nb_subkey-1]+1]);
						break;
				}
				aes(key,pt_ct[0], ct);

				if ( test_key_equality(ct, pt_ct[2], NB_SUBKEY_INIT)){
					aes(key, pt_ct[1], ct);
				}

				if ( test_key_equality(ct, pt_ct[3], NB_SUBKEY_INIT)){
					found_key = 1;
				}

				if (found_key){
					break;
				}

				//if we still have subkeys to enumeration in the current index
				if ( (indexes[current_index]+1) < current_key_facto[current_index][0]){
					//we go the the next subkey of this index
					indexes[current_index]++;
					incremente_to_do =0;
					for (j = 0 ; j < current_index ; j++){
						indexes[j] = 0; //we restart the previous indexes
					}
					current_index = 0; //and we go back to the first index

				}

				else{ //no more subkeys in this indexes
					current_index++; //we go to the next subkey
					if (current_index == updated_nb_subkey) //we stop if it was the last subkey
						incremente_to_do =0;

				}



			}

		}
		free(indexes);
		free(key);
		free(ct);
	}

	else{

		/*
		THIS IS SPACE MUST BE FILLED BY THE USER

        He can code here his own way to test the key (used a better AES or another algorithm or some custom parallelization or a different key length or send the key to a third party...).

		*/
	}

	return found_key;

}



//KEY testing for performance assessment
//CAREFUL => the aes used in the provided implementation is not optimized at all
//return 1 is key is found in the current facto
//suitable for 128 keys divided in 8 bits parts
int print_file_key_up_to_key(int** current_key_facto,int enumerate_to_real_key_rank, int updated_nb_subkey, int merge_value, unsigned char** pt_ct, int* real_key){


	int i,j;
	int* indexes;
	int current_index;
	int incremente_to_do;
	unsigned char* ct;
	unsigned char* key;
	int tmp_bool;
	int found_key = 0;

	if (enumerate_to_real_key_rank > 1){ //unfactorized keys

		indexes = (int*) calloc( updated_nb_subkey,sizeof(int));
		current_index = 0;
		incremente_to_do = 1;
		ct = (unsigned char*) malloc(NB_SUBKEY_INIT*sizeof(unsigned char));
		key = (unsigned char*) malloc(NB_SUBKEY_INIT*sizeof(unsigned char));

		while (current_index < updated_nb_subkey){ // we end when we have iterated over all the updated_nb_subkey indexes

			if (found_key){
				break;
			}
			incremente_to_do = 1;

			while(incremente_to_do){

				if ( enumerate_to_real_key_rank == 2){ //we just test the indexes, correspond to the time for defactorization without AES testing
					found_key = 1;
					for (i = 0; i < updated_nb_subkey ; i++){
						if ( current_key_facto[i][indexes[i]+1] !=  real_key[i]){
							found_key = 0;
							break;
						}
					}
				}

				else{ //defactorization + AES testing
					switch(merge_value){

						case 1:
							for (i = 0 ; i < NB_SUBKEY_INIT ; i++){
								key[i] = (unsigned char) (current_key_facto[i][indexes[i]+1]);
							}
							break;

						case 2:
							for (i = 0 ; i < NB_SUBKEY_INIT/2 ; i++){
								key[2*i] = (unsigned char) (current_key_facto[i][indexes[i]+1]/NB_KEY_VALUE_INIT);
								key[2*i+1] = (unsigned char) (current_key_facto[i][indexes[i]+1]%NB_KEY_VALUE_INIT);
							}
							break;

						case 3:
							for (i = 0 ; i < NB_SUBKEY_INIT/3 ; i++){
								key[3*i] = (unsigned char) (current_key_facto[i][indexes[i]+1]/(NB_KEY_VALUE_INIT*NB_KEY_VALUE_INIT));
								key[3*i+1] = (unsigned char) ((current_key_facto[i][indexes[i]+1]/NB_KEY_VALUE_INIT)%NB_KEY_VALUE_INIT);
								key[3*i+2] = (unsigned char) (current_key_facto[i][indexes[i]+1]%NB_KEY_VALUE_INIT);
							}
							key[NB_SUBKEY_INIT-1] = (unsigned char) (current_key_facto[updated_nb_subkey-1][indexes[updated_nb_subkey-1]+1]);
							break;
					}
					aes(key,pt_ct[0], ct);

					if ( test_key_equality(ct, pt_ct[2], NB_SUBKEY_INIT)){
						aes(key, pt_ct[1], ct);
					}

					if ( test_key_equality(ct, pt_ct[3], NB_SUBKEY_INIT)){
						found_key = 1;
					}
				}


				if (found_key){
					break;
				}

				//if we still have subkeys to enumeration in the current index
				if ( (indexes[current_index]+1) < current_key_facto[current_index][0]){
					//we go the the next subkey of this index
					indexes[current_index]++;
					incremente_to_do =0;
					for (j = 0 ; j < current_index ; j++){
						indexes[j] = 0; //we restart the previous indexes
					}
					current_index = 0; //and we go back to the first index

				}

				else{ //no more subkey in this indexes
					current_index++; //we go to the next sbox
					if (current_index == updated_nb_subkey) //we stop if it was the last sbox
						incremente_to_do =0;

				}
			}

		}
		free(indexes);
		free(ct);
		free(key);
	}

	else{ //keep keys factorized. Correspond to the actual enumeration time without testing (in a case where the testing would be done by a third party from the factorized keys)


		found_key = 1;
		for (i = 0; i < updated_nb_subkey ; i++){
			tmp_bool = 0;
			for (j = 1; j < current_key_facto[i][0]+1 ; j++){
				if ( current_key_facto[i][j] == real_key[i] ){
					tmp_bool =1;
				}
			}
			if (!tmp_bool){
				found_key = 0;
				break;
			}
		}

	}

	return found_key;
}





void decompose_bin(int current_small_hist, int current_index, hel_preprocessing_t* preprocessing, hel_enum_input_t* enum_input, int* found_key){

	int i,k;

	int current_index_small_hist;
	int current_index_big_hist_m_1;


    //if updated_nb_subkey = 16, we have 31 hist from 0 to 30.

    //if the small hist index is superior to 1, it means the current big hist is one of the non initial histograms. This first loop deals with this case

    //however, if the small hist is of index 1, we are dealing with the smaller big hist created with hist[0] and hist[1]. This case is dealt in another loop

	if (current_small_hist > 1 ){

		for (i = 1 ; i < enum_input->index_list[enum_input->convolution_order[current_small_hist]][0] +1 ; i++){
			//iterate over all the non empty index of the small hist
			if (*found_key){
				break;
			}

			current_index_small_hist = enum_input->index_list[enum_input->convolution_order[current_small_hist]][i];
			//index of the small histl

			current_index_big_hist_m_1 = current_index - current_index_small_hist;
			//associated index of the big hist minus 1

			if (current_index_big_hist_m_1 >= enum_input->binary_hists_size[current_small_hist-1]){
				//if the this index of big hist minus 1 exceed the size of big hist minus 1

				//counter_stop2 +=1.;
				break; //we can stop cuz the bin indexes of the small hist have been sorted in decreasing order
						//thus all the following indexes of the big_m_1 hist will be bigger and thus be also higher than the size of big hist minus 1
					//this sorting improve the efficiency since we can just stop the loop
			}


			//if the bin of this index of big hist minus 1 is non empty => we have a match
			if ( enum_input->binary_hists[current_small_hist-1][current_index_big_hist_m_1] ){

				enum_input->key_factorization[enum_input->convolution_order[current_small_hist]][0] = enum_input->key_list_bin[enum_input->convolution_order[current_small_hist]][current_index_small_hist][0];

				for (k = 1 ; k < enum_input->key_factorization[enum_input->convolution_order[current_small_hist]][0] +1 ; k++){

					enum_input->key_factorization[enum_input->convolution_order[current_small_hist]][k] = enum_input->key_list_bin[enum_input->convolution_order[current_small_hist]][current_index_small_hist][k];
				}

				//find the match for the next subkey
				//decompose_bin( current_iteration+1,  current_index_big_hist_m_1,  index_list, binary_hist,  binary_hist_size, key_list_bin, current_key_facto, convo_order, updated_nb_subkey, merge_value, unfactorize_bool);
				decompose_bin(current_small_hist-1, current_index_big_hist_m_1,  preprocessing,  enum_input,found_key);
			}
		}
	}


    //case where we deal with the smaller big hist ( = hist[16] if updated_nb_subkey = 16)
    //this hist has been created from hist[0] and hist[1]
    //this deal with this particular case. This case is the stop condition of the recursiv alg (i.e. we have dealt with all the 16 sboxes)
	else{

		//we have no more real big hist minus 1, it's the hist of index 0 now
		//the function works pretty much as in the previous case
		for (i = 1 ; i < enum_input->index_list[enum_input->convolution_order[0]][0] +1 ; i++){

			if (*found_key){
				break;
			}

			current_index_small_hist = enum_input->index_list[enum_input->convolution_order[0]][i];
			//index small hist

			current_index_big_hist_m_1 = current_index - current_index_small_hist;
			//index first hist

			if (current_index_big_hist_m_1 >= enum_input->binary_hists_size[0]){
				//counter_stop4 +=1.;
				break; //we can stop cuz the bin indexes of the small hist have been sorted in decreasing order
						//thus all the following indexes of the big_m_1 hist will be bigger and thus be also higher than "current_size_big_hist_minus_1"
			}

            //if non empty, we have a final match
			if ( enum_input->binary_hists[0][current_index_big_hist_m_1] ){

				enum_input->key_factorization[enum_input->convolution_order[1]][0] = enum_input->key_list_bin[enum_input->convolution_order[1]][current_index_big_hist_m_1][0];

                //first subkey
				for (k = 1 ; k < enum_input->key_factorization[enum_input->convolution_order[1]][0] +1 ; k++){
					enum_input->key_factorization[enum_input->convolution_order[1]][k] = enum_input->key_list_bin[enum_input->convolution_order[current_small_hist]][current_index_big_hist_m_1][k];
				}
				enum_input->key_factorization[enum_input->convolution_order[0]][0] = enum_input->key_list_bin[enum_input->convolution_order[0]][current_index_small_hist][0];

                //second subkey
				for (k = 1 ; k < enum_input->key_factorization[enum_input->convolution_order[0]][0] +1 ; k++){
					enum_input->key_factorization[enum_input->convolution_order[0]][k] = enum_input->key_list_bin[enum_input->convolution_order[0]][current_index_small_hist][k];
				}

				//key testing
				if ( !enum_input->enumerate_to_real_key_rank){
					*found_key = print_file_key_original(enum_input->key_factorization, enum_input->test_key_boolean, preprocessing->updated_nb_subkey, preprocessing->merge_value, enum_input->pt_ct);
				}
				else{
					*found_key = print_file_key_up_to_key(enum_input->key_factorization,enum_input->enumerate_to_real_key_rank, preprocessing->updated_nb_subkey, preprocessing->merge_value, enum_input->pt_ct, enum_input->real_key);
				}
			}
		}
	}
}



//main function for enumeration that call decompose_bin
void start_recursive_enumeration( hel_preprocessing_t* preprocessing, hel_histo_t* histo, hel_enum_input_t* enum_input, hel_enum_info_t* enum_info){

	int i,j;
	int current_small_hist = preprocessing->updated_nb_subkey-1;

    //variable used to print the current part of enumeration
	RR rr_two = conv<RR>("2");
    RR current_power= conv<RR>(  log2(conv<double>(*(enum_input->bound_start)))  );


	int total_bin_nb = (int) deg(histo->hists[preprocessing->updated_nb_subkey*2-2])+1;

	*(enum_info->nb_key_enum_found_key) = 0;

	double total_enumerated_so_far=0;

    enum_info->found_key_boolean = 0;

    //skip the bin until we start and the one we want (add the number of key we skip for printing the key counter)
	for (i = total_bin_nb-1 ; i > enum_input->bin_to_start ; i--){
		add(*(enum_info->nb_key_enum_found_key),*(enum_info->nb_key_enum_found_key), histo->hists[preprocessing->updated_nb_subkey*2-2][i]);
	}

    //enumeration loop
	for (i = enum_input->bin_to_start ; i >= enum_input->bin_to_end ; i--){

		if (enum_info->found_key_boolean){
			i++;
			break;
		}

		if (enum_input->binary_hists[preprocessing->updated_nb_subkey-1][i] ){

			decompose_bin(current_small_hist,i,  preprocessing,  enum_input, &(enum_info->found_key_boolean));

			add(*(enum_info->nb_key_enum_found_key),*(enum_info->nb_key_enum_found_key), histo->hists[preprocessing->updated_nb_subkey*2-2][i]);

			if ( log(  conv<RR>( *(enum_info->nb_key_enum_found_key)) )/log(rr_two) > current_power  ){
				cout << "current rank : 2^" << log(  conv<RR>( *(enum_info->nb_key_enum_found_key)) )/log(rr_two) << endl;
				current_power += conv<RR>("1");
			}
		}
	}

    //update the key info if found
	if(enum_info->found_key_boolean){

		enum_info->bin_found_key = i;
		enum_info->bin_found_key_bound_max = i-preprocessing->updated_nb_subkey/2;
		enum_info->bin_found_key_bound_min = i+preprocessing->updated_nb_subkey/2;

		if ( enum_info->bin_found_key_bound_max < 0 ){
			enum_info->bin_found_key_bound_max = 0;
		}
		*(enum_info->nb_key_enum_found_key_bound_max) = *(enum_info->nb_key_enum_found_key);
		for (j = enum_info->bin_found_key-1 ; j >= enum_info->bin_found_key_bound_max ; j--){
			add( *(enum_info->nb_key_enum_found_key_bound_max), *(enum_info->nb_key_enum_found_key_bound_max), histo->hists[preprocessing->updated_nb_subkey*2-2][j] );
		}



		if ( enum_info->bin_found_key_bound_min > (int) deg(histo->hists[preprocessing->updated_nb_subkey*2-2]) ){
			enum_info->bin_found_key_bound_min = deg(histo->hists[preprocessing->updated_nb_subkey*2-2]);
		}
		*(enum_info->nb_key_enum_found_key_bound_min) = *(enum_info->nb_key_enum_found_key);


		for (j = enum_info->bin_found_key ; j < enum_info->bin_found_key_bound_min ; j++){
			sub( *(enum_info->nb_key_enum_found_key_bound_min), *(enum_info->nb_key_enum_found_key_bound_min), histo->hists[preprocessing->updated_nb_subkey*2-2][j] );
		}


		if ( *(enum_info->nb_key_enum_found_key_bound_min) == conv<ZZ>(0) ){
			*(enum_info->nb_key_enum_found_key_bound_min) = conv<ZZ>(1);
		}
	}

    //key not found
	else{
		*(enum_info->nb_key_enum_found_key) = conv<ZZ>(-1);
	}
}



