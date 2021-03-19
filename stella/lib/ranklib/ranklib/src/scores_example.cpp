#include "scores_example.h"
using namespace std;
using namespace NTL;

double** get_scores_from_example(int rank){

	int i,j;

	double** scores = (double**) malloc(NB_SUBKEY_INIT*sizeof(double*));

	for (i = 0 ; i < NB_SUBKEY_INIT ; i++){
		scores[i] = (double*) malloc(NB_KEY_VALUE_INIT*sizeof(double));
	}

	switch(rank){

		case 10:
			for (i = 0; i < NB_SUBKEY_INIT ; i++){
				for (j = 0 ; j < NB_KEY_VALUE_INIT ; j++){
					scores[i][j] = global_score_10[i][j];
				}
			}
			break;

		case 21:
			for (i = 0; i < NB_SUBKEY_INIT ; i++){
				for (j = 0 ; j < NB_KEY_VALUE_INIT ; j++){
					scores[i][j] = global_score_21[i][j];
				}
			}
			break;

		case 25:
			for (i = 0; i < NB_SUBKEY_INIT ; i++){
				for (j = 0 ; j < NB_KEY_VALUE_INIT ; j++){
					scores[i][j] = global_score_25[i][j];
				}
			}
			break;

		case 29:
			for (i = 0; i < NB_SUBKEY_INIT ; i++){
				for (j = 0 ; j < NB_KEY_VALUE_INIT ; j++){
					scores[i][j] = global_score_29[i][j];
				}
			}
			break;

		case 34:
			for (i = 0; i < NB_SUBKEY_INIT ; i++){
				for (j = 0 ; j < NB_KEY_VALUE_INIT ; j++){
					scores[i][j] = global_score_34[i][j];
				}
			}
			break;

		case 39:
			for (i = 0; i < NB_SUBKEY_INIT ; i++){
				for (j = 0 ; j < NB_KEY_VALUE_INIT ; j++){
					scores[i][j] = global_score_39[i][j];
				}
			}
			break;


		default:
			fprintf(stderr,"Error: scores example does not exist for this rank\n");
			break;

	}
	
	return scores;
}



void free_scores_example(double** scores){

	int i;

	for (i = 0 ; i < NB_SUBKEY_INIT ; i++){
		free(scores[i]);
	}
	free(scores);

}
