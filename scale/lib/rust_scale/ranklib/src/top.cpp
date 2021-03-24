#include "hel_execute.h"
#include "scores_example.h"
#include "hel_if.h"

using namespace std;
using namespace NTL;

extern "C" int nb_subkey_init () { return NB_SUBKEY_INIT; }
extern "C" int nb_key_value_init() { return NB_KEY_VALUE_INIT; }
extern "C" double gs21(int i, int j) { return global_score_21[i][j]; }

extern "C" hel_result_t* helc_execute_rank(int merge_value, int nb_bins, const double* const* score_mat_init, const int* key_init){
    return hel_execute_rank(merge_value, nb_bins, (double **) score_mat_init, (int *) key_init);
}
extern "C" void helc_free_result(hel_result_t* result) {
    hel_free_result(result);
}
extern "C" double helc_result_get_estimation_rank(const hel_result_t* result) {
    ZZ estim = hel_result_get_estimation_rank((hel_result_t *) result);
    return conv<double>(estim);
}
extern "C" double helc_result_get_estimation_rank_min(const hel_result_t* result) {
    ZZ estim = hel_result_get_estimation_rank_min((hel_result_t *) result);
    return conv<double>(estim);
}
extern "C" double helc_result_get_estimation_rank_max(const hel_result_t* result) {
    ZZ estim = hel_result_get_estimation_rank_max((hel_result_t *) result);
    return conv<double>(estim);
}

extern "C" int hel_top(const double *const*log_proba, const int* real_key, int nb_bin, int merge, double *min, double *est, double *max) {
  hel_result_t *result = NULL; // will contain the results of either rank
                               // estimation or key enumeration

  //int real_key[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // real key of the simulated results

  //double **log_proba = get_scores_from_example(21);

  //int nb_bin = 2048;
  //int merge = 2;
  // setting rank estimation parameters

  //cout << "results rank estimation" << endl;
  //cout << "nb_bin = " << nb_bin << endl;
  //cout << "merge = " << merge << endl;


  result = hel_execute_rank(merge, nb_bin, (double **) log_proba, (int *) real_key);

  ZZ rank_estim_rounded = hel_result_get_estimation_rank(result);
  ZZ rank_estim_min = hel_result_get_estimation_rank_min(result);
  ZZ rank_estim_max = hel_result_get_estimation_rank_max(result);
  double time_rank = hel_result_get_estimation_time(result);
  // these result accessors are in hel_init.cpp/h

  //cout << "min: 2^" << log(rank_estim_min) / log(2) << endl;
  //cout << "actual rounded: 2^" << log(rank_estim_rounded) / log(2) << endl;
  //cout << "max: 2^" << log(rank_estim_max) / log(2) << endl;
  //cout << "time rank: " << time_rank << " seconds" << endl;
  //cout << endl << endl;
  hel_free_result(result);

  *min = conv<double>(rank_estim_min);
  *est = conv<double>(rank_estim_rounded);
  *max = conv<double>(rank_estim_max);

  return 0;
}
