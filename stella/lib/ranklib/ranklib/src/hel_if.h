#ifndef HEL_IF_H
#define HEL_IF_H

#include <NTL/ZZXFactoring.h>
#include <NTL/RR.h>

#include <stdint.h>
#include "hel_struct.h"

#include "hel_init.h"
#include "hel_histo.h"

extern "C" {
    double hel_ZZ2double(const NTL::ZZ* x);
    int64_t hel_ZZ2ll(const NTL::ZZ* x);
    double* hel_ZZX2doublearray(const NTL::ZZX* x, size_t* len);
    int64_t* hel_ZZX2llarray(const NTL::ZZX* x, size_t* len);
    double hel_RR2double(const NTL::RR* x);
    size_t hel_ZZXdeg(const NTL::ZZX* x);
    double hel_ZZXdoubleelem(const NTL::ZZX* x, size_t elem);
    void hel_double2ZZ(double x, NTL::ZZ* dest);
    void hel_double2RR(double x, NTL::RR* dest);
    NTL::ZZX* hel_alloc_ZZX_array(size_t n);
    void hel_ZZX_setlength(NTL::ZZX* x, size_t n);
    void hel_conv_hists(NTL::ZZX* dest, const NTL::ZZX* src1, const NTL::ZZX* src2);
    NTL::ZZ* hel_ZZX_index(NTL::ZZX* x, size_t n);
    void hel_ZZX_index_assign(NTL::ZZX* x, size_t n, const NTL::ZZ* val);
    void hel_ZZ_add(NTL::ZZ* dest, const NTL::ZZ* op1, const NTL::ZZ* op2);
    void hel_ZZ_incr(NTL::ZZ* dest);
    NTL::ZZX* hel_new_ZZX();
    void hel_delete_ZZX(NTL::ZZX* x);
    void hel_trunc_ZZX(NTL::ZZX* x, long m);
    double hel_ZZX_coeff(NTL::ZZX* x, long i);
    void hel_ZZX_SetCoeff(NTL::ZZX* x, long i, double y);
    void hel_ZZX_normalize(NTL::ZZX* x);
}

extern "C" int hel_top(const double * const*log_proba, const int* real_key, int nb_bin, int merge, double *min, double *est, double *max);
extern "C" hel_result_t* helc_execute_rank(int merge_value, int nb_bins, const double*const* score_mat_init, const int* key_init);
extern "C" void helc_free_result(hel_result_t* result);
extern "C" double helc_result_get_estimation_rank(const hel_result_t* result);
extern "C" double helc_result_get_estimation_rank_min(const hel_result_t* result);
extern "C" double helc_result_get_estimation_rank_max(const hel_result_t* result);
#if 0
extern "C" int hel_my_rank(int merge_value, int nb_bins, const double* const* score_mat_init, const int* key_init, hel_result_t** rresult);
#endif

#endif // HEL_IF_H
