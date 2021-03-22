
#include <stdlib.h>
#include "hel_if.h"

using namespace std;
using namespace NTL;

extern "C" double hel_ZZ2double(const NTL::ZZ* x) {
    return NTL::conv<double>(*x);
}
extern "C" int64_t hel_ZZ2ll(const NTL::ZZ* x) {
    return NTL::conv<int64_t>(*x);
}
extern "C" double* hel_ZZX2doublearray(const NTL::ZZX* x, size_t *len) {
    *len = deg(*x)+1;
    double* res = (double *) malloc(*len * sizeof(double));
    for (int i=0; i<*len; i++) {
        res[i] = hel_ZZ2double(&(*x)[i]);
    }
    return res;
}
extern "C" int64_t* hel_ZZX2llarray(const NTL::ZZX* x, size_t *len) {
    *len = deg(*x)+1;
    int64_t* res = (int64_t *) malloc(*len * sizeof(int64_t));
    for (int i=0; i<*len; i++) {
        res[i] = hel_ZZ2ll(&(*x)[i]);
    }
    return res;
}
extern "C" double hel_RR2double(const NTL::RR* x) {
    return NTL::conv<double>(*x);
}
extern "C" size_t hel_ZZXdeg(const NTL::ZZX* x) {
    return deg(*x);
}
extern "C" double hel_ZZXdoubleelem(const NTL::ZZX* x, size_t elem) {
    ZZ tmp = (*x)[elem];
    return hel_ZZ2double(&tmp);
}
extern "C" void hel_double2ZZ(double x, NTL::ZZ* dest) {
    NTL::ZZ y = conv<NTL::ZZ>(x);
    *dest = y;
}
extern "C" void hel_double2RR(double x, NTL::RR* dest) {
    *dest = conv<NTL::RR>(x);
}
extern "C" NTL::ZZX* hel_alloc_ZZX_array(size_t n) {
    return new NTL::ZZX[n];
}
extern "C" void hel_ZZX_setlength(NTL::ZZX* x, size_t n) {
    (*x).SetLength(n);
}
extern "C" void hel_conv_hists(NTL::ZZX* dest, const NTL::ZZX* src1, const NTL::ZZX* src2) {
    *dest = *src1 * *src2;
}
extern "C" NTL::ZZ* hel_ZZX_index(NTL::ZZX* x, size_t n) {
    return &((*x)[n]);
}
extern "C" void hel_ZZX_index_assign(NTL::ZZX* x, size_t n, const NTL::ZZ* val) {
    (*x)[n] = *val;
}
extern "C" void hel_ZZ_add(NTL::ZZ* dest, const NTL::ZZ* op1, const NTL::ZZ* op2) {
    *dest = *op1 + *op2;
}
extern "C" void hel_ZZ_incr(NTL::ZZ* dest) {
    *dest = *dest + 1;
}
extern "C" NTL::ZZX* hel_new_ZZX() {
    return new NTL::ZZX();
}
extern "C" void hel_delete_ZZX(NTL::ZZX* x) {
    delete x;
}
extern "C" void hel_trunc_ZZX(NTL::ZZX* x, long m) {
    trunc(*x, *x, m);
}
extern "C" double hel_ZZX_coeff(NTL::ZZX* x, long i) {
    return conv<double>(coeff(*x, i));
}
extern "C" void hel_ZZX_SetCoeff(NTL::ZZX* x, long i, double y) {
    SetCoeff(*x, i, conv<NTL::ZZ>(y));
}
extern "C" void hel_ZZX_normalize(NTL::ZZX* x) {
    (*x).normalize();
}
