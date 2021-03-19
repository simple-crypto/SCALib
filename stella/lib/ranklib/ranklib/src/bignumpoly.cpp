
#include <stdlib.h>

#include <NTL/ZZXFactoring.h>
#include "bignumpoly.h"

using namespace std;
using namespace NTL;

extern "C" {
    NTL::ZZX* bnp_new_ZZX() {
        return new NTL::ZZX();
    }
    void bnp_delete_ZZX(NTL::ZZX* x) {
        delete x;
    }

    void bnp_ZZX_setlength(NTL::ZZX* x, size_t n) {
        (*x).SetLength(n);
    }
    void bnp_trunc_ZZX(NTL::ZZX* x, long m) {
        trunc(*x, *x, m);
    }
    void bnp_ZZX_normalize(NTL::ZZX* x) {
        (*x).normalize();
    }

    void bnp_conv_hists(NTL::ZZX* dest, const NTL::ZZX* src1, const NTL::ZZX* src2) {
        *dest = *src1 * *src2;
    }

    void bnp_ZZX_incr_coef(NTL::ZZX* x, size_t n) {
        ((*x)[n]) += 1;
    }

    double bnp_ZZX_coeff(NTL::ZZX* x, long i) {
        return conv<double>(coeff(*x, i));
    }
}
