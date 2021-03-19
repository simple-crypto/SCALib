#ifndef BIGNUMPOLY_H_
#define BIGNUMPOLY_H_

#include <stdlib.h>
#include <NTL/ZZXFactoring.h>

extern "C" {
    NTL::ZZX* bnp_new_ZZX();
    void bnp_delete_ZZX(NTL::ZZX* x);

    void bnp_ZZX_setlength(NTL::ZZX* x, size_t n);
    void bnp_trunc_ZZX(NTL::ZZX* x, long m);
    void bnp_ZZX_normalize(NTL::ZZX* x);

    void bnp_conv_hists(NTL::ZZX* dest, const NTL::ZZX* src1, const NTL::ZZX* src2);

    void bnp_ZZX_incr_coef(NTL::ZZX* x, size_t n);

    double bnp_ZZX_coeff(NTL::ZZX* x, long i);
}

#endif // BIGNUMPOLY_H_
