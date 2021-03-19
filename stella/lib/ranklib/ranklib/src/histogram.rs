#[cfg(feature = "ntl")]
mod bnp {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/binding_bnp.rs"));
}

pub trait Histogram {
    fn new(size: usize) -> Self;
    fn convolve(&self, other: &Self) -> Self;
    fn coefs_f64(&self) -> Vec<f64>;
    fn from_elems(size: usize, iter: impl Iterator<Item = usize>) -> Self;
}

type FFT = std::sync::Arc<dyn realfft::RealToComplex<f64>>;
type IFFT = std::sync::Arc<dyn realfft::ComplexToReal<f64>>;

#[derive(Clone)]
pub struct F64Hist {
    state: Vec<f64>,
    fft: FFT,
    ifft: IFFT,
}

impl Histogram for F64Hist {
    fn new(size: usize) -> Self {
        let mut planner = realfft::RealFftPlanner::<f64>::new();
        Self {
            state: vec![0.0; size],
            fft: planner.plan_fft_forward(2 * size),
            ifft: planner.plan_fft_inverse(2 * size),
        }
    }
    fn convolve(&self, other: &Self) -> Self {
        assert_eq!(self.state.len(), other.state.len());
        let mut self_tr = {
            let mut tr = self.fft.make_output_vec();
            let mut input = vec![0.0; 2 * self.state.len()];
            input
                .iter_mut()
                .zip(self.state.iter())
                .for_each(|(i, s)| *i = *s);
            self.fft.process(&mut input, &mut tr).unwrap();
            tr
        };
        let other_tr = {
            let mut tr = self.fft.make_output_vec();
            let mut input = vec![0.0; 2 * self.state.len()];
            input
                .iter_mut()
                .zip(other.state.iter())
                .for_each(|(i, s)| *i = *s / (self.state.len() as f64 * 2.0));
            self.fft.process(&mut input, &mut tr).unwrap();
            tr
        };
        self_tr
            .iter_mut()
            .zip(other_tr.iter())
            .for_each(|(s, o)| *s *= *o);
        let mut res = vec![0.0; 2 * self.state.len()];
        self.ifft.process(&mut self_tr, &mut res).unwrap();
        return Self {
            state: res[..self.state.len()].iter().map(|x| x.round()).collect(),
            fft: self.fft.clone(),
            ifft: self.ifft.clone(),
        };
    }
    fn coefs_f64(&self) -> Vec<f64> {
        self.state.clone()
    }
    fn from_elems(size: usize, iter: impl Iterator<Item = usize>) -> Self {
        let mut res = Self::new(size);
        for elem in iter {
            if elem < size {
                res.state[elem] += 1.0;
            }
        }
        return res;
    }
}

#[cfg(feature = "ntl")]
pub struct BigNumHist(*mut bnp::NTL_ZZX, usize);

#[cfg(feature = "ntl")]
impl Histogram for BigNumHist {
    fn new(nb_bins: usize) -> Self {
        return Self(unsafe { bnp::bnp_new_ZZX() }, nb_bins);
    }
    fn convolve(&self, other: &Self) -> Self {
        assert_eq!(self.1, other.1);
        let res = Self::new(self.1);
        unsafe {
            bnp::bnp_conv_hists(res.0, self.0, other.0);
            bnp::bnp_trunc_ZZX(res.0, self.1 as i64);
        }
        return res;
    }
    fn coefs_f64(&self) -> Vec<f64> {
        self.coefs().collect()
    }
    fn from_elems(nb_bins: usize, iter: impl Iterator<Item = usize>) -> Self {
        unsafe {
            let hist = bnp::bnp_new_ZZX();
            bnp::bnp_ZZX_setlength(hist, nb_bins);
            for item in iter {
                if item < nb_bins {
                    bnp::bnp_ZZX_incr_coef(hist, item);
                }
            }
            bnp::bnp_ZZX_normalize(hist);
            return Self(hist, nb_bins);
        }
    }
}

#[cfg(feature = "ntl")]
impl BigNumHist {
    fn get_coeff(&self, i: usize) -> f64 {
        unsafe { bnp::bnp_ZZX_coeff(self.0, i as i64) }
    }
    fn len(&self) -> usize {
        self.1
    }
    fn coefs<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        (0..self.len()).map(move |i| self.get_coeff(i))
    }
}

#[cfg(feature = "ntl")]
impl Drop for BigNumHist {
    fn drop(&mut self) {
        unsafe {
            bnp::bnp_delete_ZZX(self.0);
        }
    }
}
