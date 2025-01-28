#[cfg(feature = "ntl")]
mod bnp {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/binding_bnp.rs"));
}

const CONV_LIMIT: f64 = 34359738368.0 as f64; //define a certain limit

pub trait Histogram {
    fn new(size: usize) -> Self;
    fn type_str(&self) -> &'static str;
    fn rescale(&self) -> Self;
    fn convolve(&self, other: &Self) -> Self;
    fn coefs_f64(&self) -> Vec<f64>;
    fn coefs_f64_upper(&self) -> Vec<f64>;
    fn from_elems(size: usize, iter: impl Iterator<Item = usize>) -> Self;
    fn scale_back(self) -> Self;
}

type FFT = std::sync::Arc<dyn realfft::RealToComplex<f64>>;
type IFFT = std::sync::Arc<dyn realfft::ComplexToReal<f64>>;

#[derive(Clone)]
pub struct F64Hist {
    state: Vec<f64>,
    fft: FFT,
    ifft: IFFT,
    state_upper: Vec<f64>,
    scale: f64,
}

impl Histogram for F64Hist {
    fn new(size: usize) -> Self {
        let mut planner = realfft::RealFftPlanner::<f64>::new();
        Self {
            state: vec![0.0; size],
            fft: planner.plan_fft_forward(2 * size),
            ifft: planner.plan_fft_inverse(2 * size),
            state_upper: vec![0.0; size],
            scale: 1.0,
        }
    }
    fn type_str(&self) -> &'static str {
        return "F64Hist";
    }
    fn rescale(&self) -> Self {
        let sum: f64 = self.state_upper.iter().sum();
        let scaler = if sum > CONV_LIMIT {
            sum / CONV_LIMIT
        } else {
            1.0
        };
        let new_scale = self.scale * scaler;
        let new_state: Vec<f64> = self.state.iter().map(|s| (s / scaler).floor()).collect();
        let new_upper_state: Vec<f64> = self
            .state_upper
            .iter()
            .map(|s| (s / scaler).ceil())
            .collect();

        return Self {
            state: new_state,
            fft: self.fft.clone(),
            ifft: self.ifft.clone(),
            state_upper: new_upper_state,
            scale: new_scale,
        };
    }
    fn convolve(&self, other: &Self) -> Self {
        assert_eq!(self.state.len(), other.state.len());
        assert_eq!(self.state_upper.len(), other.state_upper.len());
        let first_histogram = self.rescale();
        let second_histogram = other.rescale();

        let mut self_tr = {
            let mut tr = first_histogram.fft.make_output_vec();
            let mut input = vec![0.0; 2 * first_histogram.state.len()];
            input
                .iter_mut()
                .zip(first_histogram.state.iter())
                .for_each(|(i, s)| *i = *s);
            first_histogram.fft.process(&mut input, &mut tr).unwrap();
            tr
        };
        let mut self_tr_upper = {
            let mut tr = first_histogram.fft.make_output_vec();
            let mut input = vec![0.0; 2 * first_histogram.state_upper.len()];
            input
                .iter_mut()
                .zip(first_histogram.state_upper.iter())
                .for_each(|(i, s)| *i = *s);
            first_histogram.fft.process(&mut input, &mut tr).unwrap();
            tr
        };
        let other_tr = {
            let mut tr = first_histogram.fft.make_output_vec();
            let mut input = vec![0.0; 2 * first_histogram.state.len()];
            input
                .iter_mut()
                .zip(second_histogram.state.iter())
                .for_each(|(i, s)| *i = *s / (first_histogram.state.len() as f64 * 2.0));
            first_histogram.fft.process(&mut input, &mut tr).unwrap();
            tr
        };
        let other_tr_upper = {
            let mut tr = first_histogram.fft.make_output_vec();
            let mut input = vec![0.0; 2 * first_histogram.state_upper.len()];
            input
                .iter_mut()
                .zip(second_histogram.state_upper.iter())
                .for_each(|(i, s)| *i = *s / (first_histogram.state_upper.len() as f64 * 2.0));
            first_histogram.fft.process(&mut input, &mut tr).unwrap();
            tr
        };
        self_tr
            .iter_mut()
            .zip(other_tr.iter())
            .for_each(|(s, o)| *s *= *o);
        self_tr_upper
            .iter_mut()
            .zip(other_tr_upper.iter())
            .for_each(|(s, o)| *s *= *o);

        let mut res = vec![0.0; 2 * first_histogram.state.len()];
        let mut res_upper = vec![0.0; 2 * first_histogram.state_upper.len()];
        first_histogram
            .ifft
            .process(&mut self_tr, &mut res)
            .unwrap();
        first_histogram
            .ifft
            .process(&mut self_tr_upper, &mut res_upper)
            .unwrap();
        return Self {
            state: res[..first_histogram.state.len()]
                .iter()
                .map(|x| x.round())
                .collect(),
            fft: first_histogram.fft.clone(),
            ifft: first_histogram.ifft.clone(),
            state_upper: res_upper[..first_histogram.state_upper.len()]
                .iter()
                .map(|x| x.round())
                .collect(),
            scale: first_histogram.scale * second_histogram.scale,
        };
    }
    fn coefs_f64(&self) -> Vec<f64> {
        self.state.clone()
    }
    fn coefs_f64_upper(&self) -> Vec<f64> {
        self.state_upper.clone()
    }
    fn from_elems(size: usize, iter: impl Iterator<Item = usize>) -> Self {
        let mut res = Self::new(size);
        for elem in iter {
            if elem < size {
                res.state[elem] += 1.0;
                res.state_upper[elem] += 1.0;
            }
        }
        return res;
    }
    fn scale_back(self) -> Self {
        let new_scale = 1.0;
        let new_state: Vec<f64> = self.state.iter().map(|s| (s * self.scale)).collect();
        let new_upper_state: Vec<f64> = self.state_upper.iter().map(|s| (s * self.scale)).collect();
        return Self {
            state: new_state,
            fft: self.fft.clone(),
            ifft: self.ifft.clone(),
            state_upper: new_upper_state,
            scale: new_scale,
        };
    }
}

#[cfg(feature = "ntl")]
pub struct BigNumHist(*mut bnp::NTL_ZZX, usize);

#[cfg(feature = "ntl")]
impl Histogram for BigNumHist {
    fn new(nb_bins: usize) -> Self {
        return Self(unsafe { bnp::bnp_new_ZZX() }, nb_bins);
    }
    fn type_str(&self) -> &'static str {
        return "BigNumHist";
    }
    fn rescale(&self) -> Self {
        unimplemented!();
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
    fn coefs_f64_upper(&self) -> Vec<f64> {
        unimplemented!()
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
    fn scale_back(self) -> Self {
        self
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
