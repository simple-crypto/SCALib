use super::belief_propagation::{BPError, FftPlans};
use super::factor_graph::PublicValue;
use super::ClassVal;
use ndarray::{azip, s, ArrayViewMut2, Zip};
use realfft::num_complex::Complex;

type Proba = f64;

/// The minimum non-zero probability (to avoid denormalization, etc.)
const MIN_PROBA: Proba = 1e-40;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum DistrRepr {
    Uniform,
    Full(ndarray::Array2<Proba>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Distribution {
    multi: bool,
    /// (nmulti, nc), same as DistrRepr::Full(x) => x.dim()
    shape: (usize, usize),
    value: DistrRepr,
}

impl Distribution {
    pub fn multi(&self) -> bool {
        self.multi
    }
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    pub fn value(&self) -> Option<ndarray::ArrayView2<Proba>> {
        if let DistrRepr::Full(v) = &self.value {
            Some(v.view())
        } else {
            None
        }
    }
    pub fn value_mut(&mut self) -> Option<ndarray::ArrayViewMut2<Proba>> {
        if let DistrRepr::Full(v) = &mut self.value {
            Some(v.view_mut())
        } else {
            None
        }
    }
    pub fn from_array_single(array: ndarray::Array1<Proba>) -> Result<Self, BPError> {
        let l = array.len();
        if array.is_standard_layout() {
            let array = array.into_shape((1, l)).expect("Non-contiguous array");
            Ok(Self {
                multi: false,
                shape: array.dim(),
                value: DistrRepr::Full(array),
            })
        } else {
            Err(BPError::DistributionLayout(
                array.shape().to_owned(),
                array.strides().to_owned(),
            ))
        }
    }
    pub fn from_array_multi(array: ndarray::Array2<Proba>) -> Result<Self, BPError> {
        if array.is_standard_layout() {
            Ok(Self {
                multi: true,
                shape: array.dim(),
                value: DistrRepr::Full(array),
            })
        } else {
            Err(BPError::DistributionLayout(
                array.shape().to_owned(),
                array.strides().to_owned(),
            ))
        }
    }
    pub fn new_single(nc: usize) -> Self {
        Self {
            multi: false,
            shape: (1, nc),
            value: DistrRepr::Uniform,
        }
    }
    pub fn new_multi(nc: usize, nmulti: u32) -> Self {
        Self {
            multi: true,
            shape: (nmulti as usize, nc),
            value: DistrRepr::Uniform,
        }
    }
    pub fn new(multi: bool, nc: usize, nmulti: u32) -> Self {
        if multi {
            Self::new_multi(nc, nmulti)
        } else {
            Self::new_single(nc)
        }
    }
    pub fn as_uniform(&self) -> Self {
        Self {
            multi: self.multi,
            shape: self.shape,
            value: DistrRepr::Uniform,
        }
    }
    pub fn new_constant(&self, cst: &PublicValue) -> Self {
        if let PublicValue::Multi(cst) = cst {
            assert!(self.multi);
            assert_eq!(self.shape.0, cst.len());
        }
        let mut value = ndarray::Array2::zeros(self.shape);
        for (i, mut d) in value.axis_iter_mut(ndarray::Axis(0)).enumerate() {
            let c = match cst {
                PublicValue::Single(x) => *x,
                PublicValue::Multi(x) => x[i],
            };
            d[c as usize] = 1.0;
        }
        Self {
            multi: self.multi,
            shape: self.shape,
            value: DistrRepr::Full(value),
        }
    }
    pub fn reset(&mut self) -> Self {
        let mut new = self.as_uniform();
        std::mem::swap(self, &mut new);
        new
    }

    pub fn take_or_clone(&mut self, reset: bool) -> Self {
        if reset {
            self.reset()
        } else {
            self.clone()
        }
    }

    pub fn multiply<'a>(&mut self, factors: impl Iterator<Item = &'a Distribution>) {
        self.multiply_inner(factors, 0.0, 0.0);
    }

    pub fn multiply_reg<'a>(&mut self, factors: impl Iterator<Item = &'a Distribution>) {
        self.multiply_inner(factors, MIN_PROBA, MIN_PROBA * MIN_PROBA);
    }

    fn multiply_inner<'a>(
        &mut self,
        factors: impl Iterator<Item = &'a Distribution>,
        offset: Proba,
        offset_prod: Proba,
    ) {
        for factor in factors {
            assert_eq!(self.shape.1, factor.shape.1);
            if self.multi & factor.multi {
                assert_eq!(self.shape.0, factor.shape.0);
            }
            if let DistrRepr::Full(d) = &factor.value {
                assert_eq!(d.dim(), factor.shape);
                match (self.multi, factor.multi, &mut self.value) {
                    (_, false, DistrRepr::Uniform) => {
                        self.value = DistrRepr::Full(ndarray::Array::from_shape_fn(
                            self.shape,
                            |(_i, j)| d[(0, j)] + offset,
                        ));
                    }
                    (false, true, DistrRepr::Uniform) => {
                        let mut v = ndarray::Array2::ones(self.shape);
                        for d in d.axis_iter(ndarray::Axis(0)) {
                            azip!(v.slice_mut(s![0, ..]), d).for_each(|v, d| {
                                *v = *v * (*d + offset) + offset_prod;
                            });
                        }
                        self.value = DistrRepr::Full(v);
                    }
                    (true, true, DistrRepr::Uniform) => {
                        self.value = DistrRepr::Full(d.map(|d| *d + offset));
                    }
                    (true, _, DistrRepr::Full(ref mut v)) => {
                        azip!(v, d).for_each(|v, d| {
                            *v = *v * (*d + offset) + offset_prod;
                        });
                    }
                    (false, _, DistrRepr::Full(ref mut v)) => {
                        for d in d.axis_iter(ndarray::Axis(0)) {
                            azip!(v.slice_mut(s![0, ..]), d).for_each(|v, d| {
                                *v = *v * (*d + offset) + offset_prod;
                            });
                        }
                    }
                }
            }
        }
    }
    pub fn divide_reg(state: &Distribution, div: &Distribution) -> Self {
        let mut res = Self {
            multi: state.multi | div.multi,
            shape: (std::cmp::max(state.shape.0, div.shape.0), state.shape.1),
            value: DistrRepr::Uniform,
        };
        assert!(res.shape == state.shape || state.shape == (1, res.shape.1));
        assert!(res.shape == div.shape || div.shape == (1, res.shape.1));
        let one = ndarray::Array2::ones((1, 1));
        let (vst, vdiv) = match (&state.value, &div.value) {
            (DistrRepr::Uniform, DistrRepr::Uniform) => {
                return res;
            }
            (DistrRepr::Uniform, DistrRepr::Full(v)) => (&one, v),
            (DistrRepr::Full(v), DistrRepr::Uniform) => (v, &one),
            (DistrRepr::Full(vst), DistrRepr::Full(vdiv)) => (vst, vdiv),
        };
        res.value = DistrRepr::Full(
            Zip::from(
                vst.broadcast((std::cmp::max(vst.dim().0, vdiv.dim().0), vst.dim().1))
                    .unwrap(),
            )
            .and_broadcast(vdiv)
            .map_collect(|vst, vdiv| *vst / (*vdiv + MIN_PROBA)),
        );
        return res;
    }
    pub fn dividing_full(&mut self, other: &Distribution) {
        match (&mut self.value, &other.value) {
            (DistrRepr::Full(div), DistrRepr::Full(st)) => {
                ndarray::azip!(div, st).for_each(|div, st| *div = *st / *div);
            }
            _ => {
                unimplemented!();
            }
        }
    }
    pub fn not(&mut self) {
        let inv_cst = (self.shape.1 - 1) as u32;
        self.for_each_ignore(|mut d, _| {
            xor_cst_slice(d.as_slice_mut().unwrap(), inv_cst);
        });
    }
    pub fn wht(&mut self) {
        self.for_each_error(|mut d, _| {
            slice_wht(d.as_slice_mut().unwrap());
        });
    }
    pub fn cumt(&mut self) {
        self.for_each_error(|mut d, _| {
            slice_cumt(d.as_slice_mut().unwrap());
        });
    }
    pub fn cumti(&mut self) {
        self.for_each_error(|mut d, _| {
            slice_cumti(d.as_slice_mut().unwrap());
        });
    }
    pub fn opandt(&mut self) {
        self.for_each_error(|mut d, _| {
            slice_opandt(d.as_slice_mut().unwrap());
        });
    }
    pub(super) fn fft_to(
        &self,
        input_scratch: &mut [f64],
        mut dest: ArrayViewMut2<Complex<f64>>,
        fft_scratch: &mut [Complex<f64>],
        plans: &FftPlans,
    ) {
        if let DistrRepr::Full(v) = &self.value {
            for (distr, mut dest) in v.outer_iter().zip(dest.outer_iter_mut()) {
                input_scratch.copy_from_slice(distr.as_slice().unwrap());
                plans
                    .r2c
                    .process_with_scratch(input_scratch, dest.as_slice_mut().unwrap(), fft_scratch)
                    .unwrap();
            }
        }
    }
    pub(super) fn ifft(
        &mut self,
        mut input: ArrayViewMut2<Complex<f64>>,
        fft_scratch: &mut [Complex<f64>],
        plans: &FftPlans,
    ) {
        self.ensure_full();
        let mut v = self.value_mut().unwrap();
        for (mut dest, mut input) in v.outer_iter_mut().zip(input.outer_iter_mut()) {
            plans
                .c2r
                .process_with_scratch(
                    input.as_slice_mut().unwrap(),
                    dest.as_slice_mut().unwrap(),
                    fft_scratch,
                )
                .unwrap();
        }
    }
    pub fn is_full(&self) -> bool {
        match &self.value {
            DistrRepr::Full(_) => true,
            DistrRepr::Uniform => false,
        }
    }
    pub fn ensure_full(&mut self) {
        if let DistrRepr::Uniform = self.value {
            self.value = DistrRepr::Full(ndarray::Array2::from_elem(
                self.shape,
                1.0 / (self.shape.1 as Proba),
            ));
        }
    }
    pub fn full_zeros(&self) -> Self {
        Self {
            multi: self.multi,
            shape: self.shape,
            value: DistrRepr::Full(ndarray::Array2::zeros(self.shape)),
        }
    }
    pub fn map_table(&self, table: &[ClassVal]) -> Self {
        let mut value = ndarray::Array2::zeros(self.shape);
        if let DistrRepr::Full(v_orig) = &self.value {
            for (mut dest, orig) in value.outer_iter_mut().zip(v_orig.outer_iter()) {
                for (i, p) in orig.iter().enumerate() {
                    dest[table[i] as usize] += p;
                }
            }
        } else {
            // TODO optimize for bijective tables
            for mut dest in value.outer_iter_mut() {
                for i in 0..self.shape.1 {
                    dest[table[i] as usize] += 1.0 / (self.shape.1 as f64);
                }
            }
        }
        Self {
            shape: self.shape,
            multi: self.multi,
            value: DistrRepr::Full(value),
        }
    }
    pub fn map_table_inv(&self, table: &[ClassVal]) -> Self {
        let mut value = ndarray::Array2::zeros(self.shape);
        if let DistrRepr::Full(v_orig) = &self.value {
            for (mut dest, orig) in value.outer_iter_mut().zip(v_orig.outer_iter()) {
                for (i, d) in dest.iter_mut().enumerate() {
                    *d = orig[table[i] as usize];
                }
            }
            Self {
                shape: self.shape,
                multi: self.multi,
                value: DistrRepr::Full(value),
            }
        } else {
            // here we are always uniform
            Self {
                shape: self.shape,
                multi: self.multi,
                value: DistrRepr::Uniform,
            }
        }
    }
    /// Normalize sum to one, and make values not too small
    pub fn normalize(&mut self) {
        self.for_each_ignore(|mut d, _| {
            let norm_f = 1.0 / d.sum();
            d.mapv_inplace(|x| x * norm_f);
        })
    }
    /// Normalize sum to one, and make values not too small
    pub fn regularize(&mut self) {
        self.for_each_ignore(|mut d, _| {
            let norm_f = 1.0 / (d.sum() + MIN_PROBA * d.len() as f64);
            d.mapv_inplace(|x| (x + MIN_PROBA) * norm_f);
        })
    }
    pub fn make_non_zero_signed(&mut self) {
        self.for_each_ignore(|mut d, _| {
            d.mapv_inplace(|y| {
                if y.is_sign_positive() {
                    y.max(MIN_PROBA)
                } else {
                    y.min(-MIN_PROBA)
                }
            });
        });
    }
    pub fn xor_cst(&mut self, cst: &PublicValue) {
        self.for_each_ignore(|mut d, i| {
            xor_cst_slice(d.as_slice_mut().unwrap(), cst.get(i));
        });
    }
    pub fn and_cst(&mut self, cst: &PublicValue) {
        self.for_each_ignore(|mut d, i| {
            and_cst_slice(d.as_slice_mut().unwrap(), cst.get(i));
        });
    }
    pub fn inv_and_cst(&mut self, cst: &PublicValue) {
        self.for_each_ignore(|mut d, i| {
            inv_and_cst_slice(d.as_slice_mut().unwrap(), cst.get(i));
        });
    }
    pub fn add_cst(&mut self, cst: &PublicValue, sub: bool) {
        let nc = self.shape.1;
        if let DistrRepr::Full(v) = &mut self.value {
            let mut tmp = vec![0.0f64; nc];
            for (mut d, cst) in v.outer_iter_mut().zip(cst.iter(self.shape.0)) {
                let d = d.as_slice_mut().unwrap();
                let mut cst = usize::try_from(cst).unwrap() % nc;
                if sub {
                    cst = (nc - cst) % nc;
                }
                let size_first_block = self.shape.1 - cst;
                tmp[cst..].copy_from_slice(&d[..size_first_block]);
                tmp[..cst].copy_from_slice(&d[size_first_block..]);
                d.copy_from_slice(tmp.as_slice());
            }
        }
    }
    pub fn op_multiply(&self, other: &Self) -> Self {
        let mut res = self.full_zeros();
        let nc = self.shape().1;
        let u = 1.0f64 / (nc as f64);
        for (k, mut res) in res.value_mut().unwrap().outer_iter_mut().enumerate() {
            for i1 in 0..nc {
                for i2 in 0..nc {
                    let o = (((i1 * i2) as u32) % (nc as u32)) as usize;
                    res[o] += self.value().map(|d| d[(k, i1)]).unwrap_or(u)
                        * other.value().map(|d| d[(k, i2)]).unwrap_or(u);
                }
            }
        }
        res
    }
    pub fn op_multiply_factor(&self, other: &Self) -> Self {
        let mut res = self.full_zeros();
        let nc = self.shape().1;
        let u = 1.0f64 / (nc as f64);
        for (k, mut res) in res.value_mut().unwrap().outer_iter_mut().enumerate() {
            for i1 in 0..nc {
                for i2 in 0..nc {
                    let o = (((i1 * i2) as u32) % (nc as u32)) as usize;
                    res[i1] += self.value().map(|d| d[(k, o)]).unwrap_or(u)
                        * other.value().map(|d| d[(k, i2)]).unwrap_or(u);
                }
            }
        }
        res
    }
    pub fn op_multiply_cst(&self, other: &PublicValue) -> Self {
        let mut res = self.full_zeros();
        let nc = self.shape().1;
        let u = 1.0f64 / (nc as f64);
        for (k, (mut res, i2)) in res
            .value_mut()
            .unwrap()
            .outer_iter_mut()
            .zip(other.iter(self.shape().0))
            .enumerate()
        {
            let i2 = i2 as usize;
            for i1 in 0..nc {
                let o = (((i1 * i2) as u32) % (nc as u32)) as usize;
                res[o] += self.value().map(|d| d[(k, i1)]).unwrap_or(u);
            }
        }
        res
    }
    pub fn op_multiply_cst_factor(&self, other: &PublicValue) -> Self {
        let mut res = self.full_zeros();
        let nc = self.shape().1;
        let u = 1.0f64 / (nc as f64);
        for (k, (mut res, i2)) in res
            .value_mut()
            .unwrap()
            .outer_iter_mut()
            .zip(other.iter(self.shape().0))
            .enumerate()
        {
            let i2 = i2 as usize;
            for i1 in 0..nc {
                let o = (((i1 * i2) as u32) % (nc as u32)) as usize;
                res[i1] += self.value().map(|d| d[(k, o)]).unwrap_or(u);
            }
        }
        res
    }
    pub fn for_each<F, G>(&mut self, mut f: F, default: G)
    where
        F: FnMut(ndarray::ArrayViewMut1<f64>, usize),
        G: FnOnce(&mut Self),
    {
        if let DistrRepr::Full(v) = &mut self.value {
            for (i, d) in v.axis_iter_mut(ndarray::Axis(0)).enumerate() {
                f(d, i);
            }
        } else {
            default(self);
        }
    }
    pub fn for_each_ignore<F>(&mut self, f: F)
    where
        F: FnMut(ndarray::ArrayViewMut1<f64>, usize),
    {
        self.for_each(f, |_| {});
    }
    pub fn for_each_error<F>(&mut self, f: F)
    where
        F: FnMut(ndarray::ArrayViewMut1<f64>, usize),
    {
        self.for_each(f, |_| {
            unimplemented!("This function must be called on Full distributions.");
        });
    }
}

/// Walsh-Hadamard transform (non-normalized).
fn slice_wht(a: &mut [f64]) {
    // The speed of this can be much improved, with the following techiques
    // * improved memory locality
    // * use (auto-)vectorization
    // * generate small static kernels
    let len = a.len();
    let mut h = 1;
    while h < len {
        for mut i in 0..(len / (2 * h) as usize) {
            i *= 2 * h;
            for j in i..(i + h) {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Cumulative transform (U in RLDA paper)
fn slice_cumt(a: &mut [f64]) {
    // Note: the speed of this can probably be much improved, with the following techiques
    // * use (auto-)vectorization
    // * generate small static kernels
    let len = a.len();
    let mut h = 1;
    while h < len {
        for mut i in 0..(len / (2 * h) as usize) {
            i *= 2 * h;
            for j in i..(i + h) {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = y;
            }
        }
        h *= 2;
    }
}

/// Cumulative inverse transform (U^-1 in RLDA paper)
fn slice_cumti(a: &mut [f64]) {
    // Note: the speed of this can probably be much improved, with the following techiques
    // * use (auto-)vectorization
    // * generate small static kernels
    let len = a.len();
    let mut h = 1;
    while h < len {
        for mut i in 0..(len / (2 * h) as usize) {
            i *= 2 * h;
            for j in i..(i + h) {
                let x = a[j];
                let y = a[j + h];
                a[j] = x - y;
                a[j + h] = y;
            }
        }
        h *= 2;
    }
}

/// Tansform for operand of AND (V in RLDA paper), involutive
fn slice_opandt(a: &mut [f64]) {
    // Note: the speed of this can probably be much improved, with the following techiques
    // * use (auto-)vectorization
    // * generate small static kernels
    let len = a.len();
    let mut h = 1;
    while h < len {
        for mut i in 0..(len / (2 * h) as usize) {
            i *= 2 * h;
            for j in i..(i + h) {
                let x = a[j];
                let y = a[j + h];
                a[j] = x;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

fn xor_cst_slice(a: &mut [f64], cst: ClassVal) {
    let leading_zeros = cst.leading_zeros();
    if leading_zeros == 32 {
        return;
    }
    let pivot_bit = ClassVal::BITS - 1 - leading_zeros;
    let step = [1, 1 << (pivot_bit + 1) as usize];
    let n_max = [1 << pivot_bit as usize, a.len()];
    let n_iter = [n_max[0] / step[0], n_max[1] / step[1]];
    let outer_i = if n_iter[0] < n_iter[1] { 0 } else { 1 };
    for i in (0..n_max[outer_i]).step_by(step[outer_i]) {
        for j in (0..n_max[1 - outer_i]).step_by(step[1 - outer_i]) {
            let idx = i + j;
            a.swap(idx, idx ^ cst as usize);
        }
    }
}
fn and_cst_slice(a: &mut [f64], cst: ClassVal) {
    for i in 0..a.len() {
        let j = i & (cst as usize);
        if j != i {
            a[j] += a[i];
            a[i] = 0.0;
        }
    }
}
fn inv_and_cst_slice(a: &mut [f64], cst: ClassVal) {
    for i in 0..a.len() {
        let j = i & (cst as usize);
        a[i] = a[j];
    }
}
