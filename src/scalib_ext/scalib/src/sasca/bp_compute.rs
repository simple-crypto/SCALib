use super::factor_graph::PublicValue;
use ndarray::s;

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
    pub fn from_array_single(array: ndarray::Array1<Proba>) -> Self {
        let l = array.len();
        let array = array.into_shape((1, l)).expect("Non-contiguous array");
        Self {
            multi: false,
            shape: array.dim(),
            value: DistrRepr::Full(array),
        }
    }
    pub fn from_array_multi(array: ndarray::Array2<Proba>) -> Self {
        Self {
            multi: true,
            shape: array.dim(),
            value: DistrRepr::Full(array),
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
            assert_eq!(nmulti, 1);
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
        let mut value = ndarray::Array2::zeros(self.shape);
        for (mut d, c) in value
            .axis_iter_mut(ndarray::Axis(0))
            .zip(cst.as_slice().iter())
        {
            d[*c as usize] = 1.0;
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
    pub fn multiply<'a>(&mut self, factors: impl Iterator<Item = &'a Distribution>) {
        for factor in factors {
            dbg!(&self);
            dbg!(factor);
            assert_eq!(self.shape, factor.shape);
            if let DistrRepr::Full(d) = &factor.value {
                assert_eq!(d.dim(), factor.shape);
                match (self.multi, factor.multi, &mut self.value) {
                    (_, false, DistrRepr::Uniform) => {
                        self.value = DistrRepr::Full(ndarray::Array::from_shape_fn(
                            self.shape,
                            |(_i, j)| d[(0, j)],
                        ));
                    }
                    (false, true, DistrRepr::Uniform) => {
                        let mut v = ndarray::Array2::ones(self.shape);
                        for d in d.axis_iter(ndarray::Axis(0)) {
                            v *= &d.slice(s![ndarray::NewAxis, ..]);
                        }
                        self.value = DistrRepr::Full(v);
                    }
                    (true, true, DistrRepr::Uniform) => {
                        self.value = DistrRepr::Full(d.to_owned());
                    }
                    (true, _, DistrRepr::Full(ref mut v)) => {
                        *v *= d;
                    }
                    (false, _, DistrRepr::Full(ref mut v)) => {
                        for d in d.axis_iter(ndarray::Axis(0)) {
                            *v *= &d.slice(s![ndarray::NewAxis, ..]);
                        }
                    }
                }
            }
        }
        dbg!(self);
    }
    pub fn divide(state: &Distribution, div: &Distribution) -> Self {
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
        res.value = DistrRepr::Full(vst / vdiv);
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
        if let DistrRepr::Full(ref mut array) = &mut self.value {
            todo!()
        }
    }
    pub fn wht(&mut self) {
        self.for_each_error(|mut d| {
            slice_wht(d.as_slice_mut().unwrap());
        });
    }
    pub fn fft(&mut self) {
        todo!()
    }
    pub fn ifft(&mut self) {
        todo!()
    }
    pub fn is_full(&self) -> bool {
        match &self.value {
            DistrRepr::Full(_) => true,
            DistrRepr::Uniform => false,
        }
    }
    pub fn regularize(&mut self) {
        self.for_each_ignore(|mut d| {
            let norm_f = 1.0 / (d.sum() + MIN_PROBA * d.len() as f64);
            d.mapv_inplace(|x| (x + MIN_PROBA) * norm_f);
        })
    }
    pub fn make_non_zero_signed(&mut self) {
        self.for_each_ignore(|mut d| {
            d.mapv_inplace(|y| {
                if y.is_sign_positive() {
                    y.max(MIN_PROBA)
                } else {
                    y.min(-MIN_PROBA)
                }
            });
        });
    }
    fn for_each<F, G>(&mut self, f: F, default: G)
    where
        F: Fn(ndarray::ArrayViewMut1<f64>),
        G: Fn(&mut Self),
    {
        if let DistrRepr::Full(v) = &mut self.value {
            for d in v.axis_iter_mut(ndarray::Axis(0)) {
                f(d);
            }
        } else {
            default(self);
        }
    }
    fn for_each_ignore<F>(&mut self, f: F)
    where
        F: Fn(ndarray::ArrayViewMut1<f64>),
    {
        self.for_each(f, |_| {});
    }
    fn for_each_error<F>(&mut self, f: F)
    where
        F: Fn(ndarray::ArrayViewMut1<f64>),
    {
        self.for_each(f, |_| {
            unimplemented!();
        });
    }
}

/// Walsh-Hadamard transform (non-normalized).
#[inline(always)]
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
