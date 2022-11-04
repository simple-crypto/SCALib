use ndarray::s;
use super::factor_graph::PublicValue;

type Proba = f64;

/// The minimum non-zero probability (to avoid denormalization, etc.)
const MIN_PROBA: Proba = 1e-40;

#[derive(Debug, Clone)]
enum DistrRepr {
    Uniform,
    Full(ndarray::Array2<Proba>),
}

#[derive(Debug, Clone)]
pub struct Distribution {
    multi: bool,
    shape: (usize, usize),
    value: DistrRepr,
}

impl Distribution {
    pub fn multi(&self) -> bool {
        self.multi
    }
    pub fn new_single(nc: usize) -> Self {
        Self {
            multi: false,
            shape: (nc, 1),
            value: DistrRepr::Uniform,
        }
    }
    pub fn new_multi(nc: usize, nmulti: u32) -> Self {
        Self {
            multi: true,
            shape: (nc, nmulti as usize),
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
        for (mut d, c) in value.axis_iter_mut(ndarray::Axis(0)).zip(cst.as_slice().iter()) {
          d[*c as usize] = 1.0;
        }
        Self {
            multi: self.multi,
            shape: self.shape,
            value: DistrRepr::Full(value),
        }
    }
    pub fn reset(&mut self) {
        self.value = DistrRepr::Uniform;
    }
    pub fn multiply<'a>(&mut self, factors: impl Iterator<Item = &'a Distribution>) {
        for factor in factors {
            assert_eq!(self.shape, factor.shape);
            if let DistrRepr::Full(d) = &factor.value {
                assert_eq!(d.dim(), factor.shape);
                match (self.multi, factor.multi, &mut self.value) {
                    (true, _, DistrRepr::Uniform) | (false, false, DistrRepr::Uniform) => {
                        self.value = DistrRepr::Full(ndarray::Array::from_shape_fn(self.shape, |(_i, j)| d[(0, j)]));
                    }
                    (true, _, DistrRepr::Full(ref mut v)) => {
                        *v *= d;
                    }
                    (false, true, DistrRepr::Uniform) => {
                        let mut v = ndarray::Array2::ones(self.shape);
                        for d in d.axis_iter(ndarray::Axis(0)) {
                            v *= &d.slice(s![ndarray::NewAxis, ..]);
                        }
                        self.value = DistrRepr::Full(v);
                    }
                    (false, _, DistrRepr::Full(ref mut v)) => {
                        for d in d.axis_iter(ndarray::Axis(0)) {
                            *v *= &d.slice(s![ndarray::NewAxis, ..]);
                        }
                    }
                }
            }
        }
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
    pub fn not(&mut self) {
        if let DistrRepr::Full(ref mut array) = &mut self.value {
            todo!()
        }
    }
    pub fn wht(&mut self) {
        todo!()
    }
    pub fn iwht(&mut self) {
        todo!()
    }
    pub fn fft(&mut self) {
        todo!()
    }
    pub fn ifft(&mut self) {
        todo!()
    }
    pub fn regularize(&mut self) {
        todo!()
    }
}
