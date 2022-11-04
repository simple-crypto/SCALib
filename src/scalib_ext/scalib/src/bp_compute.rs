use ndarray::s;

type Proba = f64;

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
    pub fn as_cst(&self, cst: usize) -> Self {
        let mut value = ndarray::Array2::zeros(self.shape);
        value.slice_mut(s!(.., cst)).mapv_inplace(|_| 1.0);
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
        todo!()
    }
    pub fn divide<'a>(&mut self, state: &'a Distribution, div: &'a Distribution) {
        todo!()
    }
    pub fn not(&mut self) {
        if let DistrRepr::Full(mut array) = &mut self.value {
            todo!()
        }
    }
    pub fn wht(&mut self) {
        todo!()
    }
}
