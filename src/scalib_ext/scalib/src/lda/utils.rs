pub(crate) fn log2_softmax_i(v: ndarray::ArrayView1<f64>, i: usize) -> f64 {
    let max = v.fold(f64::NEG_INFINITY, |x, y| f64::max(x, *y));
    use std::f64::consts::LOG2_E;
    (v[i] - max) * LOG2_E - f64::log2(v.iter().map(|x| (x - max).exp()).sum())
}

pub(crate) trait ArrayBaseExt<A, D> {
    fn clone_row_major(&self) -> ndarray::Array<A, D>;
}
impl<A, S, D> ArrayBaseExt<A, D> for ndarray::ArrayBase<S, D>
where
    A: Clone,
    S: ndarray::RawData<Elem = A> + ndarray::Data,
    D: ndarray::Dimension,
{
    fn clone_row_major(&self) -> ndarray::Array<A, D> {
        let res = ndarray::Array::<A, D>::build_uninit(self.dim(), |a| {
            self.assign_to(a);
        });
        unsafe { res.assume_init() }
    }
}
