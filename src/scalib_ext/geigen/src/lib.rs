use cxx::UniquePtr;

mod geigen;

/// Generalized eigendecomposition solver for symmetric matrices cases
/// (a must be symmetric, and b must be positive-definite).
pub struct GEigenSolver(UniquePtr<geigen::GEigenSolver>);

impl GEigenSolver {
    pub fn new(a: &ndarray::ArrayView2<f64>, b: &ndarray::ArrayView2<f64>) -> Self {
        let a = geigen::array2matrix(a);
        let b = geigen::array2matrix(b);
        // SAFETY: a and b are valid matrices (as generated above) and their lifetime covers this
        // call.
        Self(unsafe { geigen::solve_geigen(a, b) })
    }
    pub fn get_eigenvecs(&self) -> ndarray::ArrayView2<f64> {
        // SAFETY: result of eigen::get_eigenvecs satisfies the properties of matrix2array_view.
        unsafe { geigen::matrix2array(geigen::get_eigenvecs(&self.0)) }
            .expect("Eigenvectors matrix must have positive dimensions and strides.")
    }
    pub fn get_eigenvals(&self) -> &[f64] {
        geigen::get_eigenvals(&self.0)
    }
    pub fn get_eigenvals_array(&self) -> ndarray::ArrayView1<f64> {
        let vals = self.get_eigenvals();
        ndarray::ArrayView1::from_shape((vals.len(),), vals).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ndarray_rand::RandomExt;
    #[test]
    fn test_geigensolve() {
        let x: ndarray::Array2<f64> =
            ndarray::Array2::random((5, 5), ndarray_rand::rand_distr::Uniform::from(0.0..1.0));
        let y: ndarray::Array2<f64> =
            ndarray::Array2::random((5, 5), ndarray_rand::rand_distr::Uniform::from(0.0..1.0));
        // make it symmetric
        let a = &x + &x.t();
        // make it semipositive-definite
        let b = y.dot(&y.t());
        let solver = super::GEigenSolver::new(&a.view(), &b.view());
        let vecs = solver.get_eigenvecs();
        println!("vecs: {:#?}", vecs);
        println!("vals: {:#?}", solver.get_eigenvals());
        let vals = ndarray::Array2::from_diag(&solver.get_eigenvals_array());
        let av = a.dot(&vecs);
        let bv = b.dot(&vecs).dot(&vals);
        assert!(
            av.relative_eq(&bv, 1e-14, 1e-8),
            "av: {:#?}, bv: {:#?}",
            av,
            bv
        );
    }
}
