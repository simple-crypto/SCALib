use cxx::UniquePtr;
use ndarray::{s, ArrayView1, ArrayView2};

mod geigen;

/// Generalized eigenvalues solver A*x = lambda*B*x
/// A and B must be symmetric, B must be positive-definite.
/// Solve for only the n largest eigenvalues and associated eigenvectors.
/// Eigenvalues (and associated eigenvectors) are sorted by decreasing eigenvalue.
pub trait Geigen: Sized {
    type Error: std::fmt::Debug;
    fn new(a: &ArrayView2<f64>, b: &ArrayView2<f64>, n: usize) -> Result<Self, Self::Error>;
    fn vecs(&self) -> ArrayView2<f64>;
    fn vals(&self) -> ArrayView1<f64>;
}

/// Generalized eigendecomposition solver for symmetric matrices cases
/// (a must be symmetric, and b must be positive-definite).
/// Solver based on Eigen library.
pub struct GEigenSolver {
    solver: UniquePtr<geigen::GEigenSolver>,
    n: usize,
}

impl Geigen for GEigenSolver {
    type Error = ();
    fn new(
        a: &ndarray::ArrayView2<f64>,
        b: &ndarray::ArrayView2<f64>,
        n: usize,
    ) -> Result<Self, Self::Error> {
        let a = geigen::array2matrix(a);
        let b = geigen::array2matrix(b);
        // SAFETY: a and b are valid matrices (as generated above) and their lifetime covers this
        // call.
        //
        let solver = unsafe { geigen::solve_geigen(a, b) };
        Ok(Self { solver, n })
    }
    fn vecs(&self) -> ndarray::ArrayView2<f64> {
        // SAFETY: result of eigen::get_eigenvecs satisfies the properties of matrix2array_view.
        let all_vecs = unsafe { geigen::matrix2array(geigen::get_eigenvecs(&self.solver)) }
            .expect("Eigenvectors matrix must have positive dimensions and strides.");
        all_vecs
            .clone()
            .slice_move(s![.., -(self.n as isize)..; -1])
    }
    fn vals(&self) -> ArrayView1<f64> {
        let vals = geigen::get_eigenvals(&self.solver);
        let vals: ArrayView1<f64> = vals[vals.len() - self.n..].into();
        vals.slice_move(s![..; -1])
    }
}

/// Generalized eigendecomposition solver for symmetric matrices cases
/// (a must be symmetric, and b must be positive-definite).
/// Solver based on Spectra library.
pub struct GEigenSolverP {
    solver: UniquePtr<geigen::GEigenPR>,
}

use thiserror::Error;
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum GeigenError {
    #[error("Computation has not been performed. Missing call to compute() at low-level.")]
    CholeskyNotComputed,
    #[error("The eigenvalues computation did not converged during the Cholesky decomposition.")]
    CholeskyNotConverging,
    #[error("The matrix used in the Cholesky decomposition is not positive definite.")]
    CholeskyNumericalIssue,
    #[error("Computation has not been performed. Missing call to compute() at low-level.")]
    EigenNotComputed,
    #[error("The eigenvalues computation did not converged during the eigenvalues computation.")]
    EigenNotConverging,
    #[error("A numerical issue occured during the eigenvalues computation.")]
    EigenNumericalIssue,
    #[error(
        "Tridiag decomposition failed. Likely cause: singular leakage traces matrix. \
        Add more traces or make them (slightly) more noisy."
    )]
    TridiagDecompositionFailed,
}

impl Geigen for GEigenSolverP {
    type Error = GeigenError;
    fn new(
        a: &ndarray::ArrayView2<f64>,
        b: &ndarray::ArrayView2<f64>,
        n: usize,
    ) -> Result<Self, Self::Error> {
        // ncv >= 2*nev is recommended by documentation mandatory: n < ncv <= matrix size
        // Did not work for at least one example with nev == 1, so increaded a bit while not
        // making it (too) much slower for big 'nev'.
        let ncv = std::cmp::min(5 + 2 * n, a.shape()[0]) as u32;
        let n = n as u32;
        let a = geigen::array2matrix(a);
        let b = geigen::array2matrix(b);
        let mut err: u32 = 0;
        // SAFETY: a and b are valid matrices (as generated above) and their lifetime covers this
        // call.
        let solver = unsafe { geigen::solve_geigenp(a, b, n, ncv, &mut err) };
        match err {
            0 => Ok(Self { solver }),
            1 => Err(GeigenError::CholeskyNotComputed),
            2 => Err(GeigenError::CholeskyNotConverging),
            3 => Err(GeigenError::CholeskyNumericalIssue),
            4 => Err(GeigenError::EigenNotComputed),
            5 => Err(GeigenError::EigenNotConverging),
            6 => Err(GeigenError::EigenNumericalIssue),
            7 => Err(GeigenError::TridiagDecompositionFailed),
            _ => unreachable!("Invalid error code for geigen::solve_geigenp"),
        }
    }
    fn vecs(&self) -> ArrayView2<f64> {
        // SAFETY: result of eigen::get_eigenvecs_p satisfies the properties of matrix2array_view.
        unsafe { geigen::matrix2array(geigen::get_eigenvecs_p(&self.solver)) }
            .expect("Eigenvectors matrix must have positive dimensions and strides.")
    }
    fn vals(&self) -> ArrayView1<f64> {
        geigen::get_eigenvals_p(&self.solver).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::RandomExt;
    use rand_xoshiro::Xoshiro256StarStar;
    fn test_solver<S: Geigen>(a: &ArrayView2<f64>, b: &ArrayView2<f64>, n: usize) {
        let solver = S::new(&a.view(), &b.view(), n).expect("Fail to solve");
        let vecs = solver.vecs();
        let vals = solver.vals();
        println!("vecs:\n{:#?}", vecs);
        println!("vals:\n{:#?}", vals);
        assert_eq!(vecs.shape(), [a.shape()[0], n]);
        assert_eq!(vals.shape(), [n,]);
        let vals = ndarray::Array2::from_diag(&vals);
        let av = a.dot(&vecs);
        let bv = b.dot(&vecs).dot(&vals);
        assert!(
            av.relative_eq(&bv, 1e-14, 1e-8),
            "Check eq failed.\nav:\n{:#?},\nbv:\n{:#?}",
            av,
            bv
        );
    }
    fn gen_problem(n: usize) -> (Array2<f64>, Array2<f64>) {
        // Get a seeded random number generator for reproducibility
        let seed = 42;
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let x: ndarray::Array2<f64> = ndarray::Array2::random_using(
            (n, n),
            ndarray_rand::rand_distr::Uniform::from(0.0..1.0),
            &mut rng,
        );
        let y: ndarray::Array2<f64> = ndarray::Array2::random_using(
            (n, n),
            ndarray_rand::rand_distr::Uniform::from(0.0..1.0),
            &mut rng,
        );
        // make it symmetric
        let a = &x + &x.t();
        // make it semipositive-definite
        let b = y.dot(&y.t());
        return (a, b);
    }
    #[test]
    fn test_geigensolve() {
        let (a, b) = gen_problem(5);
        for i in 1..=5 {
            test_solver::<super::GEigenSolver>(&a.view(), &b.view(), i);
        }
    }
    #[test]
    fn test_geigensolve_p() {
        let (a, b) = gen_problem(5);
        for i in 1..=4 {
            test_solver::<super::GEigenSolverP>(&a.view(), &b.view(), i);
        }
    }
}
