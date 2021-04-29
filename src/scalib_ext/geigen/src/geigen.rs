//! This module contains the cxx bridge to Eigen.
//! Do not add too many rust functions, as it slows down development (requires to re-compile all
//! the C++ everytime this file is changed).

pub use ffi::*;
use ndarray::ShapeBuilder;
use std::convert::TryInto;

#[cxx::bridge]
mod ffi {
    /// FFI-safe representation of a matrix.
    /// Does not include lifetime as this is not supported by cxx,
    /// but conceptually Matrix<'a> should contain data: &'a [f64].
    /// A matrix is valid is data+(i*row_stride+j*col_stride) points to a valid f64 for lifetime
    /// 'a, for any 0 <= i < rows, 0 <= j < cols.
    struct Matrix {
        data: *const f64,
        rows: isize,
        cols: isize,
        row_stride: isize,
        col_stride: isize,
    }
    unsafe extern "C++" {
        include!("geigen.h");
        /// Full generalized symmetric eigensolver.
        type GEigenSolver;
        // SAFETY: a and b are required to live only for the duration of the call,
        // and a and b must be valid Matrix data.
        unsafe fn solve_geigen(a: Matrix, b: Matrix) -> UniquePtr<GEigenSolver>;
        // Matrix return type implicit lifetime is 'a.
        // SAFETY: consequence of the properties of GEigenSolver
        fn get_eigenvecs<'a>(solver: &'a GEigenSolver) -> Matrix;
        // SAFETY: consequence of the properties of GEigenSolver
        fn get_eigenvals<'a>(solver: &'a GEigenSolver) -> &'a [f64];

        /// Partial generalized symmetric eigensolver.
        type GEigenPR;
        // SAFETY: a and b are required to live only for the duration of the call,
        // and a and b must be valid Matrix data.
        unsafe fn solve_geigenp(
            a: Matrix,
            b: Matrix,
            nev: u32,
            ncv: u32,
            err: &mut u32,
        ) -> UniquePtr<GEigenPR>;
        // Matrix return type implicit lifetime is 'a.
        // SAFETY: consequence of the properties of GEigenSolver
        fn get_eigenvecs_p<'a>(solver: &'a GEigenPR) -> Matrix;
        // SAFETY: consequence of the properties of GEigenSolver
        fn get_eigenvals_p<'a>(solver: &'a GEigenPR) -> &'a [f64];
    }
}

/// Converts a Matrix into a ndarray::ArrayView2<f64>
/// SAFETY: matrix must have lifetime 'a.
pub unsafe fn matrix2array<'a>(
    matrix: Matrix,
) -> Result<ndarray::ArrayView2<'a, f64>, std::num::TryFromIntError> {
    Ok(ndarray::ArrayView::from_shape_ptr(
        (matrix.rows.try_into()?, matrix.cols.try_into()?)
            .strides((matrix.row_stride.try_into()?, matrix.col_stride.try_into()?)),
        matrix.data,
    ))
}

/// Converts a ndarray::ArrayView2<f64> into a Matrix
pub fn array2matrix(a: &ndarray::ArrayView2<f64>) -> Matrix {
    let mut axes = a.axes();
    let ax0 = axes.next().unwrap();
    let ax1 = axes.next().unwrap();
    assert_eq!(ax0.axis.0, 0);
    assert_eq!(ax1.axis.0, 1);
    Matrix {
        data: a.as_ptr(),
        rows: ax0.len.try_into().unwrap(),
        cols: ax1.len.try_into().unwrap(),
        row_stride: ax0.stride,
        col_stride: ax1.stride,
    }
}
