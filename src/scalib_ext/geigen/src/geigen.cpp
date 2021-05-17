#include <memory>
#include <iostream>
#include "Eigen/Eigenvalues"
#include "Eigen/Dense"
#include "geigen.h"


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::Map;
using Eigen::Stride;
using Eigen::AlignmentType;
using Eigen::Dynamic;

typedef Map<const MatrixXd, AlignmentType::Unaligned, Stride<Dynamic, Dynamic> > MatrixMap;
typedef Spectra::SymGEigsSolver<Spectra::DenseSymMatProd<double>, Spectra::DenseCholesky<double>, Spectra::GEigsMode::Cholesky> GEigenSolverP;

static MatrixMap matrix2map(Matrix matrix);
static Matrix m2m(const MatrixXd& matrix);
static rust::Slice<const double> v2s(const VectorXd& vec);

std::unique_ptr<GEigenSolver> solve_geigen(Matrix a, Matrix b) {
    MatrixMap am = matrix2map(a);
    MatrixMap bm = matrix2map(b);
    GEigenSolver es(am,bm);
    return std::make_unique<GEigenSolver>(es);
}

Matrix get_eigenvecs(const GEigenSolver& solver) {
    return m2m(solver.eigenvectors());
}

rust::Slice<const double> get_eigenvals(const GEigenSolver& solver) {
    return v2s(solver.eigenvalues());
}

std::unique_ptr<GEigenPR> solve_geigenp(Matrix a, Matrix b, uint32_t nev, uint32_t ncv, uint32_t& err) {
    MatrixMap am = matrix2map(a);
    MatrixMap bm = matrix2map(b);
    Spectra::DenseSymMatProd<double> op(am);
    Spectra::DenseCholesky<double>  Bop(bm);
    // Construct generalized eigen solver object, requesting the largest three generalized eigenvalues
    GEigenSolverP geigs(op, Bop, nev, ncv);

    // Initialize and compute
    geigs.init();
    geigs.compute(Spectra::SortRule::LargestAlge);
    if (geigs.info() == Spectra::CompInfo::Successful) {
        err = 0;
    } else {
        err = 1;
    }
    return std::make_unique<GEigenPR>(GEigenPR(geigs.eigenvectors(), geigs.eigenvalues()));
}

Matrix get_eigenvecs_p(const GEigenPR& solver) {
    return m2m(std::get<0>(solver));
}

rust::Slice<const double> get_eigenvals_p(const GEigenPR& solver) {
    return v2s(std::get<1>(solver));
}

static MatrixMap matrix2map(Matrix matrix) {
    Stride<Dynamic, Dynamic> stride(matrix.col_stride, matrix.row_stride);
    MatrixMap map(
            matrix.data,
            matrix.rows,
            matrix.cols,
            stride
            );
    return map;
}

static Matrix m2m(const MatrixXd& matrix) {
    Matrix res;
    res.data= matrix.data();
    res.rows= matrix.rows();
    res.cols= matrix.cols();
    res.row_stride= matrix.rowStride();
    res.col_stride= matrix.colStride();
    return res;
}

static rust::Slice<const double> v2s(const VectorXd& vec) {
    return rust::Slice<const double>(vec.data(), vec.size());
}
