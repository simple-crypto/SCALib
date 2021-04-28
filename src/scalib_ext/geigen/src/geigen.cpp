#include <memory>
#include <iostream>
#include "geigen/include/Eigen/Eigenvalues"
#include "geigen/include/Eigen/Dense"
#include "geigen/include/geigen.h"


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::Map;
using Eigen::Stride;
using Eigen::AlignmentType;
using Eigen::Dynamic;

typedef Map<const MatrixXd, AlignmentType::Unaligned, Stride<Dynamic, Dynamic> > MatrixMap;

Matrix get_eigenvecs(const GEigenSolver& solver) {
    const MatrixXd& ev = solver.eigenvectors();
    return Matrix {
        data: ev.data(),
        rows: ev.rows(),
        cols: ev.cols(),
        row_stride: ev.rowStride(),
        col_stride: ev.colStride(),
    };
}

rust::Slice<const double> get_eigenvals(const GEigenSolver& solver) {
    const VectorXd& ev = solver.eigenvalues();
    return rust::Slice<const double>(ev.data(), ev.size());
}

MatrixMap matrix2map(Matrix matrix) {
    Stride<Dynamic, Dynamic> stride(matrix.col_stride, matrix.row_stride);
    MatrixMap map(
            matrix.data,
            matrix.rows,
            matrix.cols,
            stride
            );
    return map;
}

std::unique_ptr<GEigenSolver> solve_geigen(Matrix a, Matrix b) {
    MatrixMap am = matrix2map(a);
    MatrixMap bm = matrix2map(b);
    GeneralizedSelfAdjointEigenSolver<MatrixXd> es(am,bm);
    return std::make_unique<GEigenSolver>(es);
}

std::unique_ptr<GEigenSolver> solve_dummy() {
    MatrixXd X = MatrixXd::Random(2,2);
    MatrixXd A = X + X.transpose();
    cout << "Here is a random symmetric matrix, A:" << endl << A << endl;
    X = MatrixXd::Random(2,2);
    MatrixXd B = X * X.transpose();
    cout << "and a random postive-definite matrix, B:" << endl << B << endl << endl;
    GeneralizedSelfAdjointEigenSolver<MatrixXd> es(A,B);
    cout << "The eigenvalues of the pencil (A,B) are:" << endl << es.eigenvalues() << endl;
    cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;
    return std::make_unique<GEigenSolver>(es);
}

void show_eigen() {
    auto es = solve_dummy();

    double lambda = es->eigenvalues()[0];
    cout << "Consider the first eigenvalue, lambda = " << lambda << endl;
    VectorXd v = es->eigenvectors().col(0);
    //cout << "If v is the corresponding eigenvector, then A * v = " << endl << A * v << endl;
    //cout << "... and lambda * B * v = " << endl << lambda * B * v << endl << endl;
}


