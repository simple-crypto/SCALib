#ifndef _GEIGEN_H_
#define _GEIGEN_H_

#include <memory>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/Eigenvalues"
#include "Spectra/SymGEigsSolver.h"
#include "Spectra/MatOp/DenseSymMatProd.h"
#include "Spectra/MatOp/DenseCholesky.h"
#include "rust/cxx.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> GEigenSolver;
typedef std::tuple<MatrixXd, VectorXd> GEigenPR;

// include this only after the GEigenSolver typedef.
#include "geigen/src/geigen.rs.h"

std::unique_ptr<GEigenSolver> solve_geigen(Matrix a, Matrix b);
std::unique_ptr<GEigenPR> solve_geigenp(Matrix a, Matrix b, uint32_t nev, uint32_t ncv, uint32_t& err);
Matrix get_eigenvecs(const GEigenSolver& solver);
rust::Slice<const double> get_eigenvals(const GEigenSolver& solver);
Matrix get_eigenvecs_p(const GEigenPR& solver);
rust::Slice<const double> get_eigenvals_p(const GEigenPR& solver);

#endif // _GEIGEN_H_
