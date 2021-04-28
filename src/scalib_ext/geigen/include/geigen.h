#ifndef _GEIGEN_H_
#define _GEIGEN_H_

#include <memory>
#include "geigen/include/Eigen/Dense"
#include "geigen/include/Eigen/Eigenvalues"
#include "rust/cxx.h"

using Eigen::MatrixXd;

typedef Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> GEigenSolver;

// include this only after the GEigenSolver typedef.
#include "geigen/src/geigen.rs.h"

std::unique_ptr<GEigenSolver> solve_geigen(Matrix a, Matrix b);
Matrix get_eigenvecs(const GEigenSolver& solver);
rust::Slice<const double> get_eigenvals(const GEigenSolver& solver);

#endif // _GEIGEN_H_
