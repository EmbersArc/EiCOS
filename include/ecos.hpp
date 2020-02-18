#pragma once

#include <Eigen/Sparse>

class ECOSEigen
{
    // n:       Number of variables.
    // m:       Number of inequality constraints.
    // p:       Number of equality constraints.
    // l:       The dimension of the positive orthant, i.e. in Gx+s=h, s in K.
    // The first l elements of s are >=0, ncones is the number of second-order cones present in K.
    // ncones:  Number of second order cones in K.
    // q:       Vector of dimesions of each cone constraint in K.

    // A(p,n):  Equality constraint matrix.
    // b(p):    Equality constraint vector.
    // G(m,n):  Generalized inequality matrix.
    // h(m):    Generalized inequality vector.
    // c(n):    Variable weights.

    ECOSEigen(const Eigen::SparseMatrix<double> &G,
              const Eigen::SparseMatrix<double> &A,
              const Eigen::SparseMatrix<double> &c,
              const Eigen::SparseMatrix<double> &h,
              const Eigen::SparseMatrix<double> &b,
              const std::vector<size_t> &soc_dims);

private:
    size_t num_var;
    size_t num_eq;   // number of equality constraints (p)
    size_t num_ineq; // number of inequality constraints (m)
    size_t num_pc;   // number of positive constraints (l)
    size_t num_sc;   // number of second order cone constraints (ncones)

    const double delta_reg = 7e-8; // Static Regularization

    Eigen::SparseMatrix<double> K;
    void SetupKKT(const Eigen::SparseMatrix<double> &G,
                  const Eigen::SparseMatrix<double> &A,
                  const std::vector<size_t> &soc_dims);
};