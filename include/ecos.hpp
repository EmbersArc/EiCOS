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

    ECOSEigen(Eigen::SparseMatrix<double> &G,
              Eigen::SparseMatrix<double> &A,
              Eigen::SparseMatrix<double> &c,
              Eigen::SparseMatrix<double> &h,
              Eigen::SparseMatrix<double> &b,
              std::vector<size_t> &soc_dims);

private:
    size_t num_eq; // number of equality constraints (p)
    size_t num_po; // number of positive constraints (l)
    size_t num_so; // number of second order cone constraints (ncones)
    Eigen::SparseMatrix<double> K;
    void SetupKKT(Eigen::SparseMatrix<double> &G,
                  Eigen::SparseMatrix<double> &A,
                  Eigen::SparseMatrix<double> &c,
                  std::vector<size_t> &soc_dims);
};