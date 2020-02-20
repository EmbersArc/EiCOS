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
              const Eigen::SparseVector<double> &c,
              const Eigen::SparseVector<double> &h,
              const Eigen::SparseVector<double> &b,
              const std::vector<size_t> &soc_dims);

    void Solve();

private:
    size_t iteration;
    size_t max_iterations = 100;

    Eigen::SparseMatrix<double> G;
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> At;
    Eigen::SparseMatrix<double> Gt;
    Eigen::SparseVector<double> c;
    Eigen::SparseVector<double> h;
    Eigen::SparseVector<double> b;
    std::vector<size_t> soc_dims;

    size_t num_var;  // Number of variables
    size_t num_eq;   // Number of equality constraints (p)
    size_t num_ineq; // Number of inequality constraints (m)
    size_t num_pc;   // Number of positive constraints (l)
    size_t num_sc;   // Number of second order cone constraints (ncones)

    Eigen::VectorXd rhs1, rhs2; // The two right hand sides in the KKT equations.

    // The problem data scaling parameters
    double scale_rx, scale_ry, scale_rz;
    double resx0, resy0, resz0;

    const double delta_reg = 7e-8; // Static Regularization Parameter

    Eigen::SparseMatrix<double> K;
    void SetupKKT(const Eigen::SparseMatrix<double> &G,
                  const Eigen::SparseMatrix<double> &A,
                  const std::vector<size_t> &soc_dims);

    void bringToCone(Eigen::VectorXd &x);
    void computeResiduals();
    void updateStatistics();
};