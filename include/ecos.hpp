#pragma once

#include <Eigen/Sparse>

struct Settings
{
    double gamma;         // scaling the final step length
    double delta;         // regularization parameter
    double eps;           // regularization threshold
    double feastol;       // primal/dual infeasibility tolerance
    double abstol;        // absolute tolerance on duality gap
    double reltol;        // relative tolerance on duality gap
    double feastol_inacc; // primal/dual infeasibility relaxed tolerance
    double abstol_inacc;  // absolute relaxed tolerance on duality gap
    double reltol_inacc;  // relative relaxed tolerance on duality gap
    size_t nitref;        // number of iterative refinement steps
    size_t maxit;         // maximum number of iterations
    size_t verbose;       // verbosity bool for PRINTLEVEL < 3
    double linsysacc;     // rel. accuracy of search direction
    double irerrfact;     // factor by which IR should reduce err
};

struct Information
{
    double pcost;
    double dcost;
    double pres;
    double dres;
    double pinf;
    double dinf;
    std::optional<double> pinfres;
    std::optional<double> dinfres;
    double gap;
    double relgap;
    double sigma;
    double mu;
    double step;
    double step_aff;
    double kapovert;
    size_t iter;
    size_t iter_max;
    size_t nitref1;
    size_t nitref2;
    size_t nitref3;
};

struct PositiveCone
{
    size_t dim;
    Eigen::VectorXd w;
    Eigen::VectorXd v;
    Eigen::VectorXi kkt_idx;
};

struct SecondOrderCone
{
    size_t dim;            // dimension of cone
    Eigen::VectorXd skbar; // temporary variables to work with
    Eigen::VectorXd zkbar; // temporary variables to work with
    double a;              // = wbar(1)
    double d1;             // first element of D
    double w;              // = q'*q
    double eta;            // eta = (sres / zres)^(1/4)
    double eta_square;     // eta^2 = (sres / zres)^(1/2)
    Eigen::VectorXd q;     // = wbar(2:end)
    Eigen::VectorXi Didx;  // indices for D
    double u0;             // eta
    double u1;             // u = [u0; u1*q]
    double v1;             // v = [0; v1*q]
};

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
    PositiveCone lp_cone;
    std::vector<SecondOrderCone> so_cones;
    Settings settings;
    Information info, best_info;

    size_t iteration;

    Eigen::SparseMatrix<double> G;
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> At;
    Eigen::SparseMatrix<double> Gt;
    Eigen::SparseVector<double> c;
    Eigen::SparseVector<double> h;
    Eigen::SparseVector<double> b;

    Eigen::VectorXd x;      // Primal variables                     size n
    Eigen::VectorXd y;      // Multipliers for equality constaints  size p
    Eigen::VectorXd z;      // Multipliers for conic inequalities   size m
    Eigen::VectorXd s;      // Slacks for conic inequalities        size m
    Eigen::VectorXd lambda; // Scaled variable                      size m

    // Residuals
    Eigen::VectorXd rx, ry, rz; // sizes n, p, m
    double hresx, hresy, hresz;

    // Norm iterates
    double nx, ny, nz, ns;

    Eigen::SparseVector<double> x_equil;
    Eigen::SparseMatrix<double> G_equil;
    Eigen::SparseMatrix<double> A_equil;

    size_t num_var;  // Number of variables
    size_t num_eq;   // Number of equality constraints (p)
    size_t num_ineq; // Number of inequality constraints (m)
    size_t num_pc;   // Number of positive constraints (l)
    size_t num_sc;   // Number of second order cone constraints (ncones)
    size_t D;        // Degree of the cone

    Eigen::VectorXd rhs1, rhs2; // The two right hand sides in the KKT equations.

    double kap; // kappa (homogeneous embedding)
    double tau; // tau (homogeneous embedding)

    // The problem data scaling parameters
    double scale_rx, scale_ry, scale_rz;
    double resx0, resy0, resz0;
    double cx, by, hz;

    // KKT Matrix
    Eigen::SparseMatrix<double> K;
    using LDLT_t = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper>;
    LDLT_t ldlt;

    void setupKKT(const Eigen::SparseMatrix<double> &G,
                  const Eigen::SparseMatrix<double> &A);
    void updateKKT();
    void solveKKT(const Eigen::VectorXd &rhs,
                  Eigen::VectorXd &dx,
                  Eigen::VectorXd &dy,
                  Eigen::VectorXd &dz,
                  bool initialized);
    void bringToCone(Eigen::VectorXd &x);
    void computeResiduals();
    void updateStatistics();
    bool checkExitConditions(bool reduced_accuracy);
    bool updateScalings();
    void scale();
    void RHS_affine();
    void scale2add(const Eigen::VectorXd &x, Eigen::VectorXd &y);
};