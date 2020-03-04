#pragma once

#include <Eigen/Sparse>

namespace ecos_eigen
{

enum class exitcode
{
    OPTIMAL = 0,           /* Problem solved to optimality              */
    PRIMAL_INFEASIBLE = 1, /* Found certificate of primal infeasibility */
    DUAL_INFEASIBLE = 2,   /* Found certificate of dual infeasibility   */
    MAXIT = -1,            /* Maximum number of iterations reached      */
    NUMERICS = -2,         /* Search direction unreliable               */
    OUTCONE = -3,          /* s or z got outside the cone, numerics?    */
    FATAL = -7,            /* Unknown problem in solver                 */
    CLOSE_TO_OPTIMAL = 10,
    CLOSE_TO_PRIMAL_INFEASIBLE = 11,
    CLOSE_TO_DUAL_INFEASIBLE = 12,
    NOT_CONVERGED_YET = -87
};

constexpr bool debug_printing = false;

struct Settings
{
    const double gamma = 0.99;         // scaling the final step length
    const double delta = 2e-7;         // regularization parameter
    const double deltastat = 7e-8;     // static regularization parameter
    const double eps = 1e13;           // regularization threshold
    const double feastol = 1e-8;       // primal/dual infeasibility tolerance
    const double abstol = 1e-8;        // absolute tolerance on duality gap
    const double reltol = 1e-8;        // relative tolerance on duality gap
    const double feastol_inacc = 1e-4; // primal/dual infeasibility relaxed tolerance
    const double abstol_inacc = 5e-5;  // absolute relaxed tolerance on duality gap
    const double reltol_inacc = 5e-5;  // relative relaxed tolerance on duality gap
    const size_t nitref = 9;           // number of iterative refinement steps
    const size_t maxit = 100;          // maximum number of iterations
    const bool verbose = true;         // print solver output
    const double linsysacc = 1e-14;    // rel. accuracy of search direction
    const double irerrfact = 6;        // factor by which IR should reduce err
    const double stepmin = 1e-6;       // smallest step that we do take
    const double stepmax = 0.999;      // largest step allowed, also in affine dir.
    const double sigmamin = 1e-4;      // always do some centering
    const double sigmamax = 1.;        // never fully center
    const size_t equil_iters = 3;      // eqilibration iterations
    const size_t iter_max = 100;       // maximum solver iterations
    const size_t safeguard = 500;      // Maximum increase in PRES before NUMERICS is thrown.
};

struct Information
{
    double pcost;
    double dcost;
    double pres;
    double dres;
    bool pinf;
    bool dinf;
    std::optional<double> pinfres;
    std::optional<double> dinfres;
    double gap;
    std::optional<double> relgap;
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

    bool operator>(Information &other) const;
};

struct PositiveCone
{
    Eigen::VectorXd w; // size num_pc
    Eigen::VectorXd v; // size num_pc
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
    double u0;             // eta
    double u1;             // u = [u0; u1 * q]
    double v1;             // v = [0; v1 * q]
};

struct Work
{
    void allocate(size_t num_var, size_t num_eq, size_t num_ineq);
    Eigen::VectorXd x;      // Primal variables  size num_var
    Eigen::VectorXd y;      // Multipliers for equality constaints  size num_eq
    Eigen::VectorXd z;      // Multipliers for conic inequalities   size num_ineq
    Eigen::VectorXd s;      // Slacks for conic inequalities        size num_ineq
    Eigen::VectorXd lambda; // Scaled variable                      size num_ineq

    // Homogeneous embedding
    double kap; // kappa
    double tau; // tau

    // Temporary storage
    double cx, by, hz;

    Information i;
};

class ECOSEigen
{
    /** 
     * Solves
     * 
     * min  c'*x
     * s.t. A*x = b
     *      G*x <=_K h
     * 
     * where the last inequality is generalized, i.e. h - G*x belongs to the cone K. 
     * ECOS supports the positive orthant R_+, second-order cones Q_n defined as
     * 
     * Q_n = { (t,x) | t >= || x ||_2 } 
     * 
     * n:       Number of variables.
     * m:       Number of inequality constraints.
     * p:       Number of equality constraints.
     * l:       The dimension of the positive orthant, i.e. in Gx+s=h, s in K.
     * The first l elements of s are >=0, ncones is the number of second-order cones present in K.
     * ncones:  Number of second order cones in K.
     * q:       Vector of dimesions of each cone constraint in K.
     * 
     * A(p,n):  Equality constraint matrix.
     * b(p):    Equality constraint vector.
     * G(m,n):  Generalized inequality matrix.
     * h(m):    Generalized inequality vector.
     * c(n):    Variable weights.
     */

public:
    ECOSEigen(const Eigen::SparseMatrix<double> &G,
              const Eigen::SparseMatrix<double> &A,
              const Eigen::VectorXd &c,
              const Eigen::VectorXd &h,
              const Eigen::VectorXd &b,
              const Eigen::VectorXi &soc_dims);

    // traditional interface for compatibility
    ECOSEigen(int n, int m, int p, int l, int ncones, int *q,
              double *Gpr, int *Gjc, int *Gir,
              double *Apr, int *Ajc, int *Air,
              double *c, double *h, double *b);

    exitcode solve();

    const Eigen::VectorXd &solution() const;

private:
    void build(const Eigen::SparseMatrix<double> &G,
               const Eigen::SparseMatrix<double> &A,
               const Eigen::VectorXd &c,
               const Eigen::VectorXd &h,
               const Eigen::VectorXd &b,
               const Eigen::VectorXi &soc_dims);

    Settings settings;
    Work w, w_best;

    size_t num_var;  // Number of variables (n)
    size_t num_eq;   // Number of equality constraints (p)
    size_t num_ineq; // Number of inequality constraints (m)
    size_t num_pc;   // Number of positive constraints (l)
    size_t num_sc;   // Number of second order cone constraints (ncones)
    size_t dim_K;    // Dimension of KKT matrix

    PositiveCone lp_cone;
    std::vector<SecondOrderCone> so_cones;

    Eigen::SparseMatrix<double> G;
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> Gt;
    Eigen::SparseMatrix<double> At;
    Eigen::VectorXd c;
    Eigen::VectorXd h;
    Eigen::VectorXd b;

    // Residuals
    Eigen::VectorXd rx; // (size num_var)
    Eigen::VectorXd ry; // (size num_eq)
    Eigen::VectorXd rz; // (size num_ineq)
    double hresx, hresy, hresz;
    double rt;

    // Norm iterates
    double nx, ny, nz, ns;

    // Equilibration vectors
    Eigen::VectorXd x_equil; // (size n)
    Eigen::VectorXd A_equil; // (size num_eq)
    Eigen::VectorXd G_equil; // (size num_ineq)

    Eigen::VectorXd rhs1; // The right hand side in the first KKT equation.
    Eigen::VectorXd rhs2; // The right hand side in the second KKT equation.

    // The problem data scaling parameters
    double resx0, resy0, resz0;

    Eigen::VectorXd dsaff_by_W, W_times_dzaff, dsaff;

    Eigen::VectorXd tmp1;
    Eigen::VectorXd tmp2;
    Eigen::VectorXd tmp3;
    Eigen::VectorXd tmp4;
    Eigen::VectorXd tmp5;
    Eigen::VectorXd tmp6;

    // KKT Matrix
    Eigen::SparseMatrix<double> K;
    using LDLT_t = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper>;
    LDLT_t ldlt;
    std::vector<double *> KKT_ptr; // Pointer to elements for fast update
    void setupKKT();
    void initKKT();
    bool updateKKT();
    size_t solveKKT(const Eigen::VectorXd &rhs,
                    Eigen::VectorXd &dx,
                    Eigen::VectorXd &dy,
                    Eigen::VectorXd &dz,
                    bool initialize);

    void allocate();
    void initCones(const Eigen::VectorXi &soc_dims);

    void bringToCone(const Eigen::VectorXd &r, Eigen::VectorXd &s);
    void computeResiduals();
    void updateStatistics();
    exitcode checkExitConditions(bool reduced_accuracy);
    bool updateScalings(const Eigen::VectorXd &s,
                        const Eigen::VectorXd &z,
                        Eigen::VectorXd &lambda);
    void RHS_affine();
    void RHS_combined();
    void scale2add1(const Eigen::VectorXd &x, Eigen::VectorXd &y);
    void scale2add2(const Eigen::VectorXd &x, Eigen::VectorXd &y);
    void scale(const Eigen::VectorXd &z, Eigen::VectorXd &lambda);
    double lineSearch(Eigen::VectorXd &lambda,
                      Eigen::VectorXd &ds,
                      Eigen::VectorXd &dz,
                      double tau,
                      double dtau,
                      double kap,
                      double dkap);
    double conicProduct(const Eigen::VectorXd &u,
                        const Eigen::VectorXd &v,
                        Eigen::VectorXd &w);
    void conicDivision(const Eigen::VectorXd &u,
                       const Eigen::VectorXd &w,
                       Eigen::VectorXd &v);
    void backscale();
    void setEquilibration();
    void unsetEquilibration();
};

} // namespace ecos_eigen