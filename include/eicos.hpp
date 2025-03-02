#pragma once

#include <cuDSS/SparseMatrix>
#include <cuDSS/Vector>

namespace EiCOS
{

    enum class exitcode
    {
        optimal = 0,           /* Problem solved to optimality              */
        primal_infeasible = 1, /* Found certificate of primal infeasibility */
        dual_infeasible = 2,   /* Found certificate of dual infeasibility   */
        maxit = -1,            /* Maximum number of iterations reached      */
        numerics = -2,         /* Search direction unreliable               */
        outcone = -3,          /* s or z got outside the cone, numerics?    */
        fatal = -7,            /* Unknown problem in solver                 */
        close_to_optimal = 10,
        close_to_primal_infeasible = 11,
        close_to_dual_infeasible = 12,
        not_converged_yet = -87
    };

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
        const size_t nitref = 9;           // maximum number of iterative refinement steps
        const size_t maxit = 100;          // maximum number of iterations
        bool verbose = false;              // print solver output
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

        bool isBetterThan(Information &other) const;
    };

    struct LPCone
    {
        cuDSS::Vector w; // size n_lc
        cuDSS::Vector v; // size n_lc
    };

    struct SOCone
    {
        size_t dim;            // dimension of cone
        cuDSS::Vector skbar; // temporary variables to work with
        cuDSS::Vector zkbar; // temporary variables to work with
        double a;              // = wbar(1)
        double d1;             // first element of D
        double w;              // = q'*q
        double eta;            // eta = (sres / zres)^(1/4)
        double eta_square;     // eta^2 = (sres / zres)^(1/2)
        cuDSS::Vector q;     // = wbar(2:end)
        double u0;             // eta
        double u1;             // u = [u0; u1 * q]
        double v1;             // v = [0; v1 * q]
    };

    struct Work
    {
        void allocate(size_t n_var, size_t n_eq, size_t n_ineq);
        cuDSS::Vector x;      // Primal variables  size n_var
        cuDSS::Vector y;      // Multipliers for equality constaints  (size n_eq)
        cuDSS::Vector z;      // Multipliers for conic inequalities   (size n_ineq)
        cuDSS::Vector s;      // Slacks for conic inequalities        (size n_ineq)
        cuDSS::Vector lambda; // Scaled variable                      (size n_ineq)

        // Homogeneous embedding
        double kap; // kappa
        double tau; // tau

        // Temporary storage
        double cx, by, hz;

        Information i;
    };

    class Solver
    {
        /**    
     * 
     *    ..---''''---..
     *    \ '''----''' /
     *     \          /
     *      \      ########  ##  ########  ########  ########
     *       \     ########  ##  ########  ########  ########
     *        \    ##            ##        ##    ##  ##      
     *          \/ ########  ##  ##        ##    ##  ########
     *          /\ ########  ##  ##        ##    ##  ########
     *        /    ##        ##  ##        ##    ##        ##
     *       /     ########  ##  ########  ########  ########
     *      /      ########  ##  ########  ########  ######## 
     *     /          \
     *    /            \
     *    `'---....---'´
     * 
     */

    public:
        Solver(const cuDSS::SparseMatrix &G,
               const cuDSS::SparseMatrix &A,
               const cuDSS::Vector &c,
               const cuDSS::Vector &h,
               const cuDSS::Vector &b,
               const cuDSS::VectorInt &soc_dims);
        void updateData(const cuDSS::SparseMatrix &G,
                        const cuDSS::SparseMatrix &A,
                        const cuDSS::Vector &c,
                        const cuDSS::Vector &h,
                        const cuDSS::Vector &b);

        // traditional interface for compatibility
        Solver(int n, int m, int p, int l, int ncones, int *q,
               double *Gpr, int *Gjc, int *Gir,
               double *Apr, int *Ajc, int *Air,
               double *c, double *h, double *b);
        void updateData(double *Gpr, double *Apr,
                        double *c, double *h, double *b);

        exitcode solve(bool verbose = false);

        const cuDSS::Vector &solution() const;

        Settings &getSettings();
        const Information &getInfo() const;

        // void saveProblemData(const std::string &path = "problem_data.hpp");

    private:
        void build(const cuDSS::SparseMatrix &G,
                   const cuDSS::SparseMatrix &A,
                   const cuDSS::Vector &c,
                   const cuDSS::Vector &h,
                   const cuDSS::Vector &b,
                   const cuDSS::VectorInt &soc_dims);

        Settings settings;
        Work w, w_best;

        size_t n_var;  // Number of variables (n)
        size_t n_eq;   // Number of equality constraints (p)
        size_t n_ineq; // Number of inequality constraints (m)
        size_t n_lc;   // Number of linear constraints (l)
        size_t n_sc;   // Number of second order cone constraints (ncones)
        size_t dim_K;  // Dimension of KKT matrix

        LPCone lp_cone;
        std::vector<SOCone> so_cones;

        cuDSS::SparseMatrix G;
        cuDSS::SparseMatrix A;
        cuDSS::SparseMatrix Gt;
        cuDSS::SparseMatrix At;
        cuDSS::Vector c;
        cuDSS::Vector h;
        cuDSS::Vector b;

        // Residuals
        cuDSS::Vector rx; // (size n_var)
        cuDSS::Vector ry; // (size n_eq)
        cuDSS::Vector rz; // (size n_ineq)
        double hresx, hresy, hresz;
        double rt;

        // Norm iterates
        double nx, ny, nz, ns;

        // Equilibration vectors
        cuDSS::Vector x_equil; // (size n_var)
        cuDSS::Vector A_equil; // (size n_eq)
        cuDSS::Vector G_equil; // (size n_ineq)
        bool equibrilated;

        // The problem data scaling parameters
        double resx0, resy0, resz0;

        cuDSS::Vector dsaff_by_W, W_times_dzaff, dsaff;

        // KKT
        cuDSS::Vector rhs1; // The right hand side in the first  KKT equation.
        cuDSS::Vector rhs2; // The right hand side in the second KKT equation.
        cuDSS::SparseMatrix K;
        using LDLT_t = cuDSS::LDLT<cuDSS::SparseMatrix, cuDSS::Upper>;
        LDLT_t ldlt;
        std::vector<double *> KKT_V_ptr;  // Pointer to scaling/regularization elements for fast update
        std::vector<double *> KKT_AG_ptr; // Pointer to A/G elements for fast update
        void setupKKT();
        void resetKKTScalings();
        void updateKKTScalings();
        void updateKKTAG();
        size_t solveKKT(const cuDSS::Vector &rhs,
                        cuDSS::Vector &dx,
                        cuDSS::Vector &dy,
                        cuDSS::Vector &dz,
                        bool initialize);

        void allocate();

        void bringToCone(const cuDSS::Vector &r, cuDSS::Vector &s);
        void computeResiduals();
        void updateStatistics();
        exitcode checkExitConditions(bool reduced_accuracy);
        bool updateScalings(const cuDSS::Vector &s,
                            const cuDSS::Vector &z,
                            cuDSS::Vector &lambda);
        void RHSaffine();
        void RHScombined();
        void scale2add(const cuDSS::Vector &x, cuDSS::Vector &y);
        void scale(const cuDSS::Vector &z, cuDSS::Vector &lambda);
        double lineSearch(cuDSS::Vector &lambda,
                          cuDSS::Vector &ds,
                          cuDSS::Vector &dz,
                          double tau,
                          double dtau,
                          double kap,
                          double dkap);
        double conicProduct(const cuDSS::Vector &u,
                            const cuDSS::Vector &v,
                            cuDSS::Vector &w);
        void conicDivision(const cuDSS::Vector &u,
                           const cuDSS::Vector &w,
                           cuDSS::Vector &v);
        void backscale();
        void setEquilibration();
        void unsetEquilibration();
        void cacheIndices();
        void printSummary();
    };

} // namespace EiCOS
