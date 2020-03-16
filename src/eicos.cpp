#include "eicos.hpp"

#include <chrono>
#include <Eigen/SparseCholesky>
#include "printing.hpp"

namespace EiCOS
{

void Work::allocate(size_t n_var, size_t n_eq, size_t n_ineq)
{
    x.resize(n_var);
    y.resize(n_eq);
    s.resize(n_ineq);
    z.resize(n_ineq);
    lambda.resize(n_ineq);
}

/**
 * Compares stats of two iterates with each other.
 * Returns true if this is better than other, false otherwise.
 */
bool Information::isBetterThan(Information &other) const
{
    if (pinfres.has_value() and kapovert > 1.)
    {
        if (other.pinfres.has_value())
        {
            if ((gap > 0. and other.gap > 0. and gap < other.gap) and
                (pinfres > 0. and pinfres < other.pres) and
                (mu > 0. and mu < other.mu))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            if ((gap > 0. and other.gap > 0. and gap < other.gap) and
                (mu > 0. and mu < other.mu))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    else
    {
        if ((gap > 0. and other.gap > 0. and gap < other.gap) and
            (pres > 0. and pres < other.pres) and
            (dres > 0. and dres < other.dres) and
            (kapovert > 0. and kapovert < other.kapovert) and
            (mu > 0. and mu < other.mu))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

void printSparseMatrix(const Eigen::SparseMatrix<double> &m)
{
    for (int j = 0; j < m.outerSize(); j++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(m, j); it; ++it)
        {
            print("({:3},{:3}) = {}\n", it.row() + 1, it.col() + 1, it.value());
        }
    }
}

Solver::Solver(const Eigen::SparseMatrix<double> &G,
               const Eigen::SparseMatrix<double> &A,
               const Eigen::VectorXd &c,
               const Eigen::VectorXd &h,
               const Eigen::VectorXd &b,
               const Eigen::VectorXi &soc_dims)
{
    build(G, A, c, h, b, soc_dims);
}

Solver::Solver(int n, int m, int p, int /* l */, int ncones, int *q,
               double *Gpr, int *Gjc, int *Gir,
               double *Apr, int *Ajc, int *Air,
               double *c, double *h, double *b)
{
    Eigen::SparseMatrix<double> G_;
    Eigen::SparseMatrix<double> A_;
    Eigen::VectorXd c_;
    Eigen::VectorXd h_;
    Eigen::VectorXd b_;
    Eigen::VectorXi q_;

    if (Gpr and Gjc and Gir)
    {
        G_ = Eigen::Map<Eigen::SparseMatrix<double>>(m, n, Gjc[n], Gjc, Gir, Gpr);
        q_ = Eigen::Map<Eigen::VectorXi>(q, ncones);
        h_ = Eigen::Map<Eigen::VectorXd>(h, m);
    }
    if (Apr and Ajc and Air)
    {
        A_ = Eigen::Map<Eigen::SparseMatrix<double>>(p, n, Ajc[n], Ajc, Air, Apr);
        b_ = Eigen::Map<Eigen::VectorXd>(b, p);
    }
    if (c)
    {
        c_ = Eigen::Map<Eigen::VectorXd>(c, n);
    }

    build(G_, A_, c_, h_, b_, q_);
}

void Solver::build(const Eigen::SparseMatrix<double> &G,
                   const Eigen::SparseMatrix<double> &A,
                   const Eigen::VectorXd &c,
                   const Eigen::VectorXd &h,
                   const Eigen::VectorXd &b,
                   const Eigen::VectorXi &soc_dims)
{
    assert(not(c.hasNaN() or h.hasNaN() or b.hasNaN()));

    this->G = G;
    this->A = A;
    this->c = c;
    this->h = h;
    this->b = b;

    // Dimensions
    if (A.cols() > 0 and G.cols() > 0)
    {
        assert(A.cols() == G.cols());
    }
    n_var = std::max(A.cols(), G.cols());
    n_eq = A.rows();
    n_ineq = G.rows();
    n_lc = n_ineq - soc_dims.sum();
    n_sc = soc_dims.size();

    /**
     *  Dimension of KKT matrix
     *   =   # variables
     *     + # equality constraints
     *     + # inequality constraints
     *     + 2 * # second order cones (expansion of SOC scalings)
     */
    dim_K = n_var + n_eq + n_ineq + 2 * n_sc;

    // initialize cones
    so_cones.resize(soc_dims.size());
    for (int i = 0; i < soc_dims.size(); i++)
    {
        SOCone &sc = so_cones[i];
        sc.dim = soc_dims[i];
        sc.eta = 0.;
        sc.a = 0.;
    }

    allocate();

    printSummary();

    setEquilibration();

    Gt = this->G.transpose();
    At = this->A.transpose();

    setupKKT();
}

void Solver::printSummary()
{
    print_dbg("- - - - - - - - - - - - - - -\n");
    print_dbg("|      Problem summary      |\n");
    print_dbg("- - - - - - - - - - - - - - -\n");
    print_dbg("    Primal variables:  {}\n", n_var);
    print_dbg("Equality constraints:  {}\n", n_eq);
    print_dbg("     Conic variables:  {}\n", n_ineq);
    print_dbg("- - - - - - - - - - - - - - -\n");
    print_dbg("  Size of LP cone:     {}\n", n_lc);
    print_dbg("  Number of SOCs:      {}\n", n_sc);
    print_dbg("- - - - - - - - - - - - - - -\n");
    for (size_t i = 0; i < n_sc; i++)
    {
        print_dbg("  Size of SOC #{}:      {}\n", i + 1, so_cones[i].dim);
    }
    print_dbg("- - - - - - - - - - - - - - -\n");
}

/* Incomplete allocation. Heap is still used later.*/
void Solver::allocate()
{
    // Allocate work struct
    w.allocate(n_var, n_eq, n_ineq);

    // Set up LP cone
    lp_cone.v.resize(n_lc);
    lp_cone.w.resize(n_lc);

    // Set up second-order cone
    for (SOCone &sc : so_cones)
    {
        sc.q.resize(sc.dim - 1);
        sc.skbar.resize(sc.dim);
        sc.zkbar.resize(sc.dim);
    }

    W_times_dzaff.resize(n_ineq);
    dsaff_by_W.resize(n_ineq);
    dsaff.resize(n_ineq);

    rx.resize(n_var);
    ry.resize(n_eq);
    rz.resize(n_ineq);

    rhs1.resize(dim_K);
    rhs2.resize(dim_K);

    K.reserve(dim_K);

    Gt.reserve(G.nonZeros());
    At.reserve(A.nonZeros());

    size_t KKT_ptr_size = n_lc;
    for (const SOCone &sc : so_cones)
    {
        KKT_ptr_size += 3 * sc.dim + 1;
    }
    KKT_V_ptr.reserve(KKT_ptr_size);
    KKT_AG_ptr.reserve(A.nonZeros() + G.nonZeros());
}

const Eigen::VectorXd &Solver::solution() const
{
    return w.x;
}

void maxRows(Eigen::VectorXd &e, const Eigen::SparseMatrix<double> m)
{
    for (int j = 0; j < m.cols(); j++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(m, j); it; ++it)
        {
            e(it.row()) = std::max(std::fabs(it.value()), e(it.row()));
        }
    }
}

void maxCols(Eigen::VectorXd &e, const Eigen::SparseMatrix<double> m)
{
    for (int j = 0; j < m.cols(); j++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(m, j); it; ++it)
        {
            e(j) = std::max(std::fabs(it.value()), e(j));
        }
    }
}

void equilibrateRows(const Eigen::VectorXd &e, Eigen::SparseMatrix<double> &m)
{
    for (int j = 0; j < m.cols(); j++)
    {
        /* equilibrate the rows of a matrix */
        for (Eigen::SparseMatrix<double>::InnerIterator it(m, j); it; ++it)
        {
            it.valueRef() /= e(it.row());
        }
    }
}

void equilibrateCols(const Eigen::VectorXd &e, Eigen::SparseMatrix<double> &m)
{
    for (int j = 0; j < m.cols(); j++)
    {
        /* equilibrate the columns of a matrix */
        for (Eigen::SparseMatrix<double>::InnerIterator it(m, j); it; ++it)
        {
            it.valueRef() /= e(j);
        }
    }
}

void Solver::setEquilibration()
{
    x_equil.resize(n_var);
    A_equil.resize(n_eq);
    G_equil.resize(n_ineq);

    Eigen::VectorXd x_tmp(n_var);
    Eigen::VectorXd A_tmp(n_eq);
    Eigen::VectorXd G_tmp(n_ineq);

    /* Initialize equilibration vector to 1 */
    x_equil.setOnes();
    A_equil.setOnes();
    G_equil.setOnes();

    /* Iterative equilibration */
    for (size_t iter = 0; iter < settings.equil_iters; iter++)
    {
        /* Each iteration updates A and G */

        /* Zero out the temp vectors */
        x_tmp.setZero();
        A_tmp.setZero();
        G_tmp.setZero();

        /* Compute norm across columns of A, G */
        maxCols(x_tmp, A);
        maxCols(x_tmp, G);

        /* Compute norm across rows of A */
        maxRows(A_tmp, A);

        /* Compute norm across rows of G */
        maxRows(G_tmp, G);

        /* Now collapse cones together by using total over the group */
        size_t ind = n_lc;
        for (const SOCone &sc : so_cones)
        {
            const double total = G_tmp.segment(ind, sc.dim).sum();
            G_tmp.segment(ind, sc.dim).setConstant(total);
            ind += sc.dim;
        }

        /* Take the square root */
        auto sqrt_op = [](const double a) { return std::fabs(a) < 1e-6 ? 1. : std::sqrt(a); };
        x_tmp = x_tmp.unaryExpr(sqrt_op);
        A_tmp = A_tmp.unaryExpr(sqrt_op);
        G_tmp = G_tmp.unaryExpr(sqrt_op);

        /* Equilibrate the matrices */
        equilibrateRows(A_tmp, A);
        equilibrateRows(G_tmp, G);
        equilibrateCols(x_tmp, A);
        equilibrateCols(x_tmp, G);

        /* Update the equilibration matrix */
        x_equil = x_equil.cwiseProduct(x_tmp);
        A_equil = A_equil.cwiseProduct(A_tmp);
        G_equil = G_equil.cwiseProduct(G_tmp);
    }

    /* Equilibrate the c vector */
    c = c.cwiseQuotient(x_equil);

    /* Equilibrate the b vector */
    b = b.cwiseQuotient(A_equil);

    /* Equilibrate the h vector */
    h = h.cwiseQuotient(G_equil);

    equibrilated = true;
}

void restore(const Eigen::VectorXd &d, const Eigen::VectorXd &e,
             Eigen::SparseMatrix<double> &m)
{
    assert(not m.IsRowMajor);
    for (int col = 0; col < m.cols(); ++col)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(m, col); it; ++it)
        {
            it.valueRef() *= d(it.row()) * e(it.col());
        }
    }
}

void Solver::unsetEquilibration()
{
    restore(A_equil, x_equil, A);
    restore(G_equil, x_equil, G);

    /* Unequilibrate the c vector */
    c = c.cwiseProduct(x_equil);

    /* Unequilibrate the b vector */
    b = b.cwiseProduct(A_equil);

    /* Unequilibrate the h vector */
    h = h.cwiseProduct(G_equil);

    equibrilated = false;
}

/**
 * Update scalings.
 * Returns false as soon as any multiplier or slack leaves the cone,
 * as this indicates severe problems.
 */
bool Solver::updateScalings(const Eigen::VectorXd &s,
                            const Eigen::VectorXd &z,
                            Eigen::VectorXd &lambda)
{
    /* LP cone */
    lp_cone.v = s.head(n_lc).cwiseQuotient(z.head(n_lc));
    lp_cone.w = lp_cone.v.cwiseSqrt();

    /* SO cone */
    size_t cone_start = n_lc;
    for (SOCone &sc : so_cones)
    {
        /* Check residuals and quit if they're negative */
        const double sres = s(cone_start) * s(cone_start) -
                            s.segment(cone_start + 1, sc.dim - 1).squaredNorm();
        const double zres = z(cone_start) * z(cone_start) -
                            z.segment(cone_start + 1, sc.dim - 1).squaredNorm();
        if (sres <= 0 or zres <= 0)
        {
            return false;
        }

        /* Normalize variables */
        const double snorm = std::sqrt(sres);
        const double znorm = std::sqrt(zres);

        sc.skbar = s.segment(cone_start, sc.dim) / snorm;
        sc.zkbar = z.segment(cone_start, sc.dim) / znorm;

        sc.eta_square = snorm / znorm;
        sc.eta = std::sqrt(sc.eta_square);

        /* Normalized Nesterov-Todd scaling point */
        double gamma = 1. + sc.skbar.dot(sc.zkbar);
        gamma = std::sqrt(0.5 * gamma);

        const double a = (0.5 / gamma) * (sc.skbar(0) + sc.zkbar(0));
        sc.q = (0.5 / gamma) * (sc.skbar.tail(sc.dim - 1) -
                                sc.zkbar.tail(sc.dim - 1));
        const double w = sc.q.squaredNorm();

        /* Pre-compute variables needed for KKT matrix (used in KKT scaling) */
        const double c = (1. + a) + w / (1. + a);
        const double d = 1. + 2. / (1. + a) + w / std::pow(1. + a, 2);

        const double d1 = std::max(0., 0.5 * (std::pow(a, 2) + w * (1. - std::pow(c, 2) / (1. + w * d))));
        const double u0_square = std::pow(a, 2) + w - d1;

        const double c2byu02 = (c * c) / u0_square;
        if (c2byu02 - d <= 0)
        {
            return false;
        }

        sc.d1 = d1;
        sc.u0 = std::sqrt(u0_square);
        sc.u1 = std::sqrt(c2byu02);
        sc.v1 = std::sqrt(c2byu02 - d);
        sc.a = a;
        sc.w = w;

        /* Increase offset for next cone */
        cone_start += sc.dim;
    }
    /* lambda = W * z */
    scale(z, lambda);

    return true;
}

/**
 * Fast multiplication by scaling matrix.
 * Returns lambda = W * z
 */
void Solver::scale(const Eigen::VectorXd &z, Eigen::VectorXd &lambda)
{
    /* LP cone */
    lambda.head(n_lc) = lp_cone.w.cwiseProduct(z.head(n_lc));

    /* SO cone */
    size_t cone_start = n_lc;
    for (const SOCone &sc : so_cones)
    {
        /* zeta = q' * z1 */
        const double zeta = sc.q.dot(z.segment(cone_start + 1, sc.dim - 1));

        /* factor = z0 + zeta / (1 + a); */
        const double factor = z(cone_start) + zeta / (1. + sc.a);

        /* Write out result */
        lambda(cone_start) = sc.eta * (sc.a * z(cone_start) + zeta);
        lambda.segment(cone_start + 1, sc.dim - 1) =
            sc.eta * (z.segment(cone_start + 1, sc.dim - 1) + factor * sc.q);

        cone_start += sc.dim;
    }
}

/**
 * This function is reponsible for checking the exit/convergence conditions.
 * If one of the exit conditions is met, The solver displays an exit message and returns
 * the corresponding exit code. The calling function must then make sure that the solver
 * is indeed correctly exited, so a call to this function should always be followed
 * by a break statement.
 *
 * In reduced accuracy mode, reduced precisions are checked, and the exit display is augmented
 *               by "Close to". The exitcodes returned are increased by the value
 *               of mode.
 *
 * The primal and dual infeasibility flags pinf and dinf are raised
 * according to the outcome of the test.
 *
 * If none of the exit tests are met, the function returns not_converged_yet.
 * This should not be an exitflag that is ever returned to the outside world.
 */
exitcode Solver::checkExitConditions(bool reduced_accuracy)
{
    double feastol;
    double abstol;
    double reltol;

    /* Set accuracy against which to check */
    if (reduced_accuracy)
    {
        /* Check convergence against reduced precisions */
        feastol = settings.feastol_inacc;
        abstol = settings.abstol_inacc;
        reltol = settings.reltol_inacc;
    }
    else
    {
        /* Check convergence against normal precisions */
        feastol = settings.feastol;
        abstol = settings.abstol;
        reltol = settings.reltol;
    }

    /* Optimal? */
    if ((-w.cx > 0. or -w.by - w.hz >= -abstol) and
        (w.i.pres < feastol and w.i.dres < feastol) and
        (w.i.gap < abstol or w.i.relgap < reltol))
    {
        if (settings.verbose)
        {
            if (reduced_accuracy)
            {
                print("Close to optimal (within feastol={:3.1e}, reltol={:3.1e}, abstol={:3.1e}).\n",
                      std::max(w.i.dres, w.i.pres), w.i.relgap.value_or(0.), w.i.gap);
            }
            else
            {
                print("Optimal (within feastol={:3.1e}, reltol={:3.1e}, abstol={:3.1e}).\n",
                      std::max(w.i.dres, w.i.pres), w.i.relgap.value_or(0.), w.i.gap);
            }
        }

        w.i.pinf = false;
        w.i.dinf = false;

        if (reduced_accuracy)
        {
            return exitcode::close_to_optimal;
        }
        else
        {
            return exitcode::optimal;
        }
    }

    /* Dual infeasible? */
    else if ((w.i.dinfres.has_value()) and
             (w.i.dinfres.value() < feastol) and
             (w.tau < w.kap))
    {
        if (settings.verbose)
        {
            if (reduced_accuracy)
            {
                print("Close to unbounded (within feastol={:3.1e}).\n", w.i.dinfres.value());
            }
            else
            {
                print("Unbounded (within feastol={:3.1e}).\n", w.i.dinfres.value());
            }
        }

        w.i.pinf = false;
        w.i.dinf = true;

        if (reduced_accuracy)
        {
            return exitcode::close_to_dual_infeasible;
        }
        else
        {
            return exitcode::dual_infeasible;
        }
    }

    /* Primal infeasible? */
    else if (((w.i.pinfres.has_value() and w.i.pinfres < feastol) and (w.tau < w.kap)) or
             (w.tau < feastol and w.kap < feastol and w.i.pinfres < feastol))
    {
        if (reduced_accuracy)
        {
            print("Close to primal infeasible (within feastol={:3.1e}).\n", w.i.pinfres.value());
        }
        else
        {
            print("Primal infeasible (within feastol={:3.1e}).\n", w.i.pinfres.value());
        }

        w.i.pinf = true;
        w.i.dinf = false;

        if (reduced_accuracy)
        {
            return exitcode::close_to_primal_infeasible;
        }
        else
        {
            return exitcode::primal_infeasible;
        }
    }

    /* Indicate if none of the above criteria are met */
    else
    {
        return exitcode::not_converged_yet;
    }
}

void Solver::computeResiduals()
{
    /**
    * hrx = -A' * y - G' * z       rx = hrx - tau * c      hresx = ||rx||_2
    * hry =  A * x                 ry = hry - tau * b      hresy = ||ry||_2
    * hrz =  s + G * x             rz = hrz - tau * h      hresz = ||rz||_2
    * 
    * rt = kappa + c' * x + b' * y + h' * z
    */

    /* rx = -A' * y - G' * z - tau * c */
    rx = -Gt * w.z;
    if (n_eq > 0)
    {
        rx -= At * w.y;
    }
    hresx = rx.norm();
    rx -= w.tau * c;

    /* ry = A * x - tau * b */
    if (n_eq > 0)
    {
        ry = A * w.x;
        hresy = ry.norm();
        ry -= w.tau * b;
    }
    else
    {
        hresy = 0.;
    }

    /* rz = s + G * x - tau * h */
    rz = w.s + G * w.x;
    hresz = rz.norm();
    rz -= w.tau * h;

    /* rt = kappa + c' * x + b' * y + h' * z; */
    w.cx = c.dot(w.x);
    w.by = n_eq > 0 ? b.dot(w.y) : 0.;
    w.hz = h.dot(w.z);
    rt = w.kap + w.cx + w.by + w.hz;

    nx = w.x.norm();
    ny = w.y.norm();
    nz = w.z.norm();
    ns = w.s.norm();
}

void Solver::updateStatistics()
{
    w.i.gap = w.s.dot(w.z);
    w.i.mu = (w.i.gap + w.kap * w.tau) / ((n_lc + n_sc) + 1);
    w.i.kapovert = w.kap / w.tau;
    w.i.pcost = w.cx / w.tau;
    w.i.dcost = -(w.hz + w.by) / w.tau;

    /* Relative duality gap */
    if (w.i.pcost < 0.)
    {
        w.i.relgap = w.i.gap / (-w.i.pcost);
    }
    else if (w.i.dcost > 0.)
    {
        w.i.relgap = w.i.gap / w.i.dcost;
    }
    else
    {
        w.i.relgap = std::nullopt;
    }

    /* Residuals */
    const double nry = n_eq > 0 ? ry.norm() / std::max(resy0 + nx, 1.) : 0.;
    const double nrz = rz.norm() / std::max(resz0 + nx + ns, 1.);
    w.i.pres = std::max(nry, nrz) / w.tau;
    w.i.dres = rx.norm() / std::max(resx0 + ny + nz, 1.) / w.tau;

    /* Infeasibility measures */
    if ((w.hz + w.by) / std::max(ny + nz, 1.) < -settings.reltol)
    {
        w.i.pinfres = hresx / std::max(ny + nz, 1.);
    }
    if (w.cx / std::max(nx, 1.) < -settings.reltol)
    {
        w.i.dinfres = std::max(hresy / std::max(nx, 1.),
                               hresz / std::max(nx + ns, 1.));
    }

    print_dbg("TAU={:6.4e}  KAP={:6.4e}  PINFRES={:6.4e}  DINFRES={:6.4e}\n",
              w.tau, w.kap, w.i.pinfres.value_or(-1), w.i.dinfres.value_or(-1));

    if (settings.verbose)
    {
        const std::string line =
            format("{:2d}  {:+5.3e}  {:+5.3e}  {:+2.0e}  {:2.0e}  {:2.0e}  {:2.0e}  {:2.0e}",
                   w.i.iter, w.i.pcost, w.i.dcost, w.i.gap, w.i.pres, w.i.dres, w.i.kapovert, w.i.mu);

        if (w.i.iter == 0)
        {
            print("It     pcost       dcost      gap   pres   dres    k/t    mu     step   sigma     IR\n");
            print("{}    ---    ---   {:2d}/{:2d}  -\n", line, w.i.nitref1, w.i.nitref2);
        }
        else
        {
            print("{}  {:6.4f}  {:2.0e}  {:2d}/{:2d}/{:2d}\n",
                  line,
                  w.i.step, w.i.sigma,
                  w.i.nitref1,
                  w.i.nitref2,
                  w.i.nitref3);
        }
    }
}

/**
 * Scales a conic variable such that it lies strictly in the cone.
 * If it is already in the cone, r is simply copied to s.
 * Otherwise s = r + (1 + alpha) * e where alpha is the biggest residual.
 */
void Solver::bringToCone(const Eigen::VectorXd &r, Eigen::VectorXd &s)
{
    double alpha = -settings.gamma;

    /* ===== 1. Find maximum residual ===== */

    /* LP cone */
    for (size_t i = 0; i < n_lc; i++)
    {
        if (r(i) <= 0 and -r(i) > alpha)
        {
            alpha = -r(i);
        }
    }

    /* SO cone */
    size_t cone_start = n_lc;
    for (const SOCone &sc : so_cones)
    {
        const double cres = r(cone_start) -
                            r.segment(cone_start + 1, sc.dim - 1).norm();
        cone_start += sc.dim;

        if (cres <= 0 and -cres > alpha)
        {
            alpha = -cres;
        }
    }

    /* ===== 2. Compute s = r + (1 + alpha) * e ===== */

    alpha += 1.;

    /* LP cone */
    s = r;
    s.head(n_lc).array() += alpha;

    /* SO cone */
    cone_start = n_lc;
    for (const SOCone &sc : so_cones)
    {
        s(cone_start) += alpha;
        cone_start += sc.dim;
    }
}

void Solver::resetKKTScalings()
{
    size_t ptr_i = 0;

    /* LP cone */
    for (size_t k = 0; k < n_lc; k++)
    {
        *KKT_V_ptr[ptr_i++] = -1.;
    }

    /* SO cone */
    for (const SOCone &sc : so_cones)
    {
        /* D */
        for (size_t k = 0; k < sc.dim; k++)
        {
            *KKT_V_ptr[ptr_i++] = -1.;
        }

        /* -1 on diagonal */
        *KKT_V_ptr[ptr_i++] = -1.;

        /* -v */
        for (size_t k = 1; k < sc.dim; k++)
        {
            *KKT_V_ptr[ptr_i++] = 0.;
        }

        /* 1 on diagonal */
        *KKT_V_ptr[ptr_i++] = 1.;

        /* -u */
        *KKT_V_ptr[ptr_i++] = 0.;
        for (size_t k = 1; k < sc.dim; k++)
        {
            *KKT_V_ptr[ptr_i++] = 0.;
        }
    }
    assert(ptr_i == KKT_V_ptr.size());
}

exitcode Solver::solve(bool verbose)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    settings.verbose = verbose;
    exitcode code = exitcode::fatal;

    resetKKTScalings();

    /**
    * Set up first right hand side
    * 
    *   [ 0 ]
    *   [ b ]
    *   [ h ]
    * 
    */
    rhs1.setZero();
    rhs1.segment(n_var, n_eq) = b;
    rhs1.segment(n_var + n_eq, n_lc) = h.head(n_lc);
    size_t h_index = n_lc;
    size_t rhs1_index = n_var + n_eq + n_lc;
    for (const SOCone &sc : so_cones)
    {
        rhs1.segment(rhs1_index, sc.dim) = h.segment(h_index, sc.dim);
        h_index += sc.dim;
        rhs1_index += sc.dim + 2;
    }

    /**
    * Set up second right hand side
    * 
    *   [-c ]
    *   [ 0 ]
    *   [ 0 ]
    * 
    */
    rhs2.setZero();
    rhs2.head(n_var) = -c;

    /*  Set up scalings of problem data */
    const double scale_rx = c.norm();
    const double scale_ry = b.norm();
    const double scale_rz = h.norm();
    resx0 = std::max(1., scale_rx);
    resy0 = std::max(1., scale_ry);
    resz0 = std::max(1., scale_rz);

    /* Perform symbolic decomposition */
    ldlt.analyzePattern(K);

    /* Do LDLT factorization */
    ldlt.factorize(K);
    if (ldlt.info() != Eigen::Success)
    {
        print_dbg("Failed to factorize matrix while initializing!\n");
        return exitcode::fatal;
    }

    /**
	 * Primal Variables:
     * 
	 *  Solve 
     * 
     *  xhat = arg min ||Gx - h||_2^2  such that A * x = b
	 *  r = h - G * xhat
     * 
	 * Equivalent to
	 *
	 * [ 0   A'  G' ] [ xhat ]     [ 0 ]
     * [ A   0   0  ] [  y   ]  =  [ b ]
     * [ G   0  -I  ] [ -r   ]     [ h ]
     *
     *        (  r                       if alphap < 0
     * shat = < 
     *        (  r + (1 + alphap) * e    otherwise
     * 
     * where alphap = inf{ alpha | r + alpha * e >= 0 }
	 */

    /* Solve for RHS [0; b; h] */
    Eigen::VectorXd dx1(n_var);
    Eigen::VectorXd dy1(n_eq);
    Eigen::VectorXd dz1(n_ineq);
    print_dbg("Solving for RHS1.\n");
    w.i.nitref1 = solveKKT(rhs1, dx1, dy1, dz1, true);

    /* Copy out initial value of x */
    w.x = dx1;

    /* Copy out -r and bring to cone */
    bringToCone(-dz1, w.s);

    /**
	 * Dual Variables:
     * 
	 * Solve 
     * 
     * (yhat, zbar) = arg min ||z||_2^2 such that G'*z + A'*y + c = 0
	 *
	 * Equivalent to
	 *
	 * [ 0   A'  G' ] [  x   ]     [ -c ]
	 * [ A   0   0  ] [ yhat ]  =  [  0 ]
	 * [ G   0  -I  ] [ zbar ]     [  0 ]
	 *     
     *        (  zbar                       if alphad < 0
     * zhat = < 
     *        (  zbar + (1 + alphad) * e    otherwise
     * 
	 * where alphad = inf{ alpha | zbar + alpha * e >= 0 }
	 */

    /* Solve for RHS [-c; 0; 0] */
    Eigen::VectorXd dx2(n_var);
    Eigen::VectorXd dy2(n_eq);
    Eigen::VectorXd dz2(n_ineq);
    print_dbg("Solving for RHS2.\n");
    w.i.nitref2 = solveKKT(rhs2, dx2, dy2, dz2, true);

    /* Copy out initial value of y */
    w.y = dy2;

    /* Bring variable to cone */
    bringToCone(dz2, w.z);

    /**
    * Modify first right hand side
    * [ 0 ]    [-c ] 
    * [ b ] -> [ b ] 
    * [ h ]    [ h ] 
    */
    rhs1.head(n_var) = -c;

    /* Other variables */
    w.kap = 1.,
    w.tau = 1.,

    w.i.step = 0.;
    w.i.step_aff = 0.;
    w.i.pinf = false;
    w.i.dinf = false;
    w.i.iter_max = settings.iter_max;

    double pres_prev = std::numeric_limits<double>::max();

    /* Main interior point loop */
    for (w.i.iter = 0; w.i.iter <= w.i.iter_max; w.i.iter++)
    {
        computeResiduals();

        updateStatistics();

        /**
         *  SAFEGUARD: Backtrack to best previously seen iterate if
         *
         * - the update was bad such that the primal residual PRES has increased by a factor of SAFEGUARD, or
         * - the gap became negative
         *
         * If the safeguard is activated, the solver tests if reduced precision has been reached, and reports
         * accordingly. If not even reduced precision is reached, return the flag numerics.
         */
        if (w.i.iter > 0 and
            (w.i.pres > settings.safeguard * pres_prev or w.i.gap < 0.))
        {
            if (settings.verbose)
            {
                print("Unreliable search direction detected, recovering best iterate ({}) and stopping.\n",
                      w_best.i.iter);
            }

            /* Restore best iterate */
            w = w_best;

            /* Determine whether we have reached at least reduced accuracy */
            code = checkExitConditions(true);

            /* If not, exit anyways */
            if (code == exitcode::not_converged_yet)
            {
                code = exitcode::numerics;

                if (settings.verbose)
                {
                    print("\nNUMERICAL PROBLEMS (reached feastol={:3.1e}, reltol={:3.1e}, abstol={:3.1e}).",
                          std::max(w.i.dres, w.i.pres), w.i.relgap.value_or(0.), w.i.gap);
                }
                break;
            }
            else
            {
                break;
            }
        }

        pres_prev = w.i.pres;

        /* Check termination criteria to full precision and exit if necessary */
        code = checkExitConditions(false);

        if (code == exitcode::not_converged_yet)
        {
            /**
             * Full precision has not been reached yet. Check for two more cases of exit:
             *  (i) min step size, in which case we assume we won't make progress any more, and
             * (ii) maximum number of iterations reached
             * If these two are not fulfilled, another iteration will be made.
             */

            /* Did the line search cock up? (zero step length) */
            if (w.i.iter > 0 and w.i.step == settings.stepmin * settings.gamma)
            {
                if (settings.verbose)
                {
                    print("No further progress possible, recovering best iterate ({}) and stopping.", w_best.i.iter);
                }

                /* Restore best iterate */
                w = w_best;

                /* Determine whether we have reached reduced precision */
                code = checkExitConditions(true);

                if (code == exitcode::not_converged_yet)
                {
                    code = exitcode::numerics;
                    if (settings.verbose)
                    {
                        print("\nNUMERICAL PROBLEMS (reached feastol={:3.1e}, reltol={:3.1e}, abstol={:3.1e}).",
                              std::max(w.i.dres, w.i.pres), w.i.relgap.value_or(0.), w.i.gap);
                    }
                }
                break;
            }
            /* maxit reached? */
            else if (w.i.iter == w.i.iter_max)
            {
                if (settings.verbose)
                    print("\nMaximum number of iterations reached, ");

                /* Determine whether current iterate is better than what we had so far */
                if (w.i.isBetterThan(w_best.i))
                {
                    if (settings.verbose)
                        print("stopping.\n");
                }
                else
                {
                    if (settings.verbose)
                        print("recovering best iterate ({}) and stopping.\n", w_best.i.iter);
                    w = w_best;
                }

                /* Determine whether we have reached reduced precision */
                code = checkExitConditions(true);

                if (code == exitcode::not_converged_yet)
                {
                    code = exitcode::maxit;
                }
                break;
            }
            /* Stuck on NAN? */
            else if (std::isnan(w.i.pcost))
            {
                if (settings.verbose)
                    print("\nReached NaN dead end, ");

                /* Determine whether current iterate is better than what we had so far */
                if (w.i.iter == 0 or w.i.isBetterThan(w_best.i))
                {
                    if (settings.verbose)
                        print("stopping.\n");
                }
                else
                {
                    if (settings.verbose)
                        print("recovering best iterate ({}) and stopping.\n", w_best.i.iter);
                    w = w_best;

                    /* Determine whether we have reached reduced precision */
                    code = checkExitConditions(true);
                    if (code == exitcode::not_converged_yet)
                    {
                        code = exitcode::numerics;
                        print("stopping without convergence.\n");
                    }
                }
                break;
            }
        }
        else
        {
            /* Full precision has been reached, stop solver */
            break;
        }

        /**
         * SAFEGUARD:
         * Check whether current iterate is worth keeping as the best solution so far,
         * before doing another iteration
         */
        if (w.i.iter == 0)
        {
            /* We're at the first iterate, so there's nothing to compare yet */
            w_best = w;
        }
        else if (w.i.isBetterThan(w_best.i))
        {
            w_best = w;
        }

        updateScalings(w.s, w.z, w.lambda);

        updateKKTScalings();

        ldlt.factorize(K);

        if (ldlt.info() != Eigen::Success)
        {
            print_dbg("Failed to factorize matrix after update!\n");
            return exitcode::fatal;
        }

        /* Solve for RHS1, which is used later also in combined direction */
        solveKKT(rhs1, dx1, dy1, dz1, false);

        /* Affine Search Direction (predictor, need dsaff and dzaff only) */
        RHSaffine();

        print_dbg("Solving for affine search direction.\n");
        solveKKT(rhs2, dx2, dy2, dz2, false);

        /* dtau_denom = kap / tau - (c' * x1 + b * y1 + h' * z1); */
        const double dtau_denom = w.kap / w.tau - c.dot(dx1) - b.dot(dy1) - h.dot(dz1);

        /* dtauaff = (dt + c' * x2 + b * y2 + h' * z2) / dtau_denom; */
        const double dtauaff = (rt - w.kap + c.dot(dx2) + b.dot(dy2) + h.dot(dz2)) / dtau_denom;

        /* dzaff = dz2 + dtau_aff * dz1 */
        /* Let dz2   = dzaff, use this in the linesearch for unsymmetric cones */
        /* and w_times_dzaff = W * dz_aff */
        /* and dz2 = dz2 + dtau_aff * dz1 will store the unscaled dz */
        dz2 += dtauaff * dz1;
        scale(dz2, W_times_dzaff);

        /* W \ dsaff = -W * dzaff - lambda; */
        dsaff_by_W = -W_times_dzaff - w.lambda;

        /* dkapaff = -(bkap + kap * dtauaff) / tau; bkap = kap * tau*/
        const double dkapaff = -w.kap - w.kap / w.tau * dtauaff;

        /* Line search on W \ dsaff and W * dzaff */
        print_dbg("Performing line search on affine direction.\n");
        w.i.step_aff = lineSearch(w.lambda, dsaff_by_W, W_times_dzaff, w.tau, dtauaff, w.kap, dkapaff);

        /* Centering parameter */
        const double sigma = std::clamp(std::pow(1. - w.i.step_aff, 3),
                                        settings.sigmamin, settings.sigmamax);
        w.i.sigma = sigma;

        /* Combined search direction */
        RHScombined();
        print_dbg("Solving for combined search direction.\n");
        w.i.nitref3 = solveKKT(rhs2, dx2, dy2, dz2, 0);

        /* bkap = kap * tau + dkapaff * dtauaff - sigma * w.i.mu; */
        const double bkap = w.kap * w.tau + dkapaff * dtauaff - sigma * w.i.mu;

        /* dtau = ((1 - sigma) * rt - bkap / tau + c' * x2 + by2 + h' * z2) / dtau_denom; */
        const double dtau = ((1. - sigma) * rt - bkap / w.tau + c.dot(dx2) + b.dot(dy2) + h.dot(dz2)) / dtau_denom;

        /**
         * dx = x2 + dtau * x1
         * dy = y2 + dtau * y1
         * dz = z2 + dtau * z1
         */
        dx2 += dtau * dx1;
        dy2 += dtau * dy1;
        dz2 += dtau * dz1;

        /* ds_by_W = -(lambda \ bs + conelp_timesW(scaling, dz, dims))       */
        /* Note that at this point w->dsaff_by_W holds already (lambda \ ds) */
        scale(dz2, W_times_dzaff);
        dsaff_by_W = -(dsaff_by_W + W_times_dzaff);

        /* dkap = -(bkap + kap * dtau) / tau; */
        const double dkap = -(bkap + w.kap * dtau) / w.tau;

        /* Line search on combined direction */
        print_dbg("Performing line search on combined direction.\n");
        w.i.step = settings.gamma * lineSearch(w.lambda, dsaff_by_W, W_times_dzaff, w.tau, dtau, w.kap, dkap);

        /* Bring ds to the final unscaled form */
        /* ds = W * ds_by_W */
        scale(dsaff_by_W, dsaff);

        /* Update variables */
        w.x += w.i.step * dx2;
        w.y += w.i.step * dy2;
        w.z += w.i.step * dz2;
        w.s += w.i.step * dsaff;

        w.kap += w.i.step * dkap;
        w.tau += w.i.step * dtau;
    }

    /* Scale variables back */
    backscale();

    if (settings.verbose)
        print("Runtime: {}ms\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());

    return code;
}

/**
 * Scales variables by 1.0/tau, i.e. computes
 * x = x / tau
 * y = y / tau
 * z = z / tau
 * s = s / tau
 */
void Solver::backscale()
{
    w.x = w.x.cwiseQuotient(x_equil * w.tau);
    w.y = w.y.cwiseQuotient(A_equil * w.tau);
    w.z = w.z.cwiseQuotient(G_equil * w.tau);
    w.s = w.s.cwiseProduct(G_equil / w.tau);
}

/**
 * Prepares the RHS for computing the combined search direction.
 */
void Solver::RHScombined()
{
    Eigen::VectorXd ds1(n_ineq);
    Eigen::VectorXd ds2(n_ineq);

    /* ds = lambda o lambda + W \ s o Wz - sigma * mu * e) */
    conicProduct(w.lambda, w.lambda, ds1);
    conicProduct(dsaff_by_W, W_times_dzaff, ds2);

    const double sigmamu = w.i.sigma * w.i.mu;
    ds1.head(n_lc) += ds2.head(n_lc);
    ds1.head(n_lc).array() -= sigmamu;

    size_t k = n_lc;
    for (const SOCone &sc : so_cones)
    {
        ds1(k) -= sigmamu;
        ds1.segment(k, sc.dim) += ds2.segment(k, sc.dim);
        k += sc.dim;
    }

    /* dz = -(1 - sigma) * rz + W * (lambda \ ds) */
    conicDivision(w.lambda, ds1, dsaff_by_W);
    scale(dsaff_by_W, ds1);

    /* copy in RHS */
    const double one_minus_sigma = 1. - w.i.sigma;

    rhs2.head(n_var + n_eq) *= one_minus_sigma;
    rhs2.segment(n_var + n_eq, n_lc) = -one_minus_sigma * rz.head(n_lc) +
                                       ds1.head(n_lc);
    size_t rhs_index = n_var + n_eq + n_lc;
    k = n_lc;
    for (const SOCone &sc : so_cones)
    {
        rhs2.segment(rhs_index, sc.dim) = -one_minus_sigma * rz.segment(k, sc.dim) +
                                          ds1.segment(k, sc.dim);
        k += sc.dim;

        rhs_index += sc.dim;
        rhs2(rhs_index++) = 0.;
        rhs2(rhs_index++) = 0.;
    }
}

/**
 * Conic division, implements the "\" operator, v = u \ w
 */
void Solver::conicDivision(const Eigen::VectorXd &u,
                           const Eigen::VectorXd &w,
                           Eigen::VectorXd &v)
{
    /* LP cone */
    v.head(n_lc) = w.head(n_lc).cwiseQuotient(u.head(n_lc));

    /* SO cone */
    size_t cone_start = n_lc;
    for (const SOCone &sc : so_cones)
    {
        const double u0 = u(cone_start);
        const double w0 = w(cone_start);
        const double rho = u0 * u0 - u.segment(cone_start + 1, sc.dim - 1).squaredNorm();
        const double zeta = u.segment(cone_start + 1, sc.dim - 1).dot(w.segment(cone_start + 1, sc.dim - 1));
        const double factor = (zeta / u0 - w0) / rho;
        v(cone_start) = (u0 * w0 - zeta) / rho;
        v.segment(cone_start + 1, sc.dim - 1) = factor * u.segment(cone_start + 1, sc.dim - 1) +
                                                w.segment(cone_start + 1, sc.dim - 1) / u0;
        cone_start += sc.dim;
    }
}

/**
 * Conic product, implements the "o" operator, w = u o v
 * and returns e' * w (where e is the conic 1-vector)
 */
double Solver::conicProduct(const Eigen::VectorXd &u,
                            const Eigen::VectorXd &v,
                            Eigen::VectorXd &w)
{
    /* LP cone */
    w.head(n_lc) = u.head(n_lc).cwiseProduct(v.head(n_lc));
    double mu = w.head(n_lc).lpNorm<1>();

    /* SO cone */
    size_t cone_start = n_lc;
    for (const SOCone &sc : so_cones)
    {
        const double u0 = u(cone_start);
        const double v0 = v(cone_start);
        w(cone_start) = u.segment(cone_start, sc.dim).dot(v.segment(cone_start, sc.dim));
        mu += std::abs(w(cone_start));
        w.segment(cone_start + 1, sc.dim - 1) = u0 * v.segment(cone_start + 1, sc.dim - 1) +
                                                v0 * u.segment(cone_start + 1, sc.dim - 1);
        cone_start += sc.dim;
    }
    return mu;
}

double Solver::lineSearch(Eigen::VectorXd &lambda, Eigen::VectorXd &ds, Eigen::VectorXd &dz,
                          double tau, double dtau, double kap, double dkap)
{
    /* LP cone */
    double alpha;
    if (n_lc > 0)
    {
        const double rhomin = (ds.head(n_lc).cwiseQuotient(lambda.head(n_lc))).minCoeff();
        const double sigmamin = (dz.head(n_lc).cwiseQuotient(lambda.head(n_lc))).minCoeff();
        const double eps = 1e-13;
        if (-sigmamin > -rhomin)
        {
            alpha = sigmamin < 0. ? 1. / (-sigmamin) : 1. / eps;
        }
        else
        {
            alpha = rhomin < 0. ? 1. / (-rhomin) : 1. / eps;
        }
    }
    else
    {
        alpha = 10.;
    }

    /* tau and kappa */
    const double minus_tau_by_dtau = -tau / dtau;
    const double minus_kap_by_dkap = -kap / dkap;
    if (minus_tau_by_dtau > 0. and minus_tau_by_dtau < alpha)
    {
        alpha = minus_tau_by_dtau;
    }
    if (minus_kap_by_dkap > 0. and minus_kap_by_dkap < alpha)
    {
        alpha = minus_kap_by_dkap;
    }

    /* SO cone */
    size_t cone_start = n_lc;
    for (const SOCone &sc : so_cones)
    {
        /* Normalize */
        const double lknorm2 = std::pow(lambda(cone_start), 2) -
                               lambda.segment(cone_start + 1, sc.dim - 1).squaredNorm();
        if (lknorm2 <= 0.)
            continue;

        const double lknorm = std::sqrt(lknorm2);
        const Eigen::VectorXd lkbar = lambda.segment(cone_start, sc.dim) / lknorm;

        const double lknorminv = 1. / lknorm;

        /* Calculate products */
        const double lkbar_times_dsk = lkbar(0) * ds(cone_start) -
                                       lkbar.segment(1, sc.dim - 1).dot(ds.segment(cone_start + 1, sc.dim - 1));
        const double lkbar_times_dzk = lkbar(0) * dz(cone_start) -
                                       lkbar.segment(1, sc.dim - 1).dot(dz.segment(cone_start + 1, sc.dim - 1));

        /* Now construct rhok and sigmak, the first element is different */
        double factor;

        Eigen::VectorXd rho(sc.dim);
        rho(0) = lknorminv * lkbar_times_dsk;
        factor = (lkbar_times_dsk + ds(cone_start)) / (lkbar(0) + 1.);
        rho.tail(sc.dim - 1) = lknorminv * (ds.segment(cone_start + 1, sc.dim - 1) -
                                            factor * lkbar.segment(1, sc.dim - 1));
        const double rhonorm = rho.tail(sc.dim - 1).norm() - rho(0);

        Eigen::VectorXd sigma(sc.dim);
        sigma(0) = lknorminv * lkbar_times_dzk;
        factor = (lkbar_times_dzk + dz(cone_start)) / (lkbar(0) + 1.);
        sigma.tail(sc.dim - 1) = lknorminv * (dz.segment(cone_start + 1, sc.dim - 1) -
                                              factor * lkbar.segment(1, sc.dim - 1));
        const double sigmanorm = sigma.tail(sc.dim - 1).norm() - sigma(0);

        /* Update alpha */
        const double conic_step = std::max({0., sigmanorm, rhonorm});

        if (conic_step != 0.)
        {
            alpha = std::min(1. / conic_step, alpha);
        }

        cone_start += sc.dim;
    }

    /* Saturate between stepmin and stepmax */
    alpha = std::clamp(alpha, settings.stepmin, settings.stepmax);

    return alpha;
}

size_t Solver::solveKKT(const Eigen::VectorXd &rhs, // dim_K
                        Eigen::VectorXd &dx,        // n_var
                        Eigen::VectorXd &dy,        // n_eq
                        Eigen::VectorXd &dz,        // n_ineq
                        bool initialize)
{
    Eigen::VectorXd x = ldlt.solve(rhs);

    const double error_threshold = (1. + rhs.lpNorm<Eigen::Infinity>()) * settings.linsysacc;

    double nerr_prev = std::numeric_limits<double>::max(); // Previous refinement error
    Eigen::VectorXd dx_ref(dim_K);                         // Refinement vector

    const size_t mtilde = n_ineq + 2 * so_cones.size(); // Size of expanded cone block

    const Eigen::VectorXd &bx = rhs.head(n_var);
    const Eigen::VectorXd &by = rhs.segment(n_var, n_eq);
    const Eigen::VectorXd &bz = rhs.tail(mtilde);

    print_dbg("IR: it  ||ex||   ||ey||   ||ez|| (threshold: {:2.3e})\n", error_threshold);
    print_dbg("    --------------------------------------------------\n");

    /* Iterative refinement */
    size_t k_ref;
    for (k_ref = 0; k_ref <= settings.nitref; k_ref++)
    {
        /* Copy solution into arrays */
        const Eigen::VectorXd &dx = x.head(n_var);
        const Eigen::VectorXd &dy = x.segment(n_var, n_eq);
        dz.head(n_lc) = x.segment(n_var + n_eq, n_lc);
        size_t dz_index = n_lc;
        size_t x_index = n_var + n_eq + n_lc;
        for (const SOCone &sc : so_cones)
        {
            dz.segment(dz_index, sc.dim) = x.segment(x_index, sc.dim);
            dz_index += sc.dim;
            x_index += sc.dim + 2;
        }
        assert(dz_index == n_ineq and x_index == dim_K);

        /* Compute error term */

        /* Error on dx */
        /* ex = bx - A' * dy - G' * dz */
        Eigen::VectorXd ex = bx - Gt * dz;
        if (n_eq > 0)
        {
            ex -= At * dy;
        }
        ex -= settings.deltastat * dx;
        const double nex = ex.lpNorm<Eigen::Infinity>();

        /* Error on dy */
        /* ey = by - A * dx */
        Eigen::VectorXd ey = by;
        if (n_eq > 0)
        {
            ey -= A * dx;
        }
        ey += settings.deltastat * dy;
        const double ney = ey.lpNorm<Eigen::Infinity>();

        /* Error on ez */
        /* ez = bz - G * dx + V * dz_true */
        Eigen::VectorXd Gdx = G * dx;

        /* LP cone */
        Eigen::VectorXd ez(mtilde);
        ez.head(n_lc) = bz.head(n_lc) - Gdx.head(n_lc) +
                        settings.deltastat * dz.head(n_lc);

        /* SO cone */
        size_t ez_index = n_lc;
        dz_index = n_lc;
        for (const SOCone &sc : so_cones)
        {
            ez.segment(ez_index, sc.dim) = bz.segment(ez_index, sc.dim) -
                                           Gdx.segment(dz_index, sc.dim);
            ez.segment(ez_index, sc.dim - 1) += settings.deltastat * dz.segment(dz_index, sc.dim - 1);
            dz_index += sc.dim;
            ez_index += sc.dim;
            ez(ez_index - 1) -= settings.deltastat * dz(dz_index - 1);
            ez(ez_index++) = 0.;
            ez(ez_index++) = 0.;
        }
        assert(ez_index == mtilde and dz_index == n_ineq);

        const Eigen::VectorXd &dz_true = x.tail(mtilde);
        if (initialize)
        {
            ez += dz_true;
        }
        else
        {
            scale2add(dz_true, ez);
        }
        const double nez = ez.lpNorm<Eigen::Infinity>();

        print_dbg("     {}   {:.1g}    {:.1g}    {:.1g} \n", k_ref, nex, ney, nez);

        /* maximum error (infinity norm of e) */
        double nerr = std::max(nex, nez);
        if (n_eq > 0)
        {
            nerr = std::max(nerr, ney);
        }

        /* Check whether refinement brought decrease */
        if (k_ref > 0 and nerr > nerr_prev)
        {
            /* If not, undo and quit */
            x -= dx_ref;
            k_ref--;
            break;
        }

        /* Check whether to stop refining */
        if (k_ref == settings.nitref or
            (nerr < error_threshold) or
            (k_ref > 0 and nerr_prev < settings.irerrfact * nerr))
        {
            break;
        }
        nerr_prev = nerr;

        /* Solve for refinement */
        Eigen::VectorXd e(dim_K);
        e << ex, ey, ez;
        dx_ref = ldlt.solve(e);

        /* Add refinement to x */
        x += dx_ref;
    }

    /* Copy solution into arrays */
    dx = x.head(n_var);
    dy = x.segment(n_var, n_eq);
    dz.head(n_lc) = x.segment(n_var + n_eq, n_lc);
    size_t dz_index = n_lc;
    size_t x_index = n_var + n_eq + n_lc;
    for (const SOCone &sc : so_cones)
    {
        dz.segment(dz_index, sc.dim) = x.segment(x_index, sc.dim);
        dz_index += sc.dim;
        x_index += sc.dim + 2;
    }
    assert(dz_index == n_ineq and x_index == dim_K);

    return k_ref;
}

/**
 *                                            [ D   v   u  ]
 * Fast multiplication with V = W^2 = eta^2 * [ v'  1   0  ] 
 *                                            [ u'  0  -1  ]
 * Computes y += W^2 * x;
 * 
 */
void Solver::scale2add(const Eigen::VectorXd &x, Eigen::VectorXd &y)
{
    /* LP cone */
    y.head(n_lc) += lp_cone.v.cwiseProduct(x.head(n_lc));

    /* SO cone */
    size_t cone_start = n_lc;
    for (const SOCone &sc : so_cones)
    {
        const size_t i1 = cone_start;
        const size_t i2 = i1 + 1;
        const size_t i3 = i2 + sc.dim - 1;
        const size_t i4 = i3 + 1;

        /* y1 += d1 * x1 + u0 * x4 */
        y(i1) += sc.eta_square * (sc.d1 * x(i1) + sc.u0 * x(i4));

        /* y2 += x2 + v1 * q * x3 + u1 * q * x4 */
        const double v1x3_plus_u1x4 = sc.v1 * x(i3) + sc.u1 * x(i4);
        y.segment(i2, sc.dim - 1) += sc.eta_square * (x.segment(i2, sc.dim - 1) +
                                                      v1x3_plus_u1x4 * sc.q);

        const double qtx2 = sc.q.dot(x.segment(i2, sc.dim - 1));

        /* y3 += v1 * q' * x2 + x3 */
        y(i3) += sc.eta_square * (sc.v1 * qtx2 + x(i3));

        /* y4 += u0 * x1 + u1 * q' * x2 - x4 */
        y(i4) = sc.eta_square * (sc.u0 * x(i1) + sc.u1 * qtx2 - x(i4));

        /* prepare index for next cone */
        cone_start += sc.dim + 2;
    }
}

/**
 * Prepares the affine RHS for KKT system.
 * Given the special way we store the KKT matrix (sparse representation
 * of the scalings for the second-order cone), we need this to prepare
 * the RHS before solving the KKT system in the special format.
 */
void Solver::RHSaffine()
{
    /* LP cone */
    rhs2.head(n_var + n_eq) << rx, -ry;

    /* SO cone */
    rhs2.segment(n_var + n_eq, n_lc) = w.s.head(n_lc) - rz.head(n_lc);
    size_t rhs_index = n_var + n_eq + n_lc;
    size_t rz_index = n_lc;
    for (const SOCone &sc : so_cones)
    {
        rhs2.segment(rhs_index, sc.dim) =
            w.s.segment(rz_index, sc.dim) - rz.segment(rz_index, sc.dim);
        rz_index += sc.dim;

        rhs_index += sc.dim;
        rhs2.segment(rhs_index, 2).setZero();
        rhs_index += 2;
    }
}

void Solver::updateKKTScalings()
{
    size_t ptr_i = 0;

    /* LP cone */
    for (size_t k = 0; k < n_lc; k++)
    {
        *KKT_V_ptr[ptr_i++] = -lp_cone.v(k) - settings.deltastat;
    }

    /* SO cone */
    for (const SOCone &sc : so_cones)
    {
        /* D */
        *KKT_V_ptr[ptr_i++] = -sc.eta_square * sc.d1 - settings.deltastat;

        for (size_t k = 1; k < sc.dim; k++)
        {
            *KKT_V_ptr[ptr_i++] = -sc.eta_square - settings.deltastat;
        }

        /* diagonal */
        *KKT_V_ptr[ptr_i++] = -sc.eta_square;

        /* v */
        for (size_t k = 1; k < sc.dim; k++)
        {
            *KKT_V_ptr[ptr_i++] = -sc.eta_square * sc.v1 * sc.q(k - 1);
        }

        /* diagonal */
        *KKT_V_ptr[ptr_i++] = sc.eta_square + settings.deltastat;

        /* u */
        *KKT_V_ptr[ptr_i++] = -sc.eta_square * sc.u0;
        for (size_t k = 1; k < sc.dim; k++)
        {
            *KKT_V_ptr[ptr_i++] = -sc.eta_square * sc.u1 * sc.q(k - 1);
        }
    }
    assert(ptr_i == KKT_V_ptr.size());
}

void Solver::setupKKT()
{
    /**
     *      [ 0  A' G']
     *  K = [ A  0  0 ]
     *      [ G  0 -V ]
     * 
     *   V = blkdiag(I, blkdiag(I, 1, -1), ...,  blkdiag(I, 1, -1));
     *                    ^   number of second-order cones   ^
     *               ^ dimension of linear contraints
     * 
     *  Only the upper triangular part is constructed here.
     */
    K.resize(dim_K, dim_K);

    /* Number of non-zeros in KKT matrix */
    size_t K_nonzeros = At.nonZeros() + Gt.nonZeros();
    /* Static regularization */
    K_nonzeros += n_var + n_eq;
    /* Linear part of scaling block V */
    K_nonzeros += n_lc;
    for (const SOCone &sc : so_cones)
    {
        /* SOC part of scaling block V */
        K_nonzeros += 3 * sc.dim + 1;
    }
    K.reserve(K_nonzeros);

    std::vector<Eigen::Triplet<double>> K_triplets;
    K_triplets.reserve(K_nonzeros);

    /* I (1,1) Static regularization */
    for (size_t k = 0; k < n_var; k++)
    {
        K_triplets.emplace_back(k, k, settings.deltastat);
    }
    /* I (2,2) Static regularization */
    for (size_t k = n_var; k < n_var + n_eq; k++)
    {
        K_triplets.emplace_back(k, k, -settings.deltastat);
    }

    size_t col_K = n_var;

    /* A' (1,2) */
    for (int col = 0; col < At.cols(); col++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(At, col); it; ++it)
        {
            K_triplets.emplace_back(it.row(), A.cols() + col, it.value());
        }
        col_K++;
    }

    /* G' (1,3) */
    {
        size_t col_Gt = 0;

        /* Linear block */
        for (size_t col = 0; col < n_lc; col++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Gt, col_Gt); it; ++it)
            {
                K_triplets.emplace_back(it.row(), col_K, it.value());
            }
            col_Gt++;
            col_K++;
        }

        /* SOC blocks */
        for (const SOCone &sc : so_cones)
        {
            for (size_t col = 0; col < sc.dim; col++)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Gt, col_Gt); it; ++it)
                {
                    K_triplets.emplace_back(it.row(), col_K, it.value());
                }
                col_Gt++;
                col_K++;
            }
            col_K += 2;
        }
        assert(col_K == size_t(K.cols()));
        assert(col_Gt == size_t(Gt.cols()));
    }

    /* -V (3,3) */
    {
        size_t diag_idx = n_var + n_eq;

        /* First identity block */
        for (size_t k = 0; k < n_lc; k++)
        {
            K_triplets.emplace_back(diag_idx, diag_idx, -1.);
            diag_idx++;
        }

        /* SOC blocks */
        /**
         * The scaling matrix has the following structure:
         *
         *    [ 1                * ]
         *    [   1           *  * ]
         *    [     .         *  * ]      
         *    [       .       *  * ]       [ D   v   u ]      D: Identity of size conesize       
         *  - [         .     *  * ]  =  - [ u'  1   0 ]      v: Vector of size conesize - 1      
         *    [           1   *  * ]       [ v'  0' -1 ]      u: Vector of size conesize    
         *    [             1 *  * ]
         *    [   * * * * * * 1    ]
         *    [ * * * * * * *   -1 ]
         *
         *  Only the upper triangular part is constructed here.
         */
        for (const SOCone &sc : so_cones)
        {
            /* D */
            for (size_t k = 0; k < sc.dim; k++)
            {
                K_triplets.emplace_back(diag_idx, diag_idx, -1.);
                diag_idx++;
            }

            /* -1 on diagonal */
            K_triplets.emplace_back(diag_idx, diag_idx, -1.);

            /* -v */
            for (size_t k = 1; k < sc.dim; k++)
            {
                K_triplets.emplace_back(diag_idx - sc.dim + k, diag_idx, 0.);
            }
            diag_idx++;

            /* 1 on diagonal */
            K_triplets.emplace_back(diag_idx, diag_idx, 1.);

            /* -u */
            for (size_t k = 0; k < sc.dim; k++)
            {
                K_triplets.emplace_back(diag_idx - sc.dim - 1 + k, diag_idx, 0.);
            }
            diag_idx++;
        }
        assert(diag_idx == dim_K);
    }

    assert(size_t(K_triplets.size()) == K_nonzeros);

    K.setFromTriplets(K_triplets.begin(), K_triplets.end());

    assert(size_t(K.nonZeros()) == K_nonzeros);

    print_dbg("Dimension of KKT matrix: {}\n", dim_K);
    print_dbg("Non-zeros in KKT matrix: {}\n", K.nonZeros());

    cacheIndices();
}

/**
 * Save pointers for fast access
 */
void Solver::cacheIndices()
{
    /* A AND G MATRICES */

    size_t col_K = n_var;

    /* A' (1,2) */
    for (int col = 0; col < At.cols(); col++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(At, col); it; ++it)
        {
            KKT_AG_ptr.push_back(&K.coeffRef(it.row(), col_K));
        }
        col_K++;
    }

    /* G' (1,3) */
    {
        size_t col_Gt = 0;

        /* Linear block */
        for (size_t col = 0; col < n_lc; col++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Gt, col_Gt); it; ++it)
            {
                KKT_AG_ptr.push_back(&K.coeffRef(it.row(), col_K));
            }
            col_Gt++;
            col_K++;
        }

        /* SOC blocks */
        for (const SOCone &sc : so_cones)
        {
            for (size_t col = 0; col < sc.dim; col++)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Gt, col_Gt); it; ++it)
                {
                    KKT_AG_ptr.push_back(&K.coeffRef(it.row(), col_K));
                }
                col_Gt++;
                col_K++;
            }
            col_K += 2;
        }
        assert(col_K == size_t(K.cols()));
        assert(col_Gt == size_t(Gt.cols()));
    }

    /* SCALING AND RESIDUALS -V (3,3) */

    /* LP cone */
    size_t diag_idx = n_var + n_eq;
    for (size_t k = 0; k < n_lc; k++)
    {
        KKT_V_ptr.push_back(&K.coeffRef(diag_idx, diag_idx));
        diag_idx++;
    }

    /* SO cone */
    for (const SOCone &sc : so_cones)
    {
        /* D */
        KKT_V_ptr.push_back(&K.coeffRef(diag_idx, diag_idx));
        diag_idx++;
        for (size_t k = 1; k < sc.dim; k++)
        {
            KKT_V_ptr.push_back(&K.coeffRef(diag_idx, diag_idx));
            diag_idx++;
        }

        /* diagonal */
        KKT_V_ptr.push_back(&K.coeffRef(diag_idx, diag_idx));

        /* v */
        for (size_t k = 1; k < sc.dim; k++)
        {
            KKT_V_ptr.push_back(&K.coeffRef(diag_idx - sc.dim + k, diag_idx));
        }
        diag_idx++;

        /* diagonal */
        KKT_V_ptr.push_back(&K.coeffRef(diag_idx, diag_idx));

        /* u */
        KKT_V_ptr.push_back(&K.coeffRef(diag_idx - sc.dim - 1, diag_idx));
        for (size_t k = 1; k < sc.dim; k++)
        {
            KKT_V_ptr.push_back(&K.coeffRef(diag_idx - sc.dim - 1 + k, diag_idx));
        }
        diag_idx++;
    }
    assert(diag_idx == dim_K);
}

void Solver::updateKKTAG()
{
    size_t ptr_i = 0;

    /* A' (1,2) */
    for (int col = 0; col < At.cols(); col++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(At, col); it; ++it)
        {
            *KKT_AG_ptr[ptr_i++] = it.value();
        }
    }

    /* G' (1,3) */
    {
        size_t col_Gt = 0;

        /* Linear block */
        for (size_t col = 0; col < n_lc; col++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Gt, col_Gt); it; ++it)
            {
                *KKT_AG_ptr[ptr_i++] = it.value();
            }
            col_Gt++;
        }

        /* SOC blocks */
        for (const SOCone &sc : so_cones)
        {
            for (size_t col = 0; col < sc.dim; col++)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Gt, col_Gt); it; ++it)
                {
                    *KKT_AG_ptr[ptr_i++] = it.value();
                }
                col_Gt++;
            }
        }
    }
}

void Solver::updateData(const Eigen::SparseMatrix<double> &G,
                        const Eigen::SparseMatrix<double> &A,
                        const Eigen::VectorXd &c,
                        const Eigen::VectorXd &h,
                        const Eigen::VectorXd &b)
{
    for (int i = 0; i < G.nonZeros(); i++)
        this->G.valuePtr()[i] = G.valuePtr()[i];
    for (int i = 0; i < A.nonZeros(); i++)
        this->A.valuePtr()[i] = A.valuePtr()[i];
    this->c = c;
    this->h = h;
    this->b = b;

    setEquilibration();

    Gt = this->G.transpose();
    At = this->A.transpose();

    updateKKTAG();
}

void Solver::updateData(double *Gpr, double *Apr,
                        double *c, double *h, double *b)
{
    if (equibrilated)
        unsetEquilibration();

    if (Gpr)
    {
        for (int i = 0; i < G.nonZeros(); i++)
            G.valuePtr()[i] = Gpr[i];
        this->h = Eigen::Map<Eigen::VectorXd>(h, n_ineq);
    }
    if (Apr)
    {
        for (int i = 0; i < A.nonZeros(); i++)
            A.valuePtr()[i] = Apr[i];
        this->b = Eigen::Map<Eigen::VectorXd>(b, n_eq);
    }
    if (c)
    {
        this->c = Eigen::Map<Eigen::VectorXd>(c, n_var);
    }

    setEquilibration();

    Gt = this->G.transpose();
    At = this->A.transpose();

    updateKKTAG();
}

// void Solver::saveProblemData(const std::string &path)
// {
//     std::ofstream out(path);

//     Eigen::IOFormat formatter(Eigen::FullPrecision, Eigen::DontAlignCols, ", ");

//     print(out, "idxint n = {};\n", n_var);
//     print(out, "idxint m = {};\n", n_ineq);
//     print(out, "idxint p = {};\n", n_eq);
//     print(out, "idxint l = {};\n", n_lc);
//     print(out, "idxint ncones = {};\n", n_sc);

//     if (n_sc > 0)
//     {
//         Eigen::VectorXi q(n_sc);
//         for (int i = 0; i < q.size(); i++)
//         {
//             q(i) = so_cones[i].dim;
//         }
//         print(out, "idxint q[{}] = {{{}}};\n", q.size(), q.transpose().format(formatter));
//     }

//     print(out, "pfloat c[{}] = {{{}}};\n", c.size(), c.transpose().format(formatter));

//     if (n_eq > 0)
//     {
//         print(out, "pfloat b[{}] = {{{}}};\n", b.size(), b.transpose().format(formatter));
//     }
//     else
//     {
//         print(out, "pfloat *b = NULL;\n");
//     }

//     if (n_ineq > 0)
//     {
//         print(out, "pfloat h[{}] = {{{}}};\n", h.size(), h.transpose().format(formatter));
//     }
//     else
//     {
//         print(out, "pfloat *h = NULL;\n");
//     }

//     if (G.nonZeros() > 0)
//     {
//         print(out, "idxint Gjc[{}] = {{{}}};\n",
//               G.nonZeros(),
//               Eigen::Map<Eigen::VectorXi>(G.outerIndexPtr(), G.nonZeros()).transpose().format(formatter));
//         print(out, "idxint Gir[{}] = {{{}}};\n",
//               G.nonZeros(),
//               Eigen::Map<Eigen::VectorXi>(G.innerIndexPtr(), G.nonZeros()).transpose().format(formatter));
//         print(out, "pfloat Gpr[{}] = {{{}}};\n",
//               G.nonZeros(),
//               Eigen::Map<Eigen::VectorXd>(G.valuePtr(), G.nonZeros()).transpose().format(formatter));
//     }
//     else
//     {
//         print(out, "pfloat *Gpr = NULL;\n");
//         print(out, "idxint *Gir = NULL;\n");
//         print(out, "idxint *Gpr = NULL;\n");
//     }

//     if (A.nonZeros() > 0)
//     {
//         print(out, "idxint Ajc[{}] = {{{}}};\n",
//               A.nonZeros(),
//               Eigen::Map<Eigen::VectorXi>(A.outerIndexPtr(), A.nonZeros()).transpose().format(formatter));
//         print(out, "idxint Air[{}] = {{{}}};\n",
//               A.nonZeros(),
//               Eigen::Map<Eigen::VectorXi>(A.innerIndexPtr(), A.nonZeros()).transpose().format(formatter));
//         print(out, "pfloat Apr[{}] = {{{}}};\n",
//               A.nonZeros(),
//               Eigen::Map<Eigen::VectorXd>(A.valuePtr(), A.nonZeros()).transpose().format(formatter));
//     }
//     else
//     {
//         print(out, "pfloat *Apr = NULL;\n");
//         print(out, "idxint *Air = NULL;\n");
//         print(out, "idxint *Apr = NULL;\n");
//     }
//     out.close();
// }

} // namespace EiCOS