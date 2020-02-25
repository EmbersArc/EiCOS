#include "ecos.hpp"

#include <Eigen/SparseCholesky>

ECOSEigen::ECOSEigen(const Eigen::SparseMatrix<double> &G,
                     const Eigen::SparseMatrix<double> &A,
                     const Eigen::SparseVector<double> &c,
                     const Eigen::SparseVector<double> &h,
                     const Eigen::SparseVector<double> &b,
                     const std::vector<size_t> &soc_dims)
    : G(G), A(A), c(c), h(h), b(b)
{
    num_var = A.cols();
    num_eq = A.rows();
    num_sc = soc_dims.size();
    num_ineq = G.rows(); // = num_pc + num_sc
    num_pc = num_ineq - num_sc;

    so_cones.resize(num_sc);
    for (size_t i = 0; i < num_sc; i++)
    {
        so_cones[i].dim = soc_dims[i];
    }

    setupKKT(G, A);
}

/**
 * Update scalings.
 * Returns false as soon as any multiplier or slack leaves the cone,
 * as this indicates severe problems.
 */
bool ECOSEigen::updateScalings()
{
    /* LP cone */
    lp_cone.w = s.cwiseQuotient(z).cwiseSqrt();

    /* Second-order cone */
    size_t k = num_pc;
    for (SecondOrderCone soc : so_cones)
    {
        /* check residuals and quit if they're negative */
        const double sres = s(k) * s(k) - s.segment(k + 1, soc.dim - 1).squaredNorm();
        const double zres = z(k) * z(k) - z.segment(k + 1, soc.dim - 1).squaredNorm();
        if (sres <= 0 or zres <= 0)
        {
            return false;
        }

        /* normalize variables */
        double snorm = std::sqrt(sres);
        double znorm = std::sqrt(zres);

        const Eigen::VectorXd skbar = s.segment(num_pc, soc.dim) / snorm;
        const Eigen::VectorXd zkbar = z.segment(num_pc, soc.dim) / znorm;

        soc.eta_square = snorm / znorm;
        soc.eta = std::sqrt(soc.eta_square);

        /* Normalized Nesterov-Todd scaling point */
        double gamma = 1.;
        gamma += skbar.dot(zkbar);
        gamma = std::sqrt(0.5 * gamma);

        double a = (0.5 / gamma) * (skbar(0) + zkbar(0));
        Eigen::VectorXd q = (0.5 / gamma) * (skbar.tail(soc.dim - 1) - zkbar.tail(soc.dim - 1));
        double w = q.squaredNorm();

        /* Pre-compute variables needed for KKT matrix (kkt_update uses those) */

        double c = (1. + a) + w / (1. + a);
        double d = 1. + 2. / (1. + a) + w / ((1. + a) * (1. + a));

        double d1 = std::max(0., 0.5 * (a * a + w * (1.0 - (c * c) / (1. + w * d))));
        double u0_square = a * a + w - soc.d1;
        double u0 = std::sqrt(u0_square);

        double c2byu02 = (c * c) / u0_square;
        double c2byu02_d = c2byu02 - d;
        if (c2byu02_d <= 0)
        {
            return false;
        }

        soc.v1 = std::sqrt(c2byu02_d);
        soc.u1 = std::sqrt(c2byu02);
        soc.d1 = d1;
        soc.u0 = u0;

        /* increase offset for next cone */
        k += soc.dim;
    }
    /* lambda = W*z */
    scale();

    return true;
}

/**
 * Fast multiplication by scaling matrix.
 * Returns lambda = W*z
 * The exponential variables are not touched.
 */
void ECOSEigen::scale()
{
    /* LP cone */
    size_t p = num_pc;
    lambda.head(p) = lp_cone.w.head(p).cwiseProduct(z.head(p));

    /* Second-order cone */
    size_t cone_start = num_pc;
    for (SecondOrderCone &soc : so_cones)
    {
        /* zeta = q'*z1 */
        double zeta = soc.q.tail(soc.dim - 1).dot(z.segment(cone_start + 1, soc.dim - 1));

        /* factor = z0 + zeta / (1+a); */
        double factor = z(cone_start) + zeta / (1. + soc.a);

        /* second pass (on k): write out result */
        lambda(cone_start) = soc.eta * (soc.a * z(cone_start) + zeta); /* lambda[0] */
        lambda.segment(cone_start + 1, p - 1) = soc.eta * (z + factor * soc.q.tail(p - 1));

        cone_start += p;
    }
}

/**
 * This function is reponsible for checking the exit/convergence conditions of ECOS.
 * If one of the exit conditions is met, ECOS displays an exit message and returns
 * the corresponding exit code. The calling function must then make sure that ECOS
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
 * If none of the exit tests are met, the function returns ECOS_NOT_CONVERGED_YET.
 * This should not be an exitflag that is ever returned to the outside world.
 **/
bool ECOSEigen::checkExitConditions(bool reduced_accuracy)
{
    double feastol;
    double abstol;
    double reltol;

    /* Set accuracy against which to check */
    if (reduced_accuracy)
    {
        /* Check convergence against normal precisions */
        feastol = settings.feastol;
        abstol = settings.abstol;
        reltol = settings.reltol;
    }
    else
    {
        /* Check convergence against reduced precisions */
        feastol = settings.feastol_inacc;
        abstol = settings.abstol_inacc;
        reltol = settings.reltol_inacc;
    }

    /* Optimal? */
    if ((-cx > 0 or -by - hz >= -abstol) and
        (info.pres < feastol and info.dres < feastol) and
        (info.gap < abstol or info.relgap < reltol))
    {
        info.pinf = false;
        info.dinf = false;
        return true;
    }

    /* Dual infeasible? */
    else if ((info.dinfres.has_value()) and (info.dinfres.value() < feastol) and (tau < kap))
    {
        info.pinf = false;
        info.dinf = true;
        return false;
    }

    /* Primal infeasible? */
    else if (((info.pinfres.has_value() and info.pinfres < feastol) and (tau < kap)) or
             (tau < feastol and kap < feastol and info.pinfres < feastol))
    {
        info.pinf = true;
        info.dinf = false;
        return false;
    }

    /* Indicate if none of the above criteria are met */
    else
    {
        return false;
    }
}

void ECOSEigen::computeResiduals()
{
    /**
    * hrx = -A' * y - G' * z       rx = hrx - tau * c      hresx = ||rx||_2
    * hry =  A * x                 ry = hry - tau * b      hresy = ||ry||_2
    * hrz =  s + G * x             rz = hrz - tau * h      hresz = ||rz||_2
    * 
    * rt = kappa + c'*x + b'*y + h'*z
    **/

    /* rx = -A' * y - G' * z - tau * c */
    if (num_eq > 0)
    {
        rx = -At * y - Gt * z;
    }
    else
    {
        rx = -Gt * z;
    }
    rx -= tau * c;
    hresx = rx.norm();

    /* ry = A * x - tau * b */
    if (num_eq > 0)
    {
        ry = A * x;
        hresy = ry.norm();
        ry -= tau * b;
    }
    else
    {
        hresy = 0.;
    }

    /* rz = s + G * x - tau * h */
    rz = s + G * x;
    hresz = rz.norm();
    rz -= tau * h;

    nx = x.norm();
    ny = y.norm();
    nz = z.norm();
    ns = s.norm();
}

void ECOSEigen::updateStatistics()
{
    info.gap = s.dot(z);
    info.mu = (info.gap + kap * tau) / (D + 1.);
    info.kapovert = kap / tau;
    info.pcost = cx / tau;
    info.dcost = -(hz + by) / tau;

    /* Relative Duality Gap */
    if (info.pcost < 0)
    {
        info.relgap = info.gap / -info.pcost;
    }
    else if (info.dcost > 0)
    {
        info.relgap = info.gap / info.dcost;
    }
    else
    {
        // fail
    }

    /* Residuals */
    double nry = num_eq > 0 ? ry.norm() / std::max(resy0 + nx, 1.) : 0.;
    double nrz = rz.norm() / std::max(resz0 + nx + ns, 1.);
    info.pres = std::max(nry, nrz) / tau;
    info.dres = rx.norm() / std::max(resx0 + ny + nz, 1.) / tau;

    /* Infeasibility Measures */
    if ((hz + by) / std::max(ny + nz, 1.) < -settings.reltol)
    {
        info.pinfres = hresx / std::max(ny + nz, 1.);
    }
    if (cx / std::max(nx, 1.) < -settings.reltol)
    {
        info.dinfres = std::max(hresy / std::max(nx, 1.), hresz / std::max(nx + ns, 1.));
    }
}

/**
 * Scales a conic variable such that it lies strictly in the cone.
 * If it is already in the cone, r is simply copied to s.
 * Otherwise s = r + (1 + alpha) * e where alpha is the biggest residual.
 */
void ECOSEigen::bringToCone(Eigen::VectorXd &x)
{
    double alpha = -0.99;

    // ===== 1. Find maximum residual =====

    /* Positive Orthant */
    size_t i;
    for (i = 0; i < num_pc; i++)
    {
        if (x[i] <= 0 and -x[i] > alpha)
        {
            alpha = -x[i];
        }
    }

    /* Second-Order Cone */
    double cres;
    for (const SecondOrderCone &sc : so_cones)
    {
        cres = x[i];
        i++;
        cres -= x.segment(i, sc.dim - 1).norm();
        i += sc.dim - 1;

        if (cres <= 0 and -cres > alpha)
        {
            alpha = -cres;
        }
    }

    // ===== 2. Compute s = r + (1 + alpha) * e =====

    alpha += 1.;

    /* Positive Orthant */
    x.head(num_pc).array() += alpha;

    /* Second-order cone */
    i = num_pc;
    for (const SecondOrderCone &sc : so_cones)
    {
        x[i] += alpha;
        i += sc.dim;
    }
}

void ECOSEigen::Solve()
{
    // Equilibrate
    c.cwiseQuotient(x_equil);

    /**
    * Set up first right hand side
    * [ 0 ]
    * [ b ]
    * [ h ]
    **/
    rhs1.resize(K.rows());
    rhs1.setZero();
    rhs1.segment(num_var, num_eq) = b;
    rhs1.segment(num_var + num_eq, num_pc) = h.head(num_pc);
    size_t h_index = num_pc;
    size_t rhs1_index = num_var + num_eq + num_pc;
    for (const SecondOrderCone &sc : so_cones)
    {
        rhs1.segment(rhs1_index, sc.dim) = h.segment(h_index, sc.dim);
        h_index += sc.dim;
        rhs1_index += sc.dim + 2;
    }

    /**
    * Set up second right hand side
    * [-c ]
    * [ 0 ]
    * [ 0 ]
    **/
    rhs2.resize(K.rows());
    rhs2.setZero();
    rhs2.head(num_var) = -c;

    // Set up scalings of problem data
    scale_rx = c.norm();
    scale_ry = b.norm();
    scale_rz = h.norm();
    resx0 = std::max(1., scale_rx);
    resy0 = std::max(1., scale_ry);
    resz0 = std::max(1., scale_rz);

    // Do LDLT factorization
    ldlt.analyzePattern(K);
    ldlt.factorize(K);

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
	 **/
    const Eigen::VectorXd sol1 = ldlt.solve(rhs1);

    const Eigen::VectorXd x = sol1.head(num_var);
    Eigen::VectorXd r = -sol1.segment(num_var, num_eq);
    bringToCone(r);

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
	 **/
    const Eigen::VectorXd sol2 = ldlt.solve(rhs2);

    const Eigen::VectorXd y = sol2.segment(num_var, num_eq);
    Eigen::VectorXd z = sol2.segment(num_var, num_eq);
    bringToCone(z);

    /**
    * Modify first right hand side
    * [ 0 ]    [-c ] 
    * [ b ] -> [ b ] 
    * [ h ]    [ h ] 
    **/
    rhs1.head(num_var) = -c;

    bool done = false;
    for (iteration = 0; iteration < info.iter_max; iteration++)
    {
        computeResiduals();
        updateStatistics();
        done = checkExitConditions(false);

        if (done)
        {
            break;
        }

        updateKKT();

        Eigen::VectorXd dx, dy, dz;

        /* Solve for RHS1, which is used later also in combined direction */
        solveKKT(rhs1, dx, dy, dz, true);

        /* AFFINE SEARCH DIRECTION (predictor, need dsaff and dzaff only) */
        RHS_affine();
        solveKKT(rhs2, dx, dy, dz, true);
    }
}

void ECOSEigen::solveKKT(const Eigen::VectorXd &rhs,
                         Eigen::VectorXd &dx,
                         Eigen::VectorXd &dy,
                         Eigen::VectorXd &dz,
                         bool initialized)
{
    /* forward - diagonal - backward solves */
    Eigen::VectorXd x = ldlt.solve(rhs1);

    // TODO: Assign those correctly:
    Eigen::VectorXd bx, by, bz;

    const double error_threshold = (1. + rhs.lpNorm<1>()) * settings.linsysacc;

    double nerr_prev = std::numeric_limits<double>::max(); // Previous refinement error
    Eigen::VectorXd dx_ref;                                // Refinement vector

    /* iterative refinement */
    for (size_t kItRef = 0; kItRef <= settings.nitref; kItRef++)
    {
        /* copy solution into arrays */
        const Eigen::VectorXd dx = x.head(num_var);
        const Eigen::VectorXd dy = x.segment(num_var, num_eq);
        Eigen::VectorXd dz;
        dz.resize(num_ineq);
        dz.head(num_pc) = x.segment(num_var + num_eq, num_pc);
        size_t dz_index = 0;
        size_t x_index = 0;
        for (const SecondOrderCone &sc : so_cones)
        {
            dz.segment(dz_index, sc.dim) = x.segment(num_var + num_eq + num_pc + dz_index, sc.dim);
            dz_index += sc.dim;
            x_index += sc.dim + 2;
        }

        /* compute error term */

        /* 1. error on dx */
        /* ex = bx - A' * dy - G' * dz */
        Eigen::VectorXd ex = bx;
        if (num_eq > 0)
        {
            ex -= A.transpose() * dy;
        }
        ex -= G.transpose() * dz;
        const double nex = ex.lpNorm<1>();

        /* 2, error on dy */
        Eigen::VectorXd ey;
        ey.resize(num_eq);
        if (num_eq > 0)
        {
            /* ey = by - A * dx */
            ey = by - A * dx;
        }
        const double ney = ey.lpNorm<1>();

        /* 3. ez = bz - G * dx + V * dz_true */
        Eigen::VectorXd ez, Gdx;
        ez.resize(num_ineq);
        ez.setZero();
        Gdx = G * dx;
        ez.head(num_pc) = bz.head(num_pc) - Gdx.head(num_pc);
        size_t ez_index = num_pc;
        size_t bz_index = num_pc;
        for (const SecondOrderCone &sc : so_cones)
        {
            ez.segment(ez_index, sc.dim) = bz.segment(bz_index, sc.dim);
            ez_index += sc.dim;
            bz_index += sc.dim;
            ez.segment(ez_index, 2).setZero();
            ez_index += 2;
        }

        const size_t mtilde = num_ineq + 2 * so_cones.size();
        const Eigen::VectorXd truez = x.segment(num_var + num_eq, mtilde);

        if (not initialized)
        {
            scale2add(truez, ez);
        }
        else
        {
            ez += truez;
        }

        const double nez = ez.lpNorm<1>();

        /* maximum error (infinity norm of e) */
        double nerr = std::max(nex, nez);

        if (num_eq > 0)
        {
            nerr = std::max(nerr, ney);
        }

        /* Check whether refinement brought decrease */
        if (kItRef > 0 && nerr > nerr_prev)
        {
            /* If not, undo and quit */
            x -= dx;
            kItRef--;
            break;
        }

        /* Check whether to stop refining */
        if (kItRef == settings.nitref or
            (nerr < error_threshold) or
            (kItRef > 0 and nerr_prev < settings.irerrfact * nerr))
        {
            break;
        }
        nerr_prev = nerr;

        Eigen::VectorXd e(ex.size() + ey.size() + ez.size());
        e << ex, ey, ez;
        dx_ref = ldlt.solve(e);

        /* Add refinement to x*/
        x += dx_ref;
    }
}

/**
 *                                       [ D   v   u  ]
 * Slow multiplication with V =  eta^2 * [ v'  1   0  ] = W^2
 *                                       [ u   0  -1  ]
 * Computes y += W^2*x;
 */
void ECOSEigen::scale2add(const Eigen::VectorXd &x, Eigen::VectorXd &y)
{
    /* LP cone */
    y.head(num_pc) += lp_cone.v.cwiseProduct(x.head(num_pc));

    /* Second-order cone */
    size_t cone_start = num_pc;
    for (const SecondOrderCone &sc : so_cones)
    {
        const size_t dim = sc.dim + 2;
        Eigen::MatrixXd W_squared = Eigen::MatrixXd::Identity(dim, dim);

        // diagonal
        W_squared(0, 0) = sc.d1;
        W_squared(dim - 1, dim - 1) = -1.;

        // v
        W_squared.col(dim - 2).segment(1, sc.dim - 1).setConstant(sc.v1);
        // v'
        W_squared.row(dim - 2).segment(1, sc.dim - 1).setConstant(sc.v1);

        // u
        W_squared.col(dim - 1)(0) = sc.u0;
        W_squared.col(dim - 1).segment(1, sc.dim - 1).setConstant(sc.u1);
        // u'
        W_squared.row(dim - 1)(0) = sc.u0;
        W_squared.row(dim - 1).segment(1, sc.dim - 1).setConstant(sc.u1);

        W_squared *= sc.eta_square;

        y.segment(cone_start, dim) += W_squared * x.segment(cone_start, dim);

        cone_start += dim;
    }
}

/**
 * Prepares the affine RHS for KKT system.
 * Given the special way we store the KKT matrix (sparse representation
 * of the scalings for the second-order cone), we need this to prepare
 * the RHS before solving the KKT system in the special format.
 */
void ECOSEigen::RHS_affine()
{
    rhs2.head(num_var + num_eq) << rx, -ry;

    rhs2.segment(num_var + num_eq, num_pc) = rz.head(num_pc);

    size_t rhs_index = num_pc;
    size_t rz_index = num_pc;
    for (const SecondOrderCone &sc : so_cones)
    {
        rhs2.segment(num_var + num_eq + num_pc + rhs_index, sc.dim) = s.segment(rhs_index, sc.dim) - rz.segment(rz_index, sc.dim);
        rhs_index += sc.dim;
        rz_index += sc.dim;
        rhs2.segment(num_var + num_eq + num_pc + rhs_index, 2).setZero();
        rhs_index += 2;
    }
}

void ECOSEigen::updateKKT()
{
    // TODO: Faster element access.

    /* LP cone */
    for (size_t i = 0; i < lp_cone.dim; i++)
    {
        K.coeffRef(i, i) = -lp_cone.v(i) - settings.delta;
    }

    /* Second-order cone */
    size_t diag_index = lp_cone.dim;
    for (const SecondOrderCone &sc : so_cones)
    {
        /* D */
        K.coeffRef(diag_index, diag_index) = -sc.eta_square * sc.d1 - settings.delta;
        for (size_t k = 1; k < sc.dim; k++)
        {
            diag_index++;
            K.coeffRef(diag_index, diag_index) = -sc.eta_square - settings.delta;
        }

        /* v */
        diag_index++;
        for (size_t k = 0; k < sc.dim - 1; k++)
        {
            K.coeffRef(diag_index - sc.dim + k, diag_index) = -sc.eta_square * sc.v1 * sc.q(k);
        }

        /* u */
        diag_index++;
        K.coeffRef(diag_index - sc.dim - 1, diag_index) = -sc.eta_square * sc.u0;
        for (size_t k = 1; k < sc.dim; k++)
        {
            K.coeffRef(diag_index - sc.dim - 1 + k, diag_index) = -sc.eta_square * sc.u1 * sc.q(k);
        }
        K.coeffRef(diag_index, diag_index) = sc.eta_square + settings.delta;
    }

    ldlt.factorize(K);
}

void ECOSEigen::setupKKT(const Eigen::SparseMatrix<double> &G,
                         const Eigen::SparseMatrix<double> &A)
{
    /**
     *      [ 0  A' G' ]
     *  K = [ A  0  0  ]
     *      [ G  0  -V ]
     * 
     *   V = blkdiag(I, blkdiag(I, 1, -1), ...,  blkdiag(I, 1, -1));
     *                    ^   number of second-order cones   ^
     *               ^ dimension of positive contraints
     * 
     *  Only the upper triangular part is constructed here.
     **/

    const size_t K_dim = num_var + num_eq + num_ineq + 2 * num_sc;
    //                                                   ^ expanded scalings
    K.resize(K_dim, K_dim);

    At = A.transpose();
    Gt = G.transpose();

    size_t K_nonzeros = At.nonZeros() + Gt.nonZeros();
    // Static Regularization
    K_nonzeros += num_var + num_eq;
    // Positive part of scaling block V
    K_nonzeros += num_pc;
    for (const SecondOrderCone &sc : so_cones)
    {
        // SC part of scaling block V
        K_nonzeros += 3 * sc.dim + 1;
    }
    K.reserve(K_nonzeros);

    std::vector<Eigen::Triplet<double>> K_triplets;
    K_triplets.reserve(K_nonzeros);

    // Static Regularization of blocks (1,1) and (2,2)
    for (size_t k = 0; k < num_var; k++)
    {
        K_triplets.emplace_back(k, k, settings.delta);
    }
    for (size_t k = 0; k < num_eq; k++)
    {
        K_triplets.emplace_back(num_var + k, num_var + k, -settings.delta);
    }

    // A'
    for (int k = 0; k < At.outerSize(); k++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(At, k); it; ++it)
        {
            K_triplets.emplace_back(it.row(), it.col() + A.cols(), it.value());
        }
    }

    // G'
    {
        // Linear block
        Eigen::SparseMatrix<double> Gt_block;
        size_t col_K = A.cols() + At.cols();

        Gt_block = Gt.leftCols(num_pc);
        for (int k = 0; k < Gt_block.outerSize(); k++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(At, k); it; ++it)
            {
                K_triplets.emplace_back(it.row(), col_K + it.col(), it.value());
            }
        }
        col_K += num_pc;

        // SOC blocks
        size_t col_Gt = col_K;
        for (const SecondOrderCone &sc : so_cones)
        {
            Gt_block = Gt.middleCols(col_Gt, sc.dim);
            for (int k = 0; k < Gt_block.outerSize(); k++)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Gt_block, k); it; ++it)
                {
                    K_triplets.emplace_back(it.row(), col_K + it.col(), it.value());
                }
            }
            col_K += sc.dim + 2;
            col_Gt += sc.dim;
        }
    }

    // -V
    {
        size_t row_col = A.cols() + At.cols();

        // First identity block
        for (size_t k = 0; k < num_pc; k++)
        {
            K_triplets.emplace_back(row_col, row_col, -1.);
            row_col++;
        }

        // SOC blocks
        /**
         * The scaling matrix has the following structure:
         *
         *    [ 1                * ]
         *    [   1           *  * ]
         *    [     .         *  * ]      
         *    [       .       *  * ]       [ D   v  u  ]      D: Identity of size conesize       
         *  - [         .     *  * ]  =  - [ u'  1  0  ]      v: Vector of size conesize - 1      
         *    [           1   *  * ]       [ v'  0' -1 ]      u: Vector of size conesize    
         *    [             1 *  * ]
         *    [   * * * * * * 1    ]
         *    [ * * * * * * *   -1 ]
         *
         *  Only the upper triangular part is constructed here.
         **/
        for (const SecondOrderCone &sc : so_cones)
        {
            for (size_t k = 0; k < sc.dim; k++)
            {
                K_triplets.emplace_back(row_col, row_col, -1.);
                row_col++;
            }

            row_col++;
            K_triplets.emplace_back(row_col, row_col, -1.);

            // -v
            for (size_t k = 1; k < sc.dim; k++)
            {
                K_triplets.emplace_back(row_col - sc.dim + k, row_col, -1.);
            }

            row_col++;
            K_triplets.emplace_back(row_col, row_col, 1.);

            // -u
            for (size_t k = 0; k < sc.dim; k++)
            {
                K_triplets.emplace_back(row_col - sc.dim - 1 + k, row_col, -1.);
            }
        }
    }

    K.setFromTriplets(K_triplets.begin(), K_triplets.end());
    assert(size_t(K.nonZeros()) == K_nonzeros);
}