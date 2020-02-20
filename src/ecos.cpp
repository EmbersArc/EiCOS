#include "ecos.hpp"

#include <Eigen/SparseCholesky>

ECOSEigen::ECOSEigen(const Eigen::SparseMatrix<double> &G,
                     const Eigen::SparseMatrix<double> &A,
                     const Eigen::SparseVector<double> &c,
                     const Eigen::SparseVector<double> &h,
                     const Eigen::SparseVector<double> &b,
                     const std::vector<size_t> &soc_dims)
    : G(G), A(A), c(c), h(h), b(b), soc_dims(soc_dims)
{
    num_var = A.cols();
    num_eq = A.rows();
    num_sc = soc_dims.size();
    num_ineq = G.rows(); // = num_pc + num_sc
    num_pc = num_ineq - num_sc;

    SetupKKT(G, A, soc_dims);
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

    // TODO: temporary definitions
    Eigen::SparseVector<double> rx, ry, rz;
    Eigen::SparseVector<double> x, y, z;
    Eigen::SparseVector<double> s;
    double tau;
    double hresx, hresy, hresz;
    double nx, ny, nz, ns;

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
    // TODO: temporary definitions
    double gap, mu, kap, tau, D, kapovert, pcost, dcost, cx, hz, by, relgap;
    double nrx, nry, nrz, p, resy0, nx, ny, nz, ns;
    double pres, dres, reltol;
    double hresx, hresy, hresz;
    std::optional<double> pinfres, dinfres;
    Eigen::VectorXd s, z, rx, ry, rz;

    gap = s.dot(z);
    mu = (gap + kap * tau) / (D + 1.);
    kapovert = kap / tau;
    pcost = cx / tau;
    dcost = -(hz + by) / tau;

    /* Relative Duality Gap */
    if (pcost < 0)
    {
        relgap = gap / -pcost;
    }
    else if (dcost > 0)
    {
        relgap = gap / dcost;
    }
    else
    {
        // fail
    }

    /* Residuals */
    nry = p > 0 ? ry.norm() / std::max(resy0 + nx, 1.) : 0.0;
    nrz = rz.norm() / std::max(resz0 + nx + ns, 1.);
    pres = std::max(nry, nrz) / tau;
    dres = rx.norm() / std::max(resx0 + ny + nz, 1.) / tau;

    /* Infeasibility Measures */
    if ((hz + by) / std::max(ny + nz, 1.) < -reltol)
    {
        pinfres = hresx / std::max(ny + nz, 1.);
    }
    if (cx / std::max(nx, 1.) < -reltol)
    {
        dinfres = std::max(hresy / std::max(nx, 1.), hresz / std::max(nx + ns, 1.));
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
        if (x[i] <= 0 && -x[i] > alpha)
        {
            alpha = -x[i];
        }
    }

    /* Second-Order Cone */
    double cres;
    for (size_t cone_dim : soc_dims)
    {
        cres = x[i];
        i++;
        cres -= x.segment(i, cone_dim - 1).norm();
        i += cone_dim - 1;

        if (cres <= 0 && -cres > alpha)
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
    for (size_t cone_dim : soc_dims)
    {
        x[i] += alpha;
        i += cone_dim;
    }
}

void ECOSEigen::Solve()
{
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
    for (size_t cone_dim : soc_dims)
    {
        rhs1.segment(rhs1_index, cone_dim) = h.segment(h_index, cone_dim);
        h_index += cone_dim;
        rhs1_index += cone_dim + 2;
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
    using LDLT_t = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper>;
    LDLT_t ldlt;
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

    for (iteration = 0; iteration < max_iterations; iteration++)
    {
        computeResiduals();
        updateStatistics();
    }


}

void ECOSEigen::SetupKKT(const Eigen::SparseMatrix<double> &G,
                         const Eigen::SparseMatrix<double> &A,
                         const std::vector<size_t> &soc_dims)
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
    for (size_t cone_dim : soc_dims)
    {
        // SC part of scaling block V
        K_nonzeros += 3 * cone_dim + 1;
    }
    K.reserve(K_nonzeros);

    std::vector<Eigen::Triplet<double>> K_triplets;
    K_triplets.reserve(K_nonzeros);

    // Static Regularization of blocks (1,1) and (2,2)
    for (size_t k = 0; k < num_var; k++)
    {
        K_triplets.emplace_back(k, k, delta_reg);
    }
    for (size_t k = 0; k < num_eq; k++)
    {
        K_triplets.emplace_back(num_var + k, num_var + k, -delta_reg);
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
        for (size_t cone_dim : soc_dims)
        {
            Gt_block = Gt.middleCols(col_Gt, cone_dim);
            for (int k = 0; k < Gt_block.outerSize(); k++)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Gt_block, k); it; ++it)
                {
                    K_triplets.emplace_back(it.row(), col_K + it.col(), it.value());
                }
            }
            col_K += cone_dim + 2;
            col_Gt += cone_dim;
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
        for (size_t cone_dim : soc_dims)
        {
            for (size_t k = 0; k < cone_dim; k++)
            {
                K_triplets.emplace_back(row_col, row_col, -1.);
                row_col++;
            }

            row_col++;
            K_triplets.emplace_back(row_col, row_col, -1.);

            // -v
            for (size_t k = 1; k < cone_dim; k++)
            {
                K_triplets.emplace_back(row_col - cone_dim + k, row_col, -1.);
            }

            row_col++;
            K_triplets.emplace_back(row_col, row_col, 1.);

            // -u
            for (size_t k = 0; k < cone_dim; k++)
            {
                K_triplets.emplace_back(row_col - cone_dim - 1 + k, row_col, -1.);
            }
        }
    }

    K.setFromTriplets(K_triplets.begin(), K_triplets.end());
    assert(size_t(K.nonZeros()) == K_nonzeros);
}