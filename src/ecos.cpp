#include "ecos.hpp"

#include <Eigen/SparseCholesky>

ECOSEigen::ECOSEigen(const Eigen::SparseMatrix<double> &G,
                     const Eigen::SparseMatrix<double> &A,
                     const Eigen::SparseMatrix<double> &c,
                     const Eigen::SparseMatrix<double> &h,
                     const Eigen::SparseMatrix<double> &b,
                     const std::vector<size_t> &soc_dims)
{
    num_var = A.cols();
    num_eq = A.rows();
    num_sc = soc_dims.size();
    num_ineq = G.rows(); // = num_pc + num_sc
    num_pc = num_ineq - num_sc;

    SetupKKT(G, A, soc_dims);
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

    Eigen::SparseMatrix<double> At = A.transpose();
    Eigen::SparseMatrix<double> Gt = G.transpose();

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

        // First identity block of -V
        for (size_t k = 0; k < num_pc; k++)
        {
            K_triplets.emplace_back(row_col, row_col, -1.);
            row_col++;
        }

        // SOC parts of -V
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

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> ldlt;
    ldlt.compute(K);
}