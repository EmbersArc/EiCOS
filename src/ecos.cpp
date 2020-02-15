#include "ecos.hpp"

ECOSEigen::ECOSEigen(Eigen::SparseMatrix<double> &G,
                     Eigen::SparseMatrix<double> &A,
                     Eigen::SparseMatrix<double> &c,
                     Eigen::SparseMatrix<double> &h,
                     Eigen::SparseMatrix<double> &b,
                     std::vector<size_t> &soc_dims)
{
    num_eq = A.rows();
    num_so = soc_dims.size();
    num_po = h.rows() - num_so;
}

void ECOSEigen::SetupKKT(Eigen::SparseMatrix<double> &G,
                         Eigen::SparseMatrix<double> &A,
                         Eigen::SparseMatrix<double> &c,
                         std::vector<size_t> &soc_dims)
{
    /**
     *      [ 0  A' G' ]
     *  K = [ A  0  0  ]
     *      [ G  0  -V ]
     * 
     *   V = blkdiag(I, blkdiag(I,1,-1), ...,  blkdiag(I,1,-1));
     *                    ^ #number of second-order cones ^
     *               ^ size is number of positive contraints
     **/

    const size_t K_dim = A.rows() + A.cols() + G.rows() + 2 * soc_dims.size();
    K.resize(K_dim, K_dim);

    Eigen::SparseMatrix<double> At = A.transpose();
    Eigen::SparseMatrix<double> Gt = G.transpose();

    // add A
    for (int row = 0; row < A.rows(); row++)
    {
        for (int col = 0; col < A.cols(); col++)
        {
            K.insert(At.cols() + row, col) = A.coeff(row, col);
        }
    }

    // add A'
    for (int row = 0; row < At.rows(); row++)
    {
        for (int col = 0; col < At.cols(); col++)
        {
            K.insert(row, A.cols() + col) = At.coeff(row, col);
        }
    }
    
    // add G
    for (int row = 0; row < G.rows(); row++)
    {
        for (int col = 0; col < G.cols(); col++)
        {
            if (G.)
            K.insert(At.rows() + A.rows() + row, col) = G.coeff(row, col);
        }
    }

    // add G'
    for (int row = 0; row < Gt.rows(); row++)
    {
        for (int col = 0; col < Gt.cols(); col++)
        {
            K.insert(row, A.cols() + At.cols() + col) = Gt.coeff(row, col);
        }
    }



    // add linear parts to V

    // add SOC parts to V
    for (size_t c = 0; c < soc_dims.size(); c++)
    {
    }

    K.makeCompressed();
}