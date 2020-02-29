#include "ecos.hpp"
#include "data.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>

int main()
{
    using map_sparse_t = Eigen::Map<Eigen::SparseMatrix<double>>;
    map_sparse_t A(p, n, Ajc[n], Ajc, Air, Apr);
    map_sparse_t G(m, n, Gjc[n], Gjc, Gir, Gpr);

    Eigen::Map<Eigen::VectorXd> b(b_, p);
    Eigen::Map<Eigen::VectorXd> h(h_, m);
    Eigen::Map<Eigen::VectorXd> c(c_, n);
    Eigen::Map<Eigen::VectorXi> soc_dims(q_, ncones);

    ECOSEigen solver(G, A, c, h, b, soc_dims);

    solver.solve();

    fmt::print("\nDone.\n");
    fmt::print("{}\n", solver.x);
}