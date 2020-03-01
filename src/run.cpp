#include "ecos.hpp"
#include "data.hpp"
#include "timing.hpp"

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

    double t0 = tic();

    ECOSEigen solver(G, A, c, h, b, soc_dims);

    fmt::print("Time for setup:    {}ms", toc(t0));
    t0 = tic();

    solver.solve();

    fmt::print("Time for solve:    {}ms", toc(t0));

    fmt::print("\n\n");
    // fmt::print("x=\n{}\n", solver.x);
}