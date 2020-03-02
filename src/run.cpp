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
    Eigen::Map<Eigen::VectorXd> x(x_, n);

    double t0 = tic();

    ecos_eigen::ECOSEigen solver(G, A, c, h, b, soc_dims);

    fmt::print("Time for setup:    {:.3}ms\n", toc(t0));
    t0 = tic();

    solver.solve();

    fmt::print("Time for solve:    {:.3}ms\n", toc(t0));

    bool correct = (x - solver.solution()).lpNorm<Eigen::Infinity>() < 1e-7;

    if (correct)
    {
        fmt::print("Solution accurate. (error_norm = {:.4e})\n", (x - solver.solution()).lpNorm<Eigen::Infinity>());
    }
    else
    {
        fmt::print("Solution inaccurate. (error_norm = {:.4e})\n", (x - solver.solution()).lpNorm<Eigen::Infinity>());
    }

    fmt::print("\n\n");
}