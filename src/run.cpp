#include "ecos.hpp"
#include "data.hpp"
#include "timing.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>

int main()
{
    double t0 = tic();

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

    ecos_eigen::ECOSEigen solver(G_, A_, c_, h_, b_, q_);
    ecos_eigen::exitcode exitcode;

    fmt::print("Time for setup:    {:.3}ms\n", toc(t0));

    t0 = tic();
    exitcode = solver.solve();
    fmt::print("Time for solve:    {:.3}ms\n", toc(t0));

    // test data update
    t0 = tic();
    solver.updateData(G_, A_, c_, h_, b_);
    fmt::print("Time for update:    {:.3}ms\n", toc(t0));

    t0 = tic();
    exitcode = solver.solve();
    fmt::print("Time for solve:    {:.3}ms\n", toc(t0));

    assert("Solution not optimal!" && exitcode == ecos_eigen::exitcode::OPTIMAL);
}