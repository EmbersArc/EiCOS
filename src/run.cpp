#include "eicos.hpp"
#include "data_MPC01.hpp"
#include "timing.hpp"

#include "printing.hpp"

int main()
{
    double t0 = tic();

    cuDSS::SparseMatrix<double> G_;
    cuDSS::SparseMatrix<double> A_;
    cuDSS::Vector c_;
    cuDSS::Vector h_;
    cuDSS::Vector b_;
    cuDSS::VectorInt q_;

    if (Gpr and Gjc and Gir)
    {
        G_ = cuDSS::Map<cuDSS::SparseMatrix<double>>(m, n, Gjc[n], Gjc, Gir, Gpr);
        q_ = cuDSS::Map<cuDSS::VectorInt>(q, ncones);
        h_ = cuDSS::Map<cuDSS::Vector>(h, m);
    }
    if (Apr and Ajc and Air)
    {
        A_ = cuDSS::Map<cuDSS::SparseMatrix<double>>(p, n, Ajc[n], Ajc, Air, Apr);
        b_ = cuDSS::Map<cuDSS::Vector>(b, p);
    }
    if (c)
    {
        c_ = cuDSS::Map<cuDSS::Vector>(c, n);
    }

    EiCOS::Solver solver(G_, A_, c_, h_, b_, q_);
    EiCOS::exitcode exitcode;

    print("Time for setup:    {:.3}ms\n", toc(t0));

    t0 = tic();
    exitcode = solver.solve();
    print("Time for solve:    {:.3}ms\n", toc(t0));

    // // test data update
    t0 = tic();
    solver.updateData(G_, A_, c_, h_, b_);
    print("Time for update:    {:.3}ms\n", toc(t0));

    t0 = tic();
    exitcode = solver.solve();
    print("Time for solve:    {:.3}ms\n", toc(t0));

    assert("Solution not optimal!" && exitcode == EiCOS::exitcode::optimal);
}
