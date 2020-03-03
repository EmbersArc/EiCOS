#include "ecos.hpp"
#include "data.hpp"
#include "timing.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>

int main()
{
    double t0 = tic();

    ecos_eigen::ECOSEigen solver(n, m, p, l, ncones, q, Gpr, Gjc, Gir, Apr, Ajc, Air, c, h, b);

    fmt::print("Time for setup:    {:.3}ms\n", toc(t0));
    t0 = tic();

    ecos_eigen::exitcode exitcode = solver.solve();

    fmt::print("Time for solve:    {:.3}ms\n", toc(t0));

    assert("Solution not optimal!" && exitcode == ecos_eigen::exitcode::OPTIMAL);
}