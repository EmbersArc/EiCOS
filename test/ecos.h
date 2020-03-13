#pragma once

// Ugly hack to make tests work with minimal effort

#include "eicos.hpp"

using idxint = int;
using pfloat = double;
using pwork = EiCOS::Solver;

pwork *ECOS_setup(idxint n, idxint m, idxint p, idxint l, idxint ncones, idxint *q, idxint /*nexc*/,
                  pfloat *Gpr, idxint *Gjc, idxint *Gir,
                  pfloat *Apr, idxint *Ajc, idxint *Air,
                  pfloat *c, pfloat *h, pfloat *b)
{
    return new EiCOS::Solver(n, m, p, l, ncones, q, Gpr, Gjc, Gir, Apr, Ajc, Air, c, h, b);
}

idxint ECOS_solve(pwork *work)
{
    return idxint(work->solve());
}

void ECOS_updateData(pwork *work,
                     pfloat *Gpr, pfloat *Apr,
                     pfloat *c, pfloat *h, pfloat *b)
{
    work->updateData(Gpr, Apr, c, h, b);
}

void ECOS_cleanup(pwork *work, idxint /*whatever*/)
{
    delete work;
}

#define ECOS_OPTIMAL (0)       /* Problem solved to optimality              */
#define ECOS_PINF (1)          /* Found certificate of primal infeasibility */
#define ECOS_DINF (2)          /* Found certificate of dual infeasibility   */
#define ECOS_INACC_OFFSET (10) /* Offset exitflag at inaccurate results */
#define ECOS_MAXIT (-1)        /* Maximum number of iterations reached      */
#define ECOS_NUMERICS (-2)     /* Search direction unreliable               */
#define ECOS_OUTCONE (-3)      /* s or z got outside the cone, numerics?    */
#define ECOS_SIGINT (-4)       /* solver interrupted by a signal/ctrl-c     */
#define ECOS_FATAL (-7)        /* Unknown problem in solver                 */