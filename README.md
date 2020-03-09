A C++ Second Order Cone Solver based on [ECOS](https://github.com/embotech/ecos).

<!--
\begin{aligned} 
\text{minimize} \ c^T x \\
\text{subject to} \ Ax &= b \\
Gx &\preceq_K h
\end{aligend}
-->
![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Ctext%7Bminimize%7D%20%5C%20c%5ET%20x%20%5C%5C%20%5Ctext%7Bsubject%20to%7D%20%5C%20Ax%20%26%3D%20b%20%5C%5C%20Gx%20%26%5Cpreceq_K%20h%20%5Cend%7Baligend%7D)

The last constraint is generalized and includes both the positive orthant and second order cones, so that the top `p` rows of G represent the positive constraints and the remaining rows contain stacked representations of the second order cones:
<!--
Q_n = \{ \begin{bmatrix}t\\x\end{bmatrix} \mid  t \geq \lVert x \rVert_2 \} 
-->
![equation](https://latex.codecogs.com/gif.latex?Q_n%20%3D%20%5C%7B%20%5Cbegin%7Bbmatrix%7Dt%5C%5Cx%5Cend%7Bbmatrix%7D%20%5Cmid%20t%20%5Cgeq%20%5ClVert%20x%20%5CrVert_2%20%5C%7D)


Name | Explanation
--- | ---
n | Number of problem variables
m | Number of inequality constraints.
p | Number of equality constraints.
l |       The dimension of the positive orthant.
ncones |  Number of second order cones in K.
q |       Vector of dimesions of each cone constraint in K.
A(p,n) |  Equality constraint matrix.
b(p) |    Equality constraint vector.
G(m,n) |  Generalized inequality matrix.
h(m) |    Generalized inequality vector.
c(n) |    Variable weights.

### Usage
```cpp
#include "eicos.hpp"

Eigen::SparseMatrix<double> A, B;
Eigen::VectorXd c, h, b;
Eigen::VectorXi q;

// (Set up problem data)

// Construct a solver instance
EiCOS::Solver solver(G, A, c, h, b, q);

// Solve the problem
solver.solve()

// Save the solution
Eigen::VectorXd s = solver.solution();

// (Change entries in G, A, c, h, b)

// Update problem data: Using this method instead of constructing a new problem can
// save a lot of time, especially for larger problems. The only restriction is that 
// the sparsity pattern and dimensions must be the same as in the original problem.
solver.updateData(G, A, c, h, b);

// Rinse and repeat
solver.solve()

```

### Dependencies
* `Eigen` for linear algebra functionality
* `fmt` for printing and formatting

Work in progress. All tests pass at this point but features are still being implemented.
