A C++ Second Order Cone Solver based on [ECOS](https://github.com/embotech/ecos).

<!--
\begin{aligned} 
\text{minimize} \ c^T x \\
\text{subject to} \ Ax &= b \\
Gx &\preceq_K h
\end{aligend}
-->
![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Ctext%7Bminimize%7D%20%5C%20c%5ET%20x%20%5C%5C%20%5Ctext%7Bsubject%20to%7D%20%5C%20Ax%20%26%3D%20b%20%5C%5C%20Gx%20%26%5Cpreceq_K%20h%20%5Cend%7Baligend%7D)

The last constraint is generalized and includes both the positive orthant and second order cones, so that the top rows of G represent the positive constraints and the bottom rows contain stacked representations of the second order cones:
<!--
Q_n = \{ \begin{bmatrix}t\\x\end{bmatrix} \mid  t \geq \lVert x \rVert_2 \} 
-->
![equation](https://latex.codecogs.com/gif.latex?Q_n%20%3D%20%5C%7B%20%5Cbegin%7Bbmatrix%7Dt%5C%5Cx%5Cend%7Bbmatrix%7D%20%5Cmid%20t%20%5Cgeq%20%5ClVert%20x%20%5CrVert_2%20%5C%7D)

### Usage
To construct a solver instance, call `EiCOS::Solver solver(G, A, c, h, b, q)` where `A` and `B` are of type `Eigen::SparseMatrix<double>`, `c`, `h` and `b` of type `Eigen::VectorXd` and the vector of second order cone dimensions `q` of type `Eigen::VectorXi`.

After successfully calling `solver.solve()`, the solution can be accessed by calling `solver.solution()`.

To update the problem parameters, call `solver.updateData(G, A, c, h, b)`. Using this method instead of constructing a new problem can save a lot of time, especially for larger problems. The only restriction is that the sparsity pattern and dimensions must be the same as in the original problem.

### Dependencies
* `Eigen` for linear algebra functionality
* `fmt` for printing and formatting

Work in progress. All tests pass at this point but features are still being implemented.
