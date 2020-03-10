# Eigen Conic Solver

A C++ Second Order Cone Solver for problems of the form

<!--
\begin{aligned} 
\text{minimize} \ \ &c^T x \\
\text{subject to} \ \ &Ax = b \\
&Gx \preceq_K h
\end{aligend}
-->
![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cbegin%7Baligned%7D%20%5Ctext%7Bminimize%7D%20%5C%20%5C%20%26c%5ET%20x%20%5C%5C%20%5Ctext%7Bsubject%20to%7D%20%5C%20%5C%20%26Ax%20%3D%20b%20%5C%5C%20%26Gx%20%5Cpreceq_K%20h%20%5Cend%7Baligend%7D)

<!-- 
\begin{align*}
G & \dots\text{Inequality constraint matrix} \in \mathbb{R}^{n_{ineq} \times n_{var}} \\
A & \dots\text{Equality constraint matrix} \in \mathbb{R}^{n_{eq} \times n_{var}} \\
c & \dots\text{Variable weight vector} \in \mathbb{R}^{n_{var}} \\
h & \dots\text{Inequality constraint vector} \in \mathbb{R}^{n_{ineq}} \\
b & \dots\text{Equality constraint vector} \in \mathbb{R}^{n_{eq}} \\
q & \dots\text{Vector containing dimension of each cone constraint in K} \\
\\
n_{var} & \dots\text{Number of variables} \\
n_{eq} & \dots\text{Number of equality constraints} \\
n_{ineq} & \dots\text{Number of inequality constraints} \\
n_{pc} & \dots\text{Number of positive constraints (dimension of positive orthant)} \\
n_{cones} & \dots\text{Number of second order cones in K} \\
\end{align*}
-->
![symbols](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cbegin%7Balign*%7D%20G%20%26%20%5Cdots%5Ctext%7BInequality%20constraint%20matrix%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn_%7Bineq%7D%20%5Ctimes%20n_%7Bvar%7D%7D%20%5C%5C%20A%20%26%20%5Cdots%5Ctext%7BEquality%20constraint%20matrix%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn_%7Beq%7D%20%5Ctimes%20n_%7Bvar%7D%7D%20%5C%5C%20c%20%26%20%5Cdots%5Ctext%7BVariable%20weight%20vector%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn_%7Bvar%7D%7D%20%5C%5C%20h%20%26%20%5Cdots%5Ctext%7BInequality%20constraint%20vector%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn_%7Bineq%7D%7D%20%5C%5C%20b%20%26%20%5Cdots%5Ctext%7BEquality%20constraint%20vector%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn_%7Beq%7D%7D%20%5C%5C%20q%20%26%20%5Cdots%5Ctext%7BVector%20containing%20dimension%20of%20each%20cone%20constraint%20in%20K%7D%20%5C%5C%20%5C%5C%20n_%7Bvar%7D%20%26%20%5Cdots%5Ctext%7BNumber%20of%20variables%7D%20%5C%5C%20n_%7Beq%7D%20%26%20%5Cdots%5Ctext%7BNumber%20of%20equality%20constraints%7D%20%5C%5C%20n_%7Bineq%7D%20%26%20%5Cdots%5Ctext%7BNumber%20of%20inequality%20constraints%7D%20%5C%5C%20n_%7Bpc%7D%20%26%20%5Cdots%5Ctext%7BNumber%20of%20positive%20constraints%20%28dimension%20of%20positive%20orthant%29%7D%20%5C%5C%20n_%7Bcones%7D%20%26%20%5Cdots%5Ctext%7BNumber%20of%20second%20order%20cones%20in%20K%7D%20%5C%5C%20%5Cend%7Balign*%7D)

The last constraint is generalized and includes both the positive orthant and second order cones, so that the top rows of G each represent a positive constraint and the remaining rows contain stacked representations of the second order cones:
<!--
\begin{gathered}
\lVert F_ix + g_i \rVert \leq v_i^T x + w_i \\
\Leftrightarrow \\
\begin{bmatrix} v_i^T \\ -F_i \end{bmatrix} \preceq \begin{bmatrix} w_i \\ g_i \end{bmatrix} \\
i = 1,...,n_{cones}
\\
\text{with} \\
v_i \in \mathbb{R}^{n_{var}} \\
F_i \in \mathbb{R}^{q_i-1 \times n_{var}} \\
w_i \in \mathbb{R} \\
g_i \in \mathbb{R}^{q_i-1} \\
\end{gathered}
-->
![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cbegin%7Bgathered%7D%20%5ClVert%20F_ix%20&plus;%20g_i%20%5CrVert%20%5Cleq%20v_i%5ET%20x%20&plus;%20w_i%20%5C%5C%20%5CLeftrightarrow%20%5C%5C%20%5Cbegin%7Bbmatrix%7D%20v_i%5ET%20%5C%5C%20-F_i%20%5Cend%7Bbmatrix%7D%20%5Cpreceq%20%5Cbegin%7Bbmatrix%7D%20w_i%20%5C%5C%20g_i%20%5Cend%7Bbmatrix%7D%20%5C%5C%20i%20%3D%201%2C...%2Cn_%7Bcones%7D%20%5C%5C%20%5Ctext%7Bwith%7D%20%5C%5C%20v_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn_%7Bvar%7D%7D%20%5C%5C%20F_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bq_i-1%20%5Ctimes%20n_%7Bvar%7D%7D%20%5C%5C%20w_i%20%5Cin%20%5Cmathbb%7BR%7D%20%5C%5C%20g_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bq_i-1%7D%20%5C%5C%20%5Cend%7Bgathered%7D)

### Usage
```cpp
#include "eicos.hpp"

Eigen::SparseMatrix<double> G, A;
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
* `fmt` (optional) for printing and formatting

### Credits
This solver is entirely based on [ECOS](https://github.com/embotech/ecos).

* Alexander Domahidi (ECOS principal developer)
* Eric Chu (ECOS unit tests)
* Stephen Boyd (methods and maths)
