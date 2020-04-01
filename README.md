# Eigen Conic Solver

A C++ Second Order Cone Solver for problems of the form

<!--
\begin{aligned} 
\text{minimize} \ \ &c^T x \\
\text{subject to} \ \ &Ax = b \\
&Gx \preceq_K h
\end{aligend}
-->
![equation](https://latex.codecogs.com/svg.latex?\begin{aligned}&space;\text{minimize}&space;\&space;\&space;&c^T&space;x&space;\\&space;\text{subject&space;to}&space;\&space;\&space;&Ax&space;=&space;b&space;\\&space;&Gx&space;\preceq_K&space;h&space;\end{aligend})

<!-- 
\begin{align*}
x & \dots\text{Variable vector} \in \mathbb{R}^{n_{var}} \\
G & \dots\text{Inequality constraint matrix} \in \mathbb{R}^{n_{ineq} \times n_{var}} \\
A & \dots\text{Equality constraint matrix} \in \mathbb{R}^{n_{eq} \times n_{var}} \\
c & \dots\text{Variable weight vector} \in \mathbb{R}^{n_{var}} \\
h & \dots\text{Inequality constraint vector} \in \mathbb{R}^{n_{ineq}} \\
b & \dots\text{Equality constraint vector} \in \mathbb{R}^{n_{eq}} \\
q & \dots\text{Vector containing dimension of each cone constraint} \in \mathbb{N}^{n_{cones}} \\\\
n_{var} & \dots\text{Number of variables} \\
n_{eq} & \dots\text{Number of equality constraints} \\
n_{ineq} & \dots\text{Number of inequality constraints} \\
n_{pc} & \dots\text{Number of positive constraints (dimension of positive orthant)} \\
n_{cones} & \dots\text{Number of second order cones in K} \\
\end{align*}
-->
![symbols](https://latex.codecogs.com/svg.latex?\begin{align*}&space;x&space;&&space;\dots\text{Variable&space;vector}&space;\in&space;\mathbb{R}^{n_{var}}&space;\\&space;G&space;&&space;\dots\text{Inequality&space;constraint&space;matrix}&space;\in&space;\mathbb{R}^{n_{ineq}&space;\times&space;n_{var}}&space;\\&space;A&space;&&space;\dots\text{Equality&space;constraint&space;matrix}&space;\in&space;\mathbb{R}^{n_{eq}&space;\times&space;n_{var}}&space;\\&space;c&space;&&space;\dots\text{Variable&space;weight&space;vector}&space;\in&space;\mathbb{R}^{n_{var}}&space;\\&space;h&space;&&space;\dots\text{Inequality&space;constraint&space;vector}&space;\in&space;\mathbb{R}^{n_{ineq}}&space;\\&space;b&space;&&space;\dots\text{Equality&space;constraint&space;vector}&space;\in&space;\mathbb{R}^{n_{eq}}&space;\\&space;q&space;&&space;\dots\text{Vector&space;containing&space;dimension&space;of&space;each&space;cone&space;constraint}&space;\in&space;\mathbb{N}^{n_{cones}}&space;\\\\&space;n_{var}&space;&&space;\dots\text{Number&space;of&space;variables}&space;\\&space;n_{eq}&space;&&space;\dots\text{Number&space;of&space;equality&space;constraints}&space;\\&space;n_{ineq}&space;&&space;\dots\text{Number&space;of&space;inequality&space;constraints}&space;\\&space;n_{pc}&space;&&space;\dots\text{Number&space;of&space;positive&space;constraints&space;(dimension&space;of&space;positive&space;orthant)}&space;\\&space;n_{cones}&space;&&space;\dots\text{Number&space;of&space;second&space;order&space;cones&space;in&space;K}&space;\\&space;\end{align*})

The last constraint is generalized and includes both the positive orthant and second order cones, so that the top rows of `G` represent the linear constraints
<!--
\begin{gathered}
Cx \leq d \\
\Leftrightarrow \\
C \preceq d \\
\text{with} \\
C \in \mathbb{R}^{n_{eq} \times n_{var}} \\
d \in \mathbb{R}^{n_{eq}} \\
\end{gathered}
-->
![equation](https://latex.codecogs.com/svg.latex?\begin{gathered}&space;Cx&space;\leq&space;d&space;\\&space;\Leftrightarrow&space;\\&space;C&space;\preceq&space;d&space;\\&space;\text{with}&space;\\&space;C&space;\in&space;\mathbb{R}^{n_{eq}&space;\times&space;n_{var}}&space;\\&space;d&space;\in&space;\mathbb{R}^{n_{eq}}&space;\\&space;\end{gathered})

and the remaining rows contain stacked representations of the second order cones:
<!--
\begin{gathered}
\lVert F_ix + g_i \rVert \leq v_i^T x + w_i \\
\Leftrightarrow \\
-\begin{bmatrix} v_i^T \\ F_i \end{bmatrix} \preceq \begin{bmatrix} w_i \\ g_i \end{bmatrix} \\
i = 1,...,n_{cones}
\\
\text{with} \\
v_i \in \mathbb{R}^{n_{var}} \\
F_i \in \mathbb{R}^{q_i-1 \times n_{var}} \\
w_i \in \mathbb{R} \\
g_i \in \mathbb{R}^{q_i-1} \\
\end{gathered}
-->
![equation](https://latex.codecogs.com/svg.latex?\begin{gathered}&space;\lVert&space;F_ix&space;&plus;&space;g_i&space;\rVert&space;\leq&space;v_i^T&space;x&space;&plus;&space;w_i&space;\\&space;\Leftrightarrow&space;\\&space;-\begin{bmatrix}&space;v_i^T&space;\\&space;F_i&space;\end{bmatrix}&space;\preceq&space;\begin{bmatrix}&space;w_i&space;\\&space;g_i&space;\end{bmatrix}&space;\\&space;i&space;=&space;1,...,n_{cones}&space;\\&space;\text{with}&space;\\&space;v_i&space;\in&space;\mathbb{R}^{n_{var}}&space;\\&space;F_i&space;\in&space;\mathbb{R}^{q_i-1&space;\times&space;n_{var}}&space;\\&space;w_i&space;\in&space;\mathbb{R}&space;\\&space;g_i&space;\in&space;\mathbb{R}^{q_i-1}&space;\\&space;\end{gathered})

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
