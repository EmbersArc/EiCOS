A C++ Second Order Cone Solver based on [ECOS](https://github.com/embotech/ecos).

<!--
\begin{aligned} 
\text{minimize} \ c^T x \\
\text{subject to} \ Ax &= b \\
Gx &\preceq_K h
\end{aligend}
-->

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Ctext%7Bminimize%7D%20%5C%20c%5ET%20x%20%5C%5C%20%5Ctext%7Bsubject%20to%7D%20%5C%20Ax%20%26%3D%20b%20%5C%5C%20Gx%20%26%5Cpreceq_K%20h%20%5Cend%7Baligend%7D)

### Dependencies
* `Eigen` for linear algebra functionality
* `fmt` for printing and formatting

Work in progress. All tests pass at this point but features are still being implemented.
