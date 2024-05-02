# CuDDHelmholtz
CUDA implementation of parallel domain decomposition methods for preconditioning iterative solvers to the Helmholtz equation.

We consider the Helmholtz equation with zero-order absorbing boundary conditions:

$$-\Delta U - \omega^2 \alpha^2(x) U=f, \qquad \forall x\in\Omega,$$

$$\partial\_{\mathbf{n}} U + i \alpha(x) \omega U=0, \qquad \forall x\in\partial\Omega.$$

Here $\Omega$ is an open and simply connected subset of $\mathbb{R}^2$, $\alpha$ is a real valued positive function, and $f$ is a real valued function.
We solve the Helmholtz equation via the finite element method.
Let $U = u + i v$, the weak formulation is for all $\phi\in H^1(\Omega)$

$$\begin{align*}
(\nabla u, \nabla \phi) - \omega^2 (\alpha^2 u, \phi) - \omega \langle \alpha v,\phi \rangle &= (f,\phi), \\
(\nabla v, \nabla \phi) - \omega^2 (\alpha^2 v, \phi) + \omega \langle \alpha u,\phi \rangle &= 0.
\end{align*}$$

Here
$$(f, g) = \int_\Omega f g \\, dx, \qquad \langle f, g \rangle = \int_{\partial\Omega} f g \\, ds.$$

Let $\\{\phi_i\\}_{i=1}^n$ be the FE basis functions, and define the matrices

$$S_{ij} = (\nabla \phi_i, \nabla \phi_j), \quad M_{ij} = (\alpha\phi_i, \phi_j), \quad H_{ij} = \langle \phi_i, \phi_j \rangle.$$

Let $F_i = (f, \phi_i)$. Then the solutions $u_h, v_h$ are given by

$$u\_h = \sum\_{i=1}^n \hat{u}\_i \phi\_i, \qquad v\_h = \sum\_{i=1}^n \hat{v}\_i \phi\_i.$$

With the coefficients $\hat{u}, \hat{v}$ satisfying

$$\begin{pmatrix}
    S-\omega^2M & -\omega H \\
    -\omega H & \omega^2M-S
\end{pmatrix} \begin{pmatrix}
    \hat{u} \\
    \hat{v}
\end{pmatrix} = \begin{pmatrix}
    F \\
    0
\end{pmatrix}.$$

Krylov space methods for solving the Helmholtz equation are known to converge very slowly.
In addition, for high frequency problems, we must take $n$ very large, so GMRES must be restarted every $m \ll n$ Arnoldi iterations as the cost scales like $O(nm^2)$ per iteration.
With restarts, however, convergence is slower still, so a good preconditioner is essential in order to solve the Helmholtz equation efficiently.
Here we consider a domain decomposition method.
