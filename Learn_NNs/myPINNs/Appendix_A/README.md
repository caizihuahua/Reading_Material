
> **Euler equation**:
> $$
> \partial_t U + \nabla\cdot f(U)=0,x\in\Omega\subset\mathbb{R}^d,d=1,2,t\in(0,T]  \tag{1.1}
> $$
> 
> $$
> \begin{aligned}
> &\text{in the one-dimentional:}\\
> & \qquad\qquad U=\begin{pmatrix}
> \rho\\\rho u\\\rho E
> \end{pmatrix},
> f(U)=\begin{pmatrix}
> \rho u\\ \rho u^2+p \\ u(\rho E+p)
> \end{pmatrix} \\
> &\text{in the one-dimentional:}\\
> & \qquad\qquad U=\begin{pmatrix}
> \rho\\\rho u_1 \\ \rho u_2 \\\rho E
> \end{pmatrix},
> f = (G_1,G_2),
> G_i=\begin{pmatrix}
> \rho u_i\\ \delta_{i1}p+\rho u_1u_i\\ \delta_{i2}p+\rho u_2u_i \\pu_i+\rho u_i E)
> \end{pmatrix}
> \end{aligned}
> $$

Consider the Euler equation $(1.1)$ in the one-dimensional case with the periodic boundary conditions

$$
U(a,t) = U(b,t), \qquad \nabla U(a,t) = \nabla U(b,t)
$$

and the initial conditions

$$
U_0 = (\rho_0,u_0,p_0) = (1.0+0.2\sin(\pi x),1.0,1.0)
$$

in which case we have the exact solutions

$$
(\rho,u,p) = (1.0+0.2\sin(\pi(x-t)),1.0,1.0)
$$

where $x\in(-1,1)$

> **Solution**

$$
Loss = Loss_{Data(IC/BC)} + Loss_F
$$

In the test, we use randomly distributed training points and the number of training points is $N_{BC} = 50, N_{IC} = 50, N_F = 2000$. We employ a neural network that has 7 hidden layers with 20 neurons in each layer and train the model using the Adam optimizer for 5000 steps with an initial learning rate of 0.001 followed by a L-BFGS-B optimizer with 2000 steps.