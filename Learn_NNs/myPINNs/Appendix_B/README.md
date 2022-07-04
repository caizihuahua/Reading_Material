
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

> **Solution**

$$
Loss = Loss_{Data} + Loss_F = MSE_{\nabla\rho}+MSE_{p^*}+MSE_F
$$

where $x^*$ is a randomly given point 

7 hidden layers with 20 neurons at each layers

$N_{\nabla \rho}=640, N_p=50, N_F=2000$

Adam optimizer with an initial learning rate of 0.001 for 5000 steps followed by a L-BFGS-B optimizer with 3000 steps

> **promotion**

$$
Loss = MSE_{\nabla\rho}+MSE_{p^*} + MSE_{mass_0} + MSE_F
$$

$$
MSE_{mass_0} = (\int_\Omega \rho_{NN}(x,0)dx - \int_\Omega \rho_0(x)dx)^2
$$