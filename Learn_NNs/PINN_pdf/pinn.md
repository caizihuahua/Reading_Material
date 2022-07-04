- forward problem (Euler equations and **initial/boundary** formulate the loss function)
  - one-dimensional Euler equation
    - smooth solution
    - solution that have a contact discontinuity
  - two-dimensional oblique shock wave problem
  - a few scattered points clustered randomly around the discontinuities
- inverse problem (mimicking the Schlieren photography experimental technique)
  - data =density gradient $\nabla\rho(x,t)$ + the pressure $p(x^*,t)$ + conservation law   ->  density+velocity+pressure field
  - Euler equations characteristic

random points method are not accurate than traditional numerical method in forward problem

## 1. Introduction

conservation of mass, momentum and energy for compressible flow in the inviscid limit can be modeled by the Euler equations
$$
\partial_t U + \nabla\cdot f(U)=0,x\in\Omega\subset\mathbb{R}^d,d=1,2,t\in(0,T]  \tag{1.1}
$$
$$
\begin{aligned}
&\text{in the one-dimentional:}\\
& \qquad\qquad U=\begin{pmatrix}
\rho\\\rho u\\\rho E
\end{pmatrix},
f(U)=\begin{pmatrix}
\rho\\ \rho u^2+p \\ u(\rho E+p)
\end{pmatrix} \\
&\text{in the one-dimentional:}\\
& \qquad\qquad U=\begin{pmatrix}
\rho\\\rho u_1 \\ \rho u_2 \\\rho E
\end{pmatrix},
f = (G_1,G_2),
G_i=\begin{pmatrix}
\rho u_i\\ \delta_{i1}p+\rho u_1u_i\\ \delta_{i2}p+\rho u_2u_i \\pu_i+\rho u_i E)
\end{pmatrix}
\end{aligned}
$$

need also the equation of state describing the relation of the pressure and energy

consider the equation of state for a polytropic gas given by
$$
p=(\gamma-1)(\rho E-\frac{1}{2}\rho||\bold{u}||^2), \gamma=1.4 \text{ is the adiabatic index} \tag{1.2}
$$

## 2. Methodology

compressible inviscid flow governed by the Euler equation

state $U(\rho,u,p)$  + conservation laws
$$
Loss=Losss_{data}+Loss_F=MSE_{IC}+MSE_{BC}+MSE_{F}
$$

## 3. Forward problems



## 4. Inverse problem

global conservation of mass and momentum

equation of state (EOS)

### 4.1 One-dimensional prolems



#### 4.1.1 Sod problem



#### 4.1.2 Influence of the position of $x^*$ of the pressure probe



#### 4.1.3 Effect of the distribution of the training points



## 5. Conclusion and discussion

