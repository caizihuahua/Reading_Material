> # How much can one learn a partial differential equation from its solution
>
> Yuchen He, Hongkai Zhao, Yimin Zhong
>
> https://arxiv.org/abs/2204.04602 

---

## 1. Introduction

We can view a PDE as an effective way to model the underlying dynamics, for example,heat/diffusion equation, convection/transport equation, Schrodinger equation, Navier-Stokes equation.


$$
\text{underlying dynamic system}
\implies
\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad
\implies

\left\{
\begin{aligned}
&\partial_tu(x,t)=-\mathcal{L}, \quad (x,t)\in\Omega\times[0,T] \\
&u(x,0)=u_0(x). \\
&\text{where }\, \mathcal{L}f(x) \triangleq \sum_{|\alpha|=0}^np_{\alpha}(x)\partial^{\alpha}f(x)
\end{aligned}
\right.
$$

Learn a partial differential equation (PDE) from its solution data: given a set of data $\{u(x_i,t_j)\}_{i,j}$ ,find the mapping $u(x,t)$.

And we notice that $u(x,t)=\int_0^tu_t(x,t)dt+u(x,0)$, so the point is to find the mapping $u_t: (x,t)\to u_t(x,t)$.



**Tow Approaches to Learn the Mapping $u_t \to u_t(x,t)$** 

> Differential Operator Approximation (DOA)

Several recent works proposed to use various types of neural networks to approximate the operator/mapping.

$u_t(x,t),\text{for all}(x,t) \qquad\Leftarrow\qquad u_t(x_i,t_j) \qquad\Leftarrow\qquad (u(x,t),u(x,t+\Delta t))$

We don't need to do any transformation or processing on the data, we just need to ensure that the data is enough, and also, good enough. So the author thinks that this approach is general and flexible. 

But it means more data and more computational cost are required to train such a model, and it does not explicitly reconstruct the differential operator. (In other words, it does not take advantage of the compact parameterized form of a differential operator and hence requires a large degree of freedoms to represent the mapping)

For example

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/transport_eqution.png)

We can use the transport equation to approximate the system, and we  will reduce the amount of computation and get a good fit.
$$
\begin{cases}
u_t(x,t)=Cu_x(x,t)\\
u(x,0)=u_0(x))
\end{cases}
$$

So If we can use the combination of some certain candidate functions to approximate the $u_t(x,t)$, then we can compute fast and less data is required.

> Differential Operator Identification (DOI)

A PDE model typically does not have many terms. So we can determine the explicit form of the underlying PDE.

---

**polynomial interpolation.**

$x(t)\quad t\in[0,T]$, 

- how much point we need
- how to choose basis function (Lagrange polynomi, Newton  polynomials, Chebyshev polynomials)
- regression problem to solve $Ax=b$

$$
\begin{aligned}
&\begin{cases}
    &a_0+a_1\psi_1(x_1)+a_2\psi_2(x_1)+\cdots+a_n\psi_{n-1}(x_1)=f(x_1)\\
    &a_0+a_1\psi_1(x_2)+a_2\psi_2(x_2)+\cdots+a_n\psi_{n-1}(x_2)=f(x_2)\\
    &\qquad \qquad \qquad\vdots \\
    &a_0+a_1\psi_1(x_n)+a_2\psi_2(x_n)+\cdots+a_n\psi_{n-1}(x_n)=f(x_n)
\end{cases}
\implies \\ \\
&\begin{bmatrix}
    & \psi_1(x_1) & \psi_2(x_1) & \cdots & \psi_{n}(x_1) \\
    & \psi_1(x_2) & \psi_2(x_2) & \cdots & \psi_{n}(x_2) \\
    &\vdots&\vdots&\cdots&\vdots \\
    & \psi_1(x_n) & \psi_2(x_n) & \cdots & \psi_{n}(x_{n})
\end{bmatrix}
\begin{bmatrix}
    a_1\\
    a_2\\
    \vdots\\
    a_n
\end{bmatrix}
=
\begin{bmatrix}
    f(x_1)\\
    f(x_2)\\
    \vdots\\
    f(x_n)
\end{bmatrix}
\end{aligned}
$$
$$
\min
\left[
\left.
\begin{pmatrix}
\psi_1(x) \, \psi_2(x)\cdots\psi_n(x)
\end{pmatrix}
\begin{pmatrix}
a_1 \\ a_2 \\ \vdots \\ a_n
\end{pmatrix}
-
f(x)
\right|_{data}
\right]
$$

It is the same for DOI, according to the different type of the underlying system

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/solve.jpg)


$$
\min
\left[
\left.
\begin{pmatrix}
\cdots \text{candidate functions} \cdots
\end{pmatrix}
\begin{pmatrix}
\vdots \\ \xi \\ \vdots
\end{pmatrix}
-
u_t(x,t)
\right|_{data}
\right]
$$

- how much data we need according to the different underlying operator
- how to choose the dictionary (the set of the candidate functions)
- Linear regression

**Chap2** characterize the data space: 

Study the dimension of the space spanned by all snapshots of a solution trajectory with certain tolerance, and show how it is affected by the underlying PDE operator and initial data. 

**Chap3** study the identifiability and the stability issue.

**Chap4** propose a data driven and data adaptive approach for general PDEs with variable coefficients

Consistent and Sparse Local Regression (CaSLR)

- globally consistent
- built from as few terms as possible from the dictionary
- a good local fit to data using different linear combinations at different locations.

> Nathan Kutz: Data Driven Discovery of Dynamical Systems and PDEs
> https://www.youtube.com/watch?v=Oifg9avnsH4
>
> Steve Brunton: System Identification: Sparse Nonlinear Models with Control
> https://www.youtube.com/watch?v=vuJCOfdlN6Q

## 2. Data space spanned by solution trajectory


a single solution trajectory

1. a strongly elliptic operator
2. a first order hyperbolic operator

Theorem 2.8 (for a strongly elliptic operator $\mathcal{L}$), all snapshots of any single trajectory $u(x,t)$ stays $\varepsilon$ close to a linear space of dimention at most of order $O(|log\,\varepsilon|^2)$. 

If $\mathcal{L}$ is a first order hyperbolic operator, the data space spanned by all snapshots of a single trajectory stays $\varepsilon$ close to linear space of dimension $O(\varepsilon^{−\gamma})$), where γ depends on the regularity of the initial data.

### 2.1 Strongly elliptic operator

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/defination_2_1_1.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/defination_2_1_2.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/defination_2_1_1.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/corollary_2_6_1.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/corollary_2_6_2.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/lemma_2_7.jpg)

$$
u(x,t) = e^{\mu t}\sum_{k=1}^\infty c_ke^{-\lambda_kt}\phi_k(x) \tag{14}
$$
$\textit{Theorem 2.8}$ : Suppose $\mathcal{L}_\mu$ is admissible that the spectrum sits in the interior of the sector $\Sigma_\delta$ and the coefficients in $(14)$ decay as $|c_k|\le\theta k^{-\gamma}, \theta\gt 0,\gamma\gt 1$, then theres exits a linear space $V \subset L^2(\Omega)$ of dimention $C_\mathcal{L}(k)|log\,\varepsilon|^2$ such that
$$
||u(\vdot,t)-P_Vu(,\vdot,t)||\le C\varepsilon ||u_0||, \forall t\in[0,T].
$$

where $ P_V$ is the projection operator onto V and $C=C(\theta,\gamma)$. The constant $C_\mathcal{L}(k)$ is chosen from Corollary $\textit{2.3}$ and $k = \mathcal{O}(\beta/(\gamma-1))$

---

### 2.2 Hyperbolic PDE

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/A_Integral_Operator.jpg)

$\textit{Lemma 2.11}$ : Let $\boldsymbol{c}(x)\in C^p(\Omega)$ be a velocity field and $u_0\in C^p(\Omega)$, then there exists a subspace $V\subset L^2(\Omega)$ of the dimention $\mathcal{O}(\varepsilon^{-2/p})$ that
$$
\sqrt{\int_0^T||P_Vu(\vdot,t)-u(\vdot,t)||_{L^2(\Omega)}^2dt}\le\varepsilon \tag{29}
$$

### 2.3 Numerical examples

the factor which effects the dimension of the data space

> the effect of the different kind of  $\mathcal{L}$ 

First, we show how the dimension of the data space corresponding to a single solution trajectory depends on the PDE operator.

- PDE with constant coefficient

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/operator_types.jpg)

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/operator_types_figure.jpg)

- PDE with variable coefficients

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/operator_types_2.jpg)

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/operator_types_figure_2.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/dimension_analysis.jpg)

> the effect of the initial data 

- initial data with different regularity

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/initial_data.jpg)

For initial data with different regularities, we show percentage of dominant singular
values $\lambda >\epsilon$, for different threshold $\epsilon > 0$ in Figure 3.

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/initial_data_figure_1.jpg)

- initial data with different number of Fourier modes

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/initial_data_fourier_mode.jpg)

![](file:///home/boogie/1910063/Study_math/HKUST/PDE/share/initial_data_figure_2.jpg)

## 3 PDE identification from a single solution trajectory

$$
\partial_tu(x,t)=-\mathcal{L}, \quad (x,t)\in\Omega\times[0,T] \\
u(x,0)=u_0(x). \\
\text{where}\quad \mathcal{L}f(x) \triangleq \sum_{|\alpha|=0}^np_{\alpha}(x)\partial^{\alpha}f(x)
$$

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/indentification.jpg)



- if $\psi_y(x)=\delta(x-y)$, y is the parameter,

$$
\begin{aligned}
\partial_tu(y,t)&=-\sum_{\alpha=0}^np_\alpha(y)\partial^\alpha u(y,t)\\
&=(\cdots \text{candidate functions}\cdots)\vdot
    \begin{pmatrix}
    \vdots \\
    \xi \\
    \vdots
    \end{pmatrix}
\end{aligned}
$$
The PDE identification problem becomes a linear regression problem using the solution and its derivatives sampled at different locations in space and time, which has been the main study in the literature.

- if $\psi(x)=(2\pi)^{-d/2}e^{-i\zeta\vdot x}$ and coefficient is constant $p_\alpha\in\mathbb{R}$ 

$$
\begin{aligned}
\partial_t\widehat{u}(\zeta,t)&=-\sum_{|\alpha|=0}^np_\alpha\langle\partial^\alpha u(x,t),(2\pi)^{-d/2}e^{-i\zeta\vdot x}\rangle \\
&=-(2\pi)^{-d/2}\sum_{|\alpha|=0}^n p_\alpha(i\zeta)^\alpha\widehat{u}(\zeta,t)
\end{aligned}
$$



### 3.1 PDE identification with constant coefficient

One can transform it into Fourier domain and show that the underlying differential operator $\mathcal{L}$ can be identified by one trajectory at two different instants if and only if the solution contains enough Fourier modes.

$$
\widehat{u}(\zeta,t)=(2\pi)^{-d/2}\int_\Omega e^{-i\zeta\cdot x}u(x,t)dx
$$
The PDE $\partial_tu=\mathcal{L}u, \mathcal{L}=\sum_{|\alpha|=0}^np_\alpha\partial_x^\alpha u(x,t)$ is converted to an ODE
$$
\partial_t\widehat{u}(\zeta,t)=-(2\pi)^{-d/2}\sum_{|\alpha|=0}^n p_\alpha(i\zeta)^\alpha\widehat{u}(\zeta,t)
$$
The solution is
$$
\widehat{u}(\zeta,t)=\widehat{u}(\zeta,0)\text{exp}\left(-(2\pi)^{-d/2}\sum_{|\alpha|=0}^n p_\alpha(i\zeta)^\alpha t\right)
$$
But we have no way to measure the data at the initial moment, so we don't know $\widehat{u}(\zeta,0)$

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/indentification_constant_v.jpg)


### 3.2 PDE identification with variable coefficients using a single trajectory

There is no useful analysis

### 3.3 Identifiability with a single trajectory

More Fourier modes in the initial data and more snapshots along the trajectory depending on the order of the differential operators and space dimensions are needed for identifiability

### 3.4 Possible instability for identification of elliptic operator

use elliptic operator as an example to show possible instability

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/theorem_3_4.jpg)

- First we show local instability when one has short observation time for a single trajectory

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/instability_3_4.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/stability_example.jpg)

$$
\partial^\alpha u(x,t_k) = \sum_{t=0}^{m-1}(-1)^l\frac{t_k^l}{l!}\mathcal{L}^lu_0(x)
=
\begin{pmatrix}
t_k^0, t_k\,\cdots,t_k^{m-1}
\end{pmatrix}
\begin{pmatrix}
\frac{(-1)^0}{0!}\partial^\alpha \mathcal{L}^0u_0(x) \\
\frac{(-1)^1}{1!}\partial^\alpha \mathcal{L}^1u_0(x) \\
\vdots\\
\frac{(-1)^{m-1}}{m-1!}\partial^\alpha \mathcal{L}^{m-1}u_0(x) \\
\end{pmatrix}
$$


![](/home/boogie/1910063/Study_math/HKUST/PDE/share/stability_high_frequency.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/stability_high_frequency_conclusion.jpg)



## 4 Local regression and global consistency enforced PDE identification method

When the PDE has variable coefficients, this amounts to different regression problems at different measurement locations

### 4.1 Proposed PDE identification method

Unknown PDE:
$$
u_t(x,t)=\sum_{k=1}^Kc_k(x,t)f_k(x,t)
$$
where $\mathcal{F}=\{f_k:\Omega\times[0,T]\to\mathbb{R}\}_{k=1}^K$ is a dictionary, e.g. $u_x,u_{xxx},uu_x,u^2,\sin(u)$, $c_k:\Omega\times[0,T]\to\mathbb{R},k=1,2,\cdots,K$ are the respective coeffients

approximate the PDE coefficients by constants in each small patch neighborhood (in space and time) centered at $(x_j,t_j),j = 1,2,\cdots,J$ 

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/CaSLR_method.jpg)



### 4.2 Identification guarantee by local regression

Operators with constant coefficients approximation indeed can identify the underlying PDE with variable coefficients under these conditions: 
1. the variable coefficients are bounded away from zero and vary slowly on the patch
2. the solution data contain diverse information content, on the patch.

We define the admissible set for the coefficients $\mathcal{C}_\varepsilon\subset\mathbb{R}$ by
$$
\mathcal{C}_\varepsilon\triangleq(-\infty,\varepsilon)\cup{0}\cup(\varepsilon,\infty)
$$
![](/home/boogie/1910063/Study_math/HKUST/PDE/share/CaSLR_identification.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/corollary_4_2.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/theorem_4_4.jpg)



In general, the larger $K$ is, the smaller $K_B^p$ will be. In other words, the larger the dictionary, the harder the identification for the underlying PDE.

### 4.3 Data driven and data adaptive measurement selection

First, we propose to use the following numerical estimate of local Sobolev semi-norm to filter out those patches in which the solution may be singular or oscillate rapidly,
$$
\beta(x_j,t_j)=\sqrt{\frac{1}{m_j}\sum_{m=1}^{m_j}\sum_{p=1}^{P_{max}}(\partial_x^pu(x_{j,,m},t_{j,m}))^2} \tag{108}
$$
where $P_{max}$ is the maximal order of the partial derivatives in the dictionary.

We remove those local regressions at $(xj,tj)$ in $\tag{90}$ if $\beta(x_j,t_j) < \beta$ or $\beta(x_j,t_j) >  ̄\beta$ for some thresholds $\bar{\beta},\ \underline{\beta} > 0$. In this work, we fix $\beta$ to be the 1st percentile and  $\beta$ the 99th percentile of all the collected local Sobolev semi-norms

> data with noise

......

> ### My Idea

for local patch, we know 
$$
u_t(x,t)=\sum_{k=1}^Kc_{x_i,t_j}f_k(x,t)
$$

fix $x_i$, we can do regression along the time axis, then obtain $c_{x_i}(t)$

then we do regression along the snapshots ,then we may obtain $c(x,t)$s

### 4.4 Numerical Experiments

In some of the numerical examples, we use the following transition function in time with slope $s\in\mathbb{R}$ and critical point $t_c\in\mathbb{R}$
$$
\tau(t;s,t_c)=0.5+0.5\tanh(s(t-t_c)), t\in\mathbb{R} \tag{114}
$$
we use the Jaccard score
$$
J(S_0,S_1)=\frac{|S_0\cap S_1|}{|S_0\cup S_1|} \tag{115}
$$
In our case, $S_0$ represents the set of indices of the true features in a given dictionary and $S_1$ denotes the set of indices of features in an identified PDE model.

Each sensor collects data in a cubic neighborhood in space and time whose side length is $2r + 1$ in each spatial dimension and $2r_t + 1$ in time. Here $r > 0$ and $r_t > 0$ is referred to as the sensing radius and time duration respectively

> Example 1: Transport equation

$$
\begin{aligned}
&u_t(x,t)=(1+0.5\sin(\pi x)\tau(t;-10,0.5)u_x(x,t),\quad(x,t)\in[-1,1)\times(0,1]\\
&u(x,0)=\sin(4\pi(x+0.1))+\sin(6\pi x) +\cos(2\pi(x-0.5)) +\sin(2\pi(x+0.1)). \\
\end{aligned} \tag{116}
$$

over a grid with 100 points in space and 5000 points in time

Based on local patch data and using a dictionary of size 59 including up to 4-th order partial derivatives of the solution and products of them up to 3 terms, plus $\sin(u), \cos(u), \sin(ux), \text{and} \cos(ux)$. 

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/num_example_1.jpg)



> Example 2: KdV type equation

$$
\begin{aligned}
&u_t(x,t)=(3+200t\sin(\pi x))u(x,t)u_x(x,t)+\frac{5+\sin(400\pi t/3)}{100}u_{xxx}(x,t), \quad(x,t)\in[-1,1)\times(0,1.5\times10^{-2}) \\
&u(x,0)=\sin(4\pi(x+0.1))+2\sin(5\pi x)+\cos(2\pi(x-0.5))+\sin(3\pi x)+\cos(6\pi x)
\end{aligned} \tag{117}
$$

Using a dictionary of size 59 as specified in the previous example

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/num_example_2.jpg)



> Example 3: $\text{Schr}\ddot{o}\text{dinger equation}$

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/num_example_3.jpg)

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/num_example_3_1.jpg)



> Example 4: 2D circular flow

$$
\begin{aligned}
&u_t(x,y,t)=-yu_x(x,y,t)+xu_y(x,y,t), \quad(x,y,t)\in\mathbb{R}^2\times(0,2\pi] \\
&u(x,y,0)=f(x,y), \quad (x,y)\in\mathbb{R}^2
\end{aligned} \tag{119}
$$

the exact solution 

$$
u(x,y,t)=f(\sqrt{x^2+y^2}\cos(\arctan(y/x)-t),\sqrt{x^2+y^2}\sin(\arctan(y/x)-t))
$$
in this example we take $f(x,y)=\cos(4\sqrt{x^2+y^2}\cos(2\arctan(y/x)))$ 

The dictionary consists of 27 features, where partial derivatives up to order 2 and products up to 2 terms are included.

randomly locate 5 sensors with sensing radius of $r (r = 3,5,7)$ on a circle of radius $R$ centered at the origin, and repeat the experiments for 20 times

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/num_example_4.jpg)

#### 4.4.4 Identification with random initial conditions

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/initial_condition_example_1.jpg)

$$
u(x,0)=a_0+\sqrt{2}\sum_{j=1}^M
\left(
a_j\cos(\frac{\pi jx}{L})+b_j\sin(\frac{\pi jx}{L})
\right)
$$


![](/home/boogie/1910063/Study_math/HKUST/PDE/share/initial_condition_example_2.jpg)



#### 4.4.2 Effects od the proposed data-driven patch filtering

$$
\begin{aligned}
&1. \text{transport equation:}\quad u_t(x,t)=1000t\sin(4\pi t/0.03)u_x(x,t) \quad t\in[0,0.03] \\
&2. \text{heat equation:}\quad u_t(x,t)=0.5u_{xx}(x,t), \quad t\in[0,0.03]\\
&3. \text{inviscid Burger's equation:}\quad u_t=1.1u(x,t)u_x(x,t),\quad t\in[0,0.6]
\end{aligned}
$$

For $1$, we added 5%; for $2$ and $3$ we added 0.5%. No denoising process was applied

![](/home/boogie/1910063/Study_math/HKUST/PDE/share/initial_condition_example_3.jpg)



## 5 Conslusion
Using various types of linear evolution PDEs, we have shown
1. how the approximate dimension (richness) of the data space spanned by all snapshots along a solution trajectory depends on the differential operator and initial data
2. identifiability of a differential operator from its solution data.

we propose a Consistent and Sparse Local Regression(CaSLR)