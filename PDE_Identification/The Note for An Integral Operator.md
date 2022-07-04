## The Note for An Integral Operator

<img src="/home/boogie/1910063/Study_math/HKUST/PDE/share/A_Integral_Operator.jpg" alt="A_Integral_Operator" style="zoom:33%;" />

---

**Note**: 我们记该算子为$(\Phi f)(x) := \int_\Omega K(x,y)f(y)dy\\$，$\Phi$的特征向量为$\varphi_1(x),\varphi_2(x),\varphi_3(x),\dots$，与其特征值$\lambda_1\ge\lambda_2\ge\lambda_3\dots$对应。

>  **$K(x,y)$ is a symmetric semi-positive compact integral operators on $L^2(\Omega)$**

### Symmetric Integral Operator

$K(x,y) := \int_0^Tu(x,t)u(y,t)dt$，显然有$K(x,y)=K(y,x)$，所以 $\Phi$ 是对称积分算子。

### [Compact](https://hc1023.github.io/2019/11/05/integral-operators/) 

#### 算子有界

$$
\begin{aligned}
\int_\Omega [\Phi(f)(x)]^2dx&=\int_\Omega[\int_\Omega K(x,y)f(y)dy]^2dx \\
&\le \int_\Omega [\int_\Omega K^2(x,y)dy][\int_\Omega f^2(y)dy]dx \quad(\href{https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality}{\text{Cauchy–Schwarz inequality}})\\
&=[\int_{\Omega^2}K^2(x,y)dxdy][\int_\Omega f^2(y)dy] \quad (\href{https://en.wikipedia.org/wiki/Fubini%27s_theorem}{\text{Fubini's theorem}}) \\
&=||K(x,y)|||_{L^2(\Omega^2)}*||f(x)||_{L^2(\Omega)} \\
\end{aligned}
$$

$所以有||\Phi||\le||K(x,y)||=\int_{\Omega^2\times[0,t]}u(x,t)u(y,t)dtdxdy\le\infty$。第二个不等号是因为 $\Omega$ 和 $[0,T]$ 的测度有限，$u(x,t)$ 连续。

#### 紧算子
取 $L^2(\Omega)$ 的标准正交基 $\{\varphi_i(x)\}_{i=1}^\infty$，则有$\{\varphi_i(x)\varphi_j(y)\}_{i,j\ge1}$ 构成 $L^2(\Omega^2)$ 的标准正交基。则有
$$
K(x,y)=\sum_{i,j=1}a_{i,j}\varphi_i(x)\varphi_j(y), \sum_{i,j=1}^\infty|a_{i,j}|\lt\infty
$$
定义 $K_n(x,y):=\sum_{i,j=1}^{n}a_{i,j}\varphi_i(x)\varphi_j(y)$，有
$$
||\Phi-\Phi_n||\le||K(x,y)-K_n(x,y)||_{L^2(\Omega^2)}=\sum_{i\ge orj\ge n}^\infty|a_{i,j}|^2\to0(n\to\infty)
$$
所以 $\Phi$ 为紧算子。

### Normalized Eigenfunctions of $\Phi$ Form an Orthonormal Basis in $L^2(\Omega)$

因为特征值递减到零，由[**`Hilbert-Schmidt theorem`**](https://en.wikipedia.org/wiki/Hilbert%E2%80%93Schmidt_theorem)知，其特征向量$\{\varphi_i(x)\}_{i=1}^\infty$构成$L^2(\Omega)$的一组标准正交基。

### Semi-positive
$$
\begin{aligned}
\lambda_i\varphi_i(x)&=\int_\Omega K(x,y)\varphi_i(y)dy \\
\int_\Omega\lambda_i\varphi_i(x)\varphi_i(x)dx&=\int_{\Omega^2} K(x,y)\varphi_i(y)dy\varphi_i(x)dx \\
\end{aligned}
$$
对于左手边，因为 $\varphi_i(x)$ 构成 $L^2(\Omega)$ 的标准正交基，所以 $\text{LH}=\lambda_i$，对于右手边
$$
\begin{aligned}
\text{RH}&=\int_{\Omega^2}[\int_0^Tu(x,s)u(y,s)ds]\varphi_i(x)\varphi(y)dxdy \\
&=\int_0^T(\int_\Omega u(x,s)\varphi_i(x)dx)(\int_\Omega u(y,s)\varphi_i(y)dy)ds \\
&=\int_0^T(\int_\Omega u(x,s)\varphi_i(x)dx)^2dt
\ge0 \\
\end{aligned}
$$
所以
$$
\lambda_i=\text{LH}=\text{RH}\ge0
$$

### Proof for (28)
> $\int_0^T||u(\cdot,t)-P_{V_K^k}u(\cdot,t)||_{L^2(\Omega)}^2dt =  \sum_{j=k+1}^\infty\lambda_j \quad (28) $

1. #### $k=0$ 时
   因为 $\{\varphi_i(x)\}_{i=1}^\infty$ 是 $L^2(\Omega)$ 的一组正交基，所以记 $u(x,t)=\sum_{i=1}^\infty c_i(t)\varphi_i(x)$，其中$ c_i(t)=\int_\Omega u(x,t)\varphi_i(x)dx$ 。

$$
\begin{aligned}
\text{LH}&=\int_{\Omega\times[0,T]}u^2(x,t)dxdt \\
&=\int_{\Omega\times[0,T]}\sum_{i,j=1}c_i(t)\varphi_i(x)c_j(t)\varphi_j(x)dxdt \\
&=\sum_{i,j}\int_\Omega\varphi_i(x)\varphi_j(x)dx\int_0^Tc_i(t)c_j(t)dt \\
&=\sum_i\int_0^Tc_i^2(t)dt \quad (\varphi_i(x)\,\text{正交}) \\
\end{aligned}
$$

$$
\begin{aligned}
\int_0^Tc_i^2(t)dt
&=\int_0^T(\int_\Omega u(x,t)\varphi_i(x)dx)(\int_\Omega u(y,t)\varphi_i(y)dy)dt\\
&=\int_{\Omega^2}(\int_0^Tu(x,t)u(y,t)dt)\varphi_i(x)\varphi_i(y)dxdy \\
&=\int_{\Omega^2}K(x,y)\varphi_i(x)\varphi_i(y)dxdy \\
&=\int_\Omega[\int_\Omega K(x,y)\varphi_i(x)dx] \varphi_i(y)dy \\
&=\int_\Omega \lambda_i\varphi_i(y)\varphi_i(y)dy \\
&=\lambda_i \\
\end{aligned}
$$

所以有
$$
\text{LH}=\sum_{i=1}^\infty\int_0^Tc_i^2(t)dt =\sum_{i=1}^\infty\lambda_i
$$


2. #### 对任意的的 $k$
$u(\cdot,t)-P_{V_K^k}u(\cdot,t)=\sum_{i=k+1}^\infty c_i(t)\,\varphi_i(x),$

由上推导，易知不等式 (28) 成立。
