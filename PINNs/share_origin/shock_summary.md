## 使用神经网络捕捉激波

> 问题描述：流体力学中无粘可压缩的流动，该流动可用欧拉方程来描述

$$
\partial_tU+\nabla\cdot f(U)=0, \quad x\in\Omega\subset\mathbb{R}^d, \quad d=1,2, \quad t\in(0,T]
$$

- 一维

$$
U=\begin{pmatrix}
\rho \\
\rho u \\
\rho E
\end{pmatrix},
\quad
f(U)=\begin{pmatrix}
\rho u \\
\rho u^2+p \\
u(\rho E+p)
\end{pmatrix}
$$

- 二维

$$
U=\begin{pmatrix}
\rho \\
\rho u_1 \\
\rho u_2 \\
\rho E
\end{pmatrix},
\quad
f=(G_1,G_2),\quad
G_i(U) = \begin{pmatrix}
\rho u_i \\
\delta_{i1}p+\rho u_1u_i \\
\delta_{i2}p+\rho u_2u_i \\
pu_i+\rho u_i E
\end{pmatrix},
\quad i=1,2
$$

我们还需要一个方程来封闭这个问题，例如我们可以有
$$
p = (\gamma-1)(\rho E-\frac{1}{2}\rho||u||^2), \quad \gamma=1.4
$$


> 研究方法：内嵌物理知识神经网络(Physics-Informed Neural Network,PINN)

![image-20220716220854769](pics\PINNs_structure.png)

神经网络：喂入 $(x,t)$ ，吐出 $U_{nn}(x,t)$ 。（也就是能得到映射关系 $U_{nn}$

现在我们考虑两部分损失

1. 数据损失（$Loss_{data}$，uniformed loss）：所有已知点 $(x_j,t_j)$ 处，计算 $U_{nn}(x_j,t_j)$ 与已知数据 $U(x_j,t_j)$ 的损失
2.  PDE损失（$Loss_{pde}$，informed loss）：把 $U_{nn}$ 带入算子 $\partial_t+\nabla\cdot f$，计算数据增强点 $(x_j^{pde},t_j^{pde})$ 处的损失，或者叫做残差（**residual**）

$$
Loss = Loss_{data} + Loss_{pde}
$$

MSE：均方根误差（mean square errors）

我们记残差（residual）为
$$
r(x,t) \triangleq \partial_tU_{nn}(x,t)+\nabla\cdot f(U_{nn}(x,t))
$$

$$
\begin{aligned}
Loss_{pde}&\\
&= MSE_{pde}\\ 
&= \frac{1}{N_{pde}}\sum_{j=1}^{N_{pde}}r(x_{j}^{pde},t_j^{pde})^2 \\
&=\frac{1}{N_{pde}}\sum_{j=1}^{N_{pde}}|\partial_tU_{nn}(x_{j}^{pde},t_j^{pde})+\nabla\cdot f(U_{nn}(x_{j}^{pde},t_j^{pde}))|^2
\end{aligned}
$$

### 1. 聚集采样（clustered）

> 在会出现激波的地方采更多的点

#### 正问题（forward problem, 已知初边值 IC/BC）

$$
\begin{aligned}
& \partial_tU+\nabla\cdot f(U)=0 \\
& U(x,t) = g(x,t),\quad x\in\partial\Omega, \quad t\in[0,T] \\
& U(x,0)=U_0(x), \quad x\in\Omega
\end{aligned}
$$

$$
Loss = Loss_{data} + Loss_{pde} = MSE_{IC} + MSE_{BC} + MSE_{pde}
$$

例子：

控制方程：
$$
\partial_t \begin{pmatrix}
\rho \\
\rho u \\
\rho E
\end{pmatrix}
+\nabla \cdot \begin{pmatrix}
\rho u \\
\rho u^2+p \\
u(\rho E+p)
\end{pmatrix}
= 0, \quad E = \frac{p}{\rho(\gamma-1)}+\frac{1}{2}||u||^2, \quad \gamma=1.4 \\
$$
考虑空间为 $(0,1)$，边值 $BC$ 为 Dirichlet 边值条件
$$
U(0,t)=(1.4,0.1,1.0), \quad U(1,t)=(1.0,0.1,1.0)
$$
初值 $IC$ （间断点在 $x=0.5$ 处）：
$$
(\rho_L,u_L,p_L) = (1.4,0.1,1.0), \quad (\rho_r,u_r,p_r) = (1.0,0.1,1.0)
$$
实际解：
$$
\rho(x,t)=\left\{
\begin{aligned}
1.4, \quad x<0.5+0.1t, \\
1.0, \quad x>0.5+0.1t, \\
\end{aligned}
\right.
\quad u(x,t)=0.1, \quad p(x,t)=1.0
$$

考虑初边值损失
$$
\begin{aligned}
& MSE_{IC}=\frac{1}{N_i}\sum_{j=1}^{N_i}|U_0(x_j^{IC})-U_{nn}(x_j^{IC})|^2 \\
& MSE_{BC}=\frac{1}{N_b}\sum_{j=1}^{N_b}(|U_{nn}(0,t_j^{BC})-U(0,t_j^{BC})|^2 + |U_{nn}(1,t_j^{BC})-U(1,t_j^{BC})|^2 \\
\end{aligned}
$$

采样策略

<img src="pics\forward_problem_ex1.png" alt="image-20220716232337696" style="zoom: 30%;" />

误差估计
$$
Error_\rho := || \rho(x,2)-\rho_{nn}(x,2)||_{L^2}/ || \rho(x,2)||_{L^2}
$$
<img src="pics\forward_problem_ex1_error.png" alt="image-20220716233426825" style="zoom: 50%;" />

### 2. 增加额外的损失函数，调整全局损失函数权重

#### 逆问题（inverse problem, 不知道初边值，只知道某些数据）

> motivated by mimicking the Schlieren photography experimental technique used traditionally in high-speed aerodynamics, we use the data on density gradient $\nabla\rho(x,t)$

这里我们考虑已知的数据为：某些点处的 $\nabla\rho(x,t)$ 和 $p(x^*,t)$，其中 $x^*$ 是预先选取的。
$$
Loss = Loss_{data} + Loss_{pde} = MSE_{\nabla\rho}+MSE_{p^*}+MSE_{pde}
$$
<img src="pics\inverse_problem_MSE.png" alt="image-20220716234224455" style="zoom: 50%;" />

<img src="pics\inverse_problem_nabla_rho_loss.png" alt="image-20220717000227396" style="zoom: 40%;" />

![](pics\inverse_problem_sample_points.png)





但hi对于如下连续解的问题，只使用这三个损失函数拟合出来的结果不太好。
$$
U=(\rho,u,p)=(1.0+0.2\sin(\pi (x-t)),1.0,1.0)
$$
训练完后作图， $t=0$ 

![image-20220716234845814](pics\inverse_problem_ex6.png)
$$
Loss = MSE_{\nabla\rho}+MSE_{p^*}+MSE_{pde} + MSE_{mass_0}
$$

$$
MSE_{mass_0} = (\int_\Omega\rho_{nn}(x,0)dx - \int_\Omega\rho_0(x)dx)^2
$$

![image-20220717082004410](pics\inverse_problem_ex3.png)
$$
Loss = \omega_{\nabla\rho}MSE_{\nabla\rho}+\omega_{p^*}MSE_{p^*}+\omega_{m_0}MSE_{mass_0}+\omega_{ped}MSE_{pde}
$$

$$
\omega_{\nabla\rho}=\omega_{pde}=0.1, \quad \omega_{p^*}+\omega_{m_0}=1.0, \quad dx=0.008, \\
N_{\nabla\rho}=2334, \quad N_{pde}=3338, \quad N_{p^*}=200,\quad x^*=2.5
$$

4层中间层，每层120个神经元，Adam算法，初始时刻学习率0.0005，8 000步。然后L-BGGS-B迭代200 000步

![image-20220717082745387](pics\inverse_problem_ex3_results.png)

现在考虑
$$
Loss = \omega_{\nabla\rho}MSE_{\nabla\rho}+\omega_{p^*}MSE_{p^*}+\omega_{m_0}MSE_{mass_0}+\omega_{ped}MSE_{pde}+\omega_{Mom}MSE_{Mom}
$$
<img src="pics\inverse_problem_momentum_loss.png" alt="image-20220717083224100" style="zoom: 50%;" />

![image-20220717083329146](pics\inverse_problem_ex3_results_pro.png)

So we suggest to choose the point $x^*$ from the domain that is between the initial discontinuous point and the shock point at the final time.

> 使用经典形式的欧拉方程 （in characteristic form）

$$
U_t+AU_x=0, \quad A(U)=\frac{\partial f}{\partial U}
$$

$$
LU_t+DLU_x = 0
$$

![image-20220717083959552](pics\euler_in_characteristic_form.png)
$$
Loss_{pde} = ||LU_t+DLU_x||
$$

$$
Loss = MSE_{\nabla\rho}+MSE_{p^*}+MSE_{mass_0}+MSE_{pde}+MSE_{Mom}
$$

3*120，Adam lr=0.0005 12 000steps + L-BFGS-B 200 000steps

![image-20220717084333298](pics\euler_in_characteristic_form_results.png)

### 对于 shock-tube 问题

#### 调整全局权重策略

![image-20220717091121451](pics\problem_setting.png)

<img src="pics\shock_tube_problem.png" alt="image-20220717090541956" style="zoom: 50%;" />

<img src="pics\shock_tube_problem_figure.png" alt="image-20220717123304581" style="zoom:50%;" />

> The Euler equations are of hyperbolic type which means that it, in essence, propagates the initial condition through the domain.

$$
Loss = w_{ib}Loss_{IC/BC}+w_{pde}Loss_{pde}, \\
0\le Loss_{IC/BC} \le Loss_{pde}
$$
$\omega_{ib}$ 是 $\omega_{pde}$ 的100倍，$\omega_{ib}=10$ 是 $\omega_{pde}=0.1$

其中我们记，$G(\theta):=Loss$，$G(\theta)_f:=Loss_{pde}$，$G(\theta)_{IC}:=Loss_{IC/BC}$

<img src="pics\loos_vs_weighted_loss.png" alt="image-20220717091522377" style="zoom: 40%;" />

<img src="pics\zoom_in_density.png" alt="image-20220717092014377" style="zoom:50%;" />

#### 调整局部权重

**Fact 1**: the shock wave has no thickness in theory without considering viscosity. So it can not be governed by the differential equations for its infinite gradient but can be controlled by a physics compression process from the left and right statuses (Rankine–Hugoniot conditions).

**Fact 2**: NNs are also bad at approximating first-order discontinuous functions( the universal approximation theorem is only effective with continuous functions). Then the training of NNs tends to reduce the gradient in a steep jumping region.

但是我们需要考虑的是对于那些落在激波上的采样点，这样的采样点我们称为 "trouble point"。

对于 “trouble point” ，我们希望在最后的结果中，这些点处的梯度无穷大，这样才符合物理规律。但这样在这些点处计算出的损失也是非常大的，而神经网络希望减小这些损失，也就是神经网络更倾向于输出连续的东西，所以这里就会有矛盾。

1. 物理规律希望，梯度越大，但是这会带来很大的损失
2. 神经网络希望，损失很小，但是这会使梯度减小

所以我们考虑这样的损失函数
$$
Loss = \lambda(x,t)Loss_{pde}+Loss_{data}
$$
也即我们有
$$
\lambda(x,t)Loss_{pde}=\frac{1}{\varepsilon_2(|\nabla\cdot\vec{U}|-\nabla\cdot\vec{U})+1}(\frac{\partial U}{\partial t}+\nabla\cdot f(U))
$$
直观解释：

重点在与训练光滑的情况，然后把光滑的情况 **“拼接“** 起来，这样来形成激波。

<img src="pics\weighted_pinn_local_loss.png" alt="image-20220717124206769" style="zoom: 67%;" />

文中的实验结果：

<img src="pics\new_weighted_pinn_problem_setting.png" alt="image-20220717095448810" style="zoom:50%;" />

<img src="pics\new_weighted_pinn.png" alt="image-20220717095357864" style="zoom: 50%;" />

<img src="pics\new_weighted_pinn_vs_origin_pinn.png" alt="image-20220717095729260" style="zoom:50%;" />



### 3. 增加人工粘滞项（Artificial Viscosity，AV）

$$
\begin{aligned}
& \partial_tU+\nabla\cdot f(U)=0 \\
& U(x,t) = g(x,t),\quad x\in\partial\Omega, \quad t\in[0,T] \\
& U(x,0)=U_0(x), \quad x\in\Omega
\end{aligned}
$$

我们考虑
$$
\partial_tU+\nabla\cdot f(U)=\nu\partial_x^2U
$$

$$
r(x,t) \triangleq=\partial_tU+\nabla\cdot f(U)-\nu\partial_x^2U
$$


其中 $\nu$ 是可以事先指定，也可以通过网络学习。
$$
Loss_{pde} = MSE_{pde}
=\frac{1}{N_{pde}}\sum_{j=1}^{N_{pde}}r(x_{j}^{pde},t_j^{pde})^2
\\=\frac{1}{N_{pde}}\sum_{j=1}^{N_{pde}}|\partial_tU_{nn}(x_{j}^{pde},t_j^{pde})+\nabla\cdot f(U_{nn}(x_{j}^{pde},t_j^{pde}))-\nu\partial_x^2U(x_{j}^{pde},t_j^{pde})|^2
$$
例子（**事先指定**）：

<img src="pics\burgers_without_av.png" alt="image-20220717110554816" style="zoom: 67%;" />

我们在计算误差时候考虑，这样的控制方程

<img src="pics\bburgers_with_av_result.png" alt="image-20220717110746580" style="zoom:80%;" />

> 如何调整人工粘滞系数

#### 2.1 全局可学习的人工粘滞项 （Learnable Global Artificial Viscosity)

![image-20220717111530081](pics\learnable_global_av.png)

#### 2.2 带参数的人工粘滞项（Parametric Artificial Viscosity Map）

> 只在激波处增加人工粘滞项

![image-20220717114643177](pics\parametric_av.png)

![image-20220717114814115](pics\parametrix_av_figure.png)

预先知识：**一个激波** 

**$\nu$ 高斯分布**（均值在间断点，方差 $\omega_\nu$ ），在间断点附近有粘滞项，其他地方很小

$\theta$ 记录列如：间断点发生点（截距），间断点传播速度（斜率）等等，这些是需要学习的

圆点表示我们的 **数据增强点**，颜色代表 **粘滞系数** 

#### 4.3 基于残差的人工粘滞项（Residual-Based Artificial Viscosity Map）

![image-20220717114959295](pics\residual_based_av1.png)

![image-20220717115039087](pics\residual_based_av2.png)

![image-20220717115130114](pics\residual_based_av_figure.png)



### 参考

Physics-informed neural networks for high-speed flows
https://arxiv.org/abs/2105.09506

Discontinuity Computing with PINNs
https://arxiv.org/abs/2206.03864

Solving Hydrodynamic Shock-Tube Problems Using wPINNs with Domain Extension
http://dx.doi.org/10.13140/RG.2.2.29724.00642/1

PINNs with Adaptive Localized Artificial Viscosity
https://arxiv.org/abs/2203.08802
