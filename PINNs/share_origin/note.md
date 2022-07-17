## 使用神经网络捕捉激波



\begin{aligned}
& U(-1,t) = U(1,t), \quad \nabla U(-1,t) = \nabla U(1,t) \\
& U_0=(\rho_0,u_0,p_0)=(1.0+0.2\sin(\pi x),1.0,1.0)
\end{aligned}

所考虑的问题是流体力学中无粘可压缩的流动，该流动可用欧拉方程来描述：
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

我们还需要一个方程来封闭这个问题
$$
p = (\gamma-1)(\rho E-\frac{1}{2}\rho||u||^2), \quad \gamma=1.4
$$


> 内嵌物理知识神经网络(Physics-Informed Neural Networks,PINNs)

![image-20220716220854769](pics\PINNs_structure.png)

神经网络：喂入 $(x,t)$ ，吐出 $U_{nn}(x,t)$ 。（也就是能得到映射关系 $U_{nn}$

现在我们考虑两部分损失

1. 数据损失（$Loss_{data}$，uniformed loss）：所有已知点 $(x_i,t_j)$ 处，计算 $U_{nn}(x_i,t_j)$ 与已知数据 $U(x_i,t_j)$ 的损失
2.  PDE损失（$Loss_{pde}$，informed loss）：把 $U_{nn}$ 带入算子 $\partial_t+\nabla\cdot f$，计算采样点 $(x,t)$ 处的损失

$$
Loss = Loss_{data} + Loss_{pde}
$$

MSE：均方根误差（mean square errors）
$$
Loss_{pde} = MSE_{pde} = \frac{1}{N_{pde}}\sum_{j=1}^{N_{pde}}|\partial_tU_{nn}(x_{j}^{pde},t_j^{pde})+\nabla\cdot f(U_{nn}(x_{j}^{pde},t_j^{pde}))|^2
$$

### 1. 聚集采样（clustered sampling）

#### 正问题（forward problem, 已知初边值 IC/BC）

$$
Loss = MSE_{IC} + MSE_{BC} + MSE_{pde}
$$

例子：

控制方程：
$$
\partial_t \begin{pmatrix}
\rho \\
\rho u
\end{pmatrix}
+\nabla \cdot \begin{pmatrix}
\rho u \\
\rho u^2+p
\end{pmatrix}
= 0
$$
考虑空间为 $(0,1)$，边值 $BC$ 为 Dirichlet 边值条件
$$
U(0,t)=(1.4,0.1,1.0), \quad U(1,t)=(1.0,0.1,1.0)
$$
初值 $IC$：
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

![image-20220716232337696](pics\forward_problem_ex1.png)

误差估计
$$
Error_\rho := || \rho(x,2)-\rho_{nn}(x,2)||_{L^2}/ || \rho(x,2)||_{L^2}
$$
![image-20220716233426825](pics\forward_problem_ex1_error.png)

### 2. 增加额外的损失函数，全局调整损失函数权重

#### 逆问题（inverse problem, 不知道初边值，只知道某些数据）

这里我们考虑已知的数据为：某些点处的 $\nabla\rho(x,t)$ 和 $p(x^*,t)$
$$
Loss = MSE_{\nabla\rho}+MSE_{p^*}+MSE_{pde}
$$
![image-20220716234224455](pics\inverse_problem_MSE.png)

![image-20220717000227396](pics\inverse_problem_nabla_rho_loss.png)

![](pics\inverse_problem_sample_points.png)

其中 $x^*$ 是预先已知的。

可是通过这三个损失函数拟合出来的结果不太好。

比如对于如下解的问题
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
![image-20220717083224100](pics\inverse_problem_momentum_loss.png)

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

![image-20220717091121451](pics\problem_setting.png)

![image-20220717090541956](pics\shock_tube_problem.png)

考虑的都是正问题（已知初边值），采样1000*1000，

The Euler equations are of hyperbolic type which means that it, in essence, propagates the initial condition through the domain.
$$
Loss = w_{ib}Loss_{IC/BC}+w_{pde}Loss_{pde}, \\
0\le Loss{IC/BC} \le Loss_{pde}
$$
$\omega_{ib}$ 是 $\omega_{pde}$ 的100倍，$\omega_{ib}=10$ 是 $\omega_{pde}=0.1$

其中我们记，$G(\theta):=Loss$，$G(\theta)_f:=Loss_{pde}$，$G(\theta)_{IC}:=Loss_{IC/BC}$

![image-20220717091522377](pics\loos_vs_weighted_loss.png)

![image-20220717092014377](pics\zoom_in_density.png)
