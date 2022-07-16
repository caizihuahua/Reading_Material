## 使用神经网络捕捉激波

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

我们还需要一个方程来封闭这个问题，比如说 理想状态的气体方程。但是这里我们并不考虑温度和热量问题。
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
