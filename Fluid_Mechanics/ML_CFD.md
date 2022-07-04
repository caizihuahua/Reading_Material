RANS
$$
\begin{aligned}
& \frac{\partial u}{\partial t}+(u\cdot\nabla)u=-\nabla p+\frac{1}{Re}\nabla^2u \\
& \nabla\cdot u=0
\end{aligned}
$$
%%%%
$$
\overline{U}\frac{\partial \overline{U}}{\partial x}+\overline{V}\frac{\partial \overline{U}}{\partial y}+\overline{W}\frac{\partial \overline{U}}{\partial z}+\frac{\partial\overline{u'u'}}{\partial x}+\frac{\partial\overline{u'v'}}{\partial y}+\frac{\partial\overline{u'w'}}{\partial z} \\
=-\nabla{\overline{p}}+\frac{1}{Re}\nabla^2\overline{U}
$$
Reynolds averaged turbulence modeling using deep neural network with embedded invatiance

Subgrid modeling for two-dimensional turbulence using neural networks
$$
u(x,t)\approx\overline{u}+\sum_{k=1}^r\varphi_k(x)a_k(t)
$$




## Wall Function and Wall Modeling

---

### 1. What are Wall Function and How do they work

> What are wall functions and why are they needed

- Gradient of velocity, temperature etc. close to the wall are large (no slipping condition)
- A fine grid resolution is required to accurately resolve the gradient
- Thin cells result in high aspect ratios, poor cell quality and high total cell count
- Can we model the variation between the cell centroid and the wall with non-linear function (rather than a piecewise-linear one)
- Then the wall adjacent cell does not have to be as thin
- Experimental measurements and DNS simulations show the variation close to the wall, which can be used as a model ($y^+$ and $U^+$ plot)
- Wall functions are the empirical functions that are fitted to the observed behaviour close to the wall
  - viscous sub-layer
  - buffer layer
  - log-law region
  - ![image-20220629092339432](./ML_CFD_pics\wall_functions.jpg)


> What are the standard wall functions

-  The standard wall functions are

$$
U^+=y^+, \quad y^+\lt5 \\
U^+=\frac{1}{\kappa}\log{Ey^+}, \quad 30\lt y^+\lt 200
$$

- $\kappa$ and $E$ are empirical coefficients (0.4187 and 9.793 respectively)
- y^+ and U^+ are dimensionless velocity and wall normal distance

$$
y^+=\frac{yu_\tau}{\nu}, \qquad U^+=\frac{U}{u_\tau}
$$

- $u_\tau$ is a reference velocity based on the wall shear stress (as the velocity is zero at the wall)

$$
u_\tau=\sqrt{\tau_\omega / \rho}
$$

- the buffer region
  - the wall function intersect at $y^+=11.25$
  - so choose 11.25 to be the 

> What are automatic wall functions (for buffer layer)

- An alternative would to be fit a single, smooth function through the entire range of $y^+$
- One example function is Spalding's wall function:

$$
y^+=U^++0.1108[e^{0.4U^+}-1-0.4U^+-\frac{1}{2}(0.4U^+)^2-\frac{1}{6}(0.4U^+)^3]
$$

- This function is smooth, continuous and valid for $y^+<300$, and give a good fit in the buffer region
- In practice, most CFD codes use a proprietary blending between the viscous sub-layer and log-law region
- The CFD wall treatment is often called automatic, as the user does not have to explicitly specify the wall function or the value of $y^+$ ahead of time.
- The CFD code evaluates $y^+$ and then blends/selects the appropriate wall function on the fly, without any user input

> What do CFD codes actually do

- At the wall, the velocity is zero (the no-slip condition) and the velocity at the cell centroid ($U_p$) is calculated from the momentum equations

- We need to know the gradient and the wall-shear stress

  - If the velocity variation across the cell is linear, the wall shear stress is
  
  - $$
    \tau_\omega=\nu(\frac{\partial U}{\partial y})_{y=0}=\nu(\frac{U_p-0}{y_p})=\nu\frac{U_p}{y_p}
    $$
  
  - if the variation across the cell is non-linear (wall function approach), then the wall shear stress is
  
  - $$
    \tau_\omega=\nu(\frac{\partial U}{\partial y})_{y=0}=\frac{u_t U_p}{\frac{1}{\kappa}\log(Ey^+)}
    $$
  
  - use the linear style in the non-linear region, just replace the $\nu_\omega=\frac{y_p u_t}{\frac{1}{\kappa}\log(Ey^+)}$ in the log-law region
  
  - $$
    \tau_\omega=\nu_\omega\frac{U_p}{y_p}
    $$
  
- $\nu_\omega=\nu+\nu_t$, laminar and turbulent component

  - $$
    \nu_t=\left\{
    \begin{aligned}
    
    &0 & y^+\lt11.25 \\
    &\nu(\frac{y^+}{\frac{1}{\kappa}\log(Ey^+)}-1)& y^+\gt11.25
    
    \end{aligned}
    \right.
    $$

  - near the wall is the laminar


> What value of $y^+$ should I choose for my simulations



### 2. What are Thermal (Temperature) Wall Function

> What are thermal (temperature) wall functions and why are they needed?

- Near the wall gradients of velocity and temperature are large
- The near wall gradient determine the shear stress and heat transfer
  - ![image-20220629112355962](ML_CFD_pics\thermal_wall_function.jpg)
- The wall may be hotter or colder than the freestream flow
  - ![image-20220629112804567](ML_CFD_pics\cold_hot_wall.jpg)

> What empirical functions can we fit to the data

- The viscous sub-layer and log-law empirical fits for temperature are

  - $$
    T^*=Pr y^* \quad y^*\lt 5 \\
    T^* = Pr_t(\frac{1}{\kappa}\log(Ey^*)+P) \quad 30\lt y^*\lt 200
    $$

- $Pr$ is the molecular Prandtl number, $Pr_t$ is the turbulent Prandtl number and $P$ is an additional function

- $y^*$ and $T^*$ are dimensionless wall normal distance and temperature respectively

  - $$
    y^*=\frac{\rho y u_\tau}{\mu} \\
    T^*=\frac{(T_\omega-T)\rho c_pu_\tau}{q_\omega} \\
    u_t = C_\mu^{1/4}\sqrt{k}
    $$

  - $k$ mean the turbulence kinetic energy

> What effect does the Prandtl number have on the empirical functions

- The molecular Prandtl number $Pr=\nu/\alpha$ is the ratio of momentum and thermal diffusivities
- The turbulent Prandtl number is treated as a constant (0.85)
- The molecular Prandtl number determines the shape and thickness of the thermal boundary layer
  - Air $Pr=0.71$ ; Water $Pr=5.68$

> What about the function $P$

- $P$ is another empirical function

- The most common form is the function given by Jayatilleke(1969)

  - $$
    P=9.24[(\frac{Pr}{Pr_t})^{3/4}-1][1+0.28e^{-0.007(Pr/Pr_t)}]
    $$

- 

> How are thermal (temperature) wall functions different to momentum (velocity) wall function

- For the momentum equations, the near wall viscosity is given by

  - $$
    \nu_\omega=\left\{
    \begin{aligned}
    &\nu &y^*\lt11.25 \\
    &\frac{u_\tau y_p}{\frac{1}{\kappa}\log(Ey^*)} &y^*\gt11.25
    \end{aligned}
    \right.
    $$

- The wall shear stress is then given by:

  - $$
    \tau_\omega=\rho\nu(\frac{\partial U}{\partial y})_{y=0}=\rho\nu_\omega\frac{U_p}{y_p}
    $$

  - 

> How are thermal (temperature) wall functions applied and what  do CFD codes actually do?
