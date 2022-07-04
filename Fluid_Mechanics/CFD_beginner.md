$$
\frac{\partial \rho}{\partial t}=\nabla\cdot(\rho\bold{u})=0 \\
\frac{\partial(\rho \bold{u})}{\partial t}+\bold{u}\cdot\nabla(\rho \bold{u})=\nabla\cdot\sigma+\rho\bold{f} \\
\frac{\partial E_t}{\partial t}+\nabla\cdot(E_t\bold{u})=\nabla\cdot\sigma\bold{u}-\nabla\cdot\bold{q}
$$

- **Thermal Conduction**: (Diffusion) The spread of heat across materials such as solids or fluids, from regions of high temperature to regions of lower temperatures.
- **Thermal Convection**: The transport of heat with the flow of a fluid. Fluid flow can be driven by external work (forced convection) or by buoyancy, which is the movement of fluid with varying density in the presence of gravity (natural convection).
- **Thermal Radiation**: The generation and absorption of heat through electromagnetic waves.
- **Phase Changes**: The release or absorption of heat through transitions such as boiling, melting, condensation, etc.

---

<img src="./ML_CFD_pics\book_scale.jpg" alt="image-20220701171747871" style="zoom:50%;" />

### mass conservation in three dimensions

<img src="./ML_CFD_pics\book_mass.jpg" alt="image-20220701172225602" style="zoom:50%;" />

取定正方向很重要
$$
\frac{\partial \rho}{\partial t}+\frac{\partial (\rho u)}{\partial x}+\frac{\partial (\rho v)}{\partial y}+\frac{\partial (\rho w)}{\partial z}=0 \\
\implies \frac{\partial \rho}{\partial t}+\text{div}(\rho \bold{u})=0 \\
\implies
\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \bold{u})=0
$$
the second term is called the **convective term**

### Rates of change following a fluid particle and for a fluid element

substantive derivative
$$
\frac{D\phi}{Dt} =\frac{\partial \phi}{\partial t}+\frac{\partial \phi}{\partial x}\frac{dx}{dt}+\frac{\partial \phi}{\partial y}\frac{dy}{dt}+\frac{\partial \phi}{\partial z}\frac{dz}{dt}\\ = \frac{\partial \phi}{\partial t}+\bold{u}\cdot\grad\phi
$$
$D\phi/Dt$ defines rate of change of property $\phi$ per unit mass

fluid particles (track the motion) **Lagrangian** approach

fluid elements (fixed in space) **Eulerian** approach

### 2.1.3 Momentum equation in three dimensions

$$
\frac{\partial \rho\phi}{\partial t}+\nabla\cdot(\rho\phi\bold{u}) = \rho\frac{D\phi}{Dt}
$$

x-momentum:
$$
\rho\frac{Du}{Dt}=\frac{\partial(-p+\tau_{xx})}{\partial x}+\frac{\partial\tau_{yx}}{\partial y}+\frac{\partial\tau_{zx}}{\partial z}+S_{Mx}
$$

### 2.1.4 Energy equation in three dimensions

**transport equation** for proper $\phi$
$$
\frac{\partial(\rho\phi)}{\partial t}+\text{div}(\rho\phi\bold{u})=\text{div}(\Gamma\,\text{grad}\phi)+S_{\phi}
$$
rate of increase of $\phi$ of fluid element + net rate of flow of $\phi$ out of fluid element **=** rate of increase of $\phi$ due to diffusion + rate of increase of $\phi$ due to source

**rate of change + convective = diffusive + source**

note (Gauss's divergence theorem)
$$
\int_V \nabla\cdot(\bold{f})dV = \int_S\bold{n}\cdot\bold{f}dS
$$
