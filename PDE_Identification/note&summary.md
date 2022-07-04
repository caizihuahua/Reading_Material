

the compact parameterized form of a differential operator


> data dimension
> identifiability
> CaSLR

1. how the approximate dimension(richness) of the data space spanned by all snapshots along a solution trajectory depends on the differential operator and initial data.
2. identifiability of a differential operator from its solution data.
3. propose a Consistent and Sparse Local Regression(CaSLR) method, which enforces global consistency and involves as few terms as possible from the dictionary using local measurement data from a single solution trajectory

---

PDEs of various types are for test by means of numerical analysis

PDE learning
- directly approximate the operator, diffenrential operator approximation(DOA). chose a finite dimensional space, such as on a discretized meshes on physical domain, ro transformed space(Fourier/Galerkin space), doesnt explicitylt reconstruct the differential operator, no help with understanding the underlying physics or laws.
  - need a large degree of freedoms to present mapping

- differential operator identification(DOI), using a combination of candidates from a dictionaty of basic diffretial operators and their functions(also called dictionaty, based on prior knowledge)

above, the problem is formulated as a regression problem (sparsity) from a prescribed dictionary

trajectory and less observed data

1. how underlying PDE operator and initial data affect the data space(the dimension of the space spanned by all snapshots of a solution trajectory)  (important to DOA approaches)

2. PDE identification problem using a single trajectory(indentifiability)
3. Consistent and Sparse Local Rgression(CaSLR) 
   1. globaly consistent
   2. built as few as terms as possible from the dictinary
   3. a good local fit to data using different linear combinations at different locations


---

### 2 Data space spanned by solution trajectory

how large is the data space spanned by all snapshots along a solution trajectory

characterize the information content of a solution trajectory by estimating the least dimentsion of a linear space (in  L2) which all snapshots u(,t) are $\epsilon$ close to in L2 norm

in Theorem 2.8 that for general elliotic operator for $L$ , all snapshots of ant single trajectory u(x,t) stays $\epsilon$ close to linear space od dimension at most of order $O(|log\epsilon|^2)$

if $L$ is a first order hyoerbolic operator, stays close to linear space of dimmension $O(\varepsilon ^{-\gamma})$

two possible challenges for a DOA approach in practice
1. limitied data space to train the approximation
2. a large number of parameters and a large amount of data as well as an expensive training process

---

### 3 PDE identification from a single solution trajectory

based on a combination of candidates from a dictionary

identifiability and stability

when many snapshots along a single trajectory are used as the test functions, the linear system has a fast decay and hence is ill-conditioned, which affect both the accuracy and teh stability of the identification problem

> PDE identification with constant coefficient

Fourier modes in the initial data and one can observe a single trajectory
at two different instants, one can recover those constant coefficients.

be identified at two different instants if and only if the solution contains enough Fourier modes

we donot know what type of PDE or initial data apriori, so the sampling strategy and PDE identification method should be data-driven and data-adapted

!! there exits a local patch that can identify PDE with constant coefficients by local regression if the solution has enough Fourier modes

why in Fourier mode, if the local sensor is not close enough to each other the how we can compute Ux ,Uxx.

so for PDE with constant coefficient, it is not necessary to collect data from at all points on a rectangular grid in space and time.

if sensitive to noise(zero in a nerghborhood)

> PDE identification with variable coefficients using a single trajectory

to identify a consistent differential operator that is built from as few as terms as possible from the library that can fit observed data well locally by using different linear combinations at different locations.

mabye these work is to identify the type of the underlying PDE, and once the type of PDE is determined, more accurate estimation of coefficient can be achieved by independent local regression or appropriate regularization(DOI plus regularization)

> identifiability with a single trajectory

if the initial condition is randomly generated the $S$ is almost surely non-singular if the parameters $p_\alpha$ are sufficiently soomth

> possible instability for identification of elliptic operator

!!! first we show local instability when one has short observation time for a single trajectory


the high frequency component of the perturbation will have a limited impact on the solution. in particular, both the order  of the differential operator and the smoothness of the initial data affect the instability estimate

### Local regression and the global consistency enfored PDE identification method

!!! a single trajectory (corresponding to unknown and uncontrollable initial data!!!)

enforce consistensy and sparsity!

local regression maybe means pick 1% rows from library matrix ??

> prior
 
measured/computed solution and its derivatives are available at certain locations

> identification guarantee by local regression

from operator with constant coefficients to variable coefficients
1. the variable coefficients are bounded away from zero and vary slowly on the patch
2. the solution data contain diverse information content on the patch

local regression based on identification problem is to find what terms in the PDE operator those nonzero coefficients correspond to

$L$ is a linear differential operator and the solution has sufficiently many Fourier modes, then we can always find a local patch $B$ such thar $\chi _B^p\gt0$, then we can identify the PDE accurately

A linear differential operator L with constant coefficients can be identified exactly by local regression, if the solution u contains sufficiently many Fourier modes.

In general, the larger $K$ is, the smaller $KpB$ will be. In other words, the larger the dictionary, the harder the identification for the underlying PDE

> data drven and data adaptive measurement selection

using solution data indiscriminatly for PDE identification may not be a good strategy.

to improve CaSLR robustness and accuracy, we propose the following process to select patches containing reliable accurate information

```
Dear Professor Xiang:

I have finished reading the article you sent me, and following are my opinions and summaries, although I still don't quite understand the proof process in the article.

An important idea in differential operator identification (DOI) is to solve a euqation Ax=b. And the most important steps are:
1) get the solution data of underlying system from sensors
2) determine the candidate functions in the dictionary
3) conduct a sparse regression to find the coefficient of the underlying operator(i.e. Lasson Regression or Sequential Threshold Regression)

Above is the general method for identify the underlyting operator. However, it is costly or not practical to collect the data from all points in the system. And the more data you use, the more likely the matrix A becomes ill-conditioned,i.e.,u_x(numerically computed derivatives).
So the method is not always the best way for all kinds of the differential operators.

Take the differential operator with constant coefficient as an example. If you conduct Fourier transform for u_t=-L(u) where L is a time-independent linear differential operator, then there is no need to compute the derivatives of u_x,u_xx.
So you can just use data from local sensors in one snapshots along a single trajectory to form the matrix A. And then you get the coefficients.
That's to say, for each different differential operator, there is a different optimal solution to get the coefficients.

The focus of this article is to figure out how many local sensors we should put for different differential operators and where to put them in the uderlying system,
and how we can find the coefficients of the uderlying operator from the data collected from the local sensors. 

These questions are correspond to the different chapters respectively, how big the data space is, how to select good patches (where to collect data) in the system, prospose a new method (CaSLR).

The chapter 2 of the article give us an relationship between the dimension of the data space spanned by a solution trajectoty and the error estimation.
And in the "Numerical Examples" section, they use the singular value of the matrix u(x_j,t_k) to represent the dimention of the data space to corroborate our analysis.

The chapter 3 show us the identifiablility and the instability of the different differential operator identification according to the data collected by chapter 2.

The chapter 4, show us a new method to conduct the local regression, the Consistent and Sparse Local Regression(CaSLR). This method uses a new constraint to improve global stability and promote sparsity of the coefficients.
And then author gives us a analysis result that the CaSLR works when the data meet certain common requirments.
And author also gives us a way to select the place (local patches) to put our sensors, in orther words, where to collect our data we need. But,in my oponion the right place is hard to know in advance.
Finally give us some numerical experiments to show the identifiablility of the CaSLR and the effects of patches selector, which performs really good in the test.

In my opinion this article is concerned about the fact that if there is not enough local sensors in the system, how can we figure out the time-independent underlying operator, or even further, is there a possibility that we can find the uderlying operator.

For me, because of the terms, i.e., "dictionary","local sensor", and also my poor PDE knowledge, i made little progress in reading this article for the first few days.
But when I figured out the exact meaning of these terms, I understood the main steps and processes of PDE identification, and I also understood the main difficulty to solve the euqation Ax=b, and I finally understood the research focus and motivation of this article.
This is a time-consuming process, but I was glad that I got the gist of this article, I think.

So to better understanding of the articles you will send me later, I plan to start reading a PDE book written by Evans to increase my knowledge for PDE and go over the theory of matrix decomposition this weekend.

Looking forward to your advice, and wish you a happy May Day.

best wishes,
Meng Li
```









