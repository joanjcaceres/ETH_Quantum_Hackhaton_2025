# YQuantum Challenge Description (Alice & Bob)

# 1. Getting Started - Understanding Wigner Functions and State Reconstruction

Before coding, we strongly recommend reviewing the [Theory Background](https://github.com/schlegeldavid/yq25_alice-bob_challenge/blob/main/TheoryBackground.md) of Wigner tomography and the role of displaced parity measurements.

## Task A: Generate Wigner Functions

The goal of this task is to generate Wigner functions for different quantum states using [dynamiqs](https://www.dynamiqs.org/stable/),  a simulation library based on [JAX](https://docs.jax.dev/en/latest/quickstart.html), that allows computations using CPUs or GPUs.

### **🔧 What to Do**

Compute the following states and visualize the associated Wigner functions:

- **Fock states** $\ket{n}$
- **Coherent States** $\ket{\alpha}$
- **Cat states (2-cat and 3-cat states)**
- **Dissipative Cat State from a Two-Photon Exchange Hamiltonian**
    - For this, we assume a dissipator of the form

$$
\frac{d\rho}{dt} = \mathcal{L}(\rho) = -i\[H, \rho\] + \kappa_b\mathcal{D}\[\hat{b}\](\rho)
$$
        

$$
H = g_2^* {{}a^\dagger}^2 b + g_2 a^2 b^\dagger + \epsilon_b^*b + \epsilon_bb^\dagger
$$

Here $\mathcal{D}\[X\](\rho) = X\rho X^\dagger - \frac{1}{2}\rho X^\dagger X - \frac{1}{2}X^\dagger X \rho$ is the so-called *dissipator of the system.* Using `dynamiqs` simulate the time-evolution of this system with the following parameters:

$$
g_2 = 1.0, \epsilon_d = -4, \kappa_b = 10
$$

(For now, we pretend that the parameters are without dimensions). Use an initial state $\ket{\psi_0}$ in which both the buffer and the memory are in the vacuum. You can play with a different Hilbert-space truncation. Simulate the dynamics for a time $T = 4$. Plot the Wigner function of mode $a$ as a GIF. 

### **📦 Deliverables**

- Static Wigner function plots for each state.
- A **GIF animation** showing the evolution of the **dissipative cat state’s Wigner function**.

## **Task B: Density Matrix Reconstruction from Wigner Data**

In this task, you will **reconstruct the quantum state $\rho$** from the **measured Wigner function $W(x,p)$.** This is a concrete example of [**quantum state tomography**](https://en.wikipedia.org/wiki/Quantum_tomography), where a continuous-variable quantum state is inferred from a set of measurable quantities.

You already have access to simulated data: the Wigner function $W(x,p)$ sampled on a 2D grid. Your goal is to construct a fit procedure that transforms this phase-space data into an estimated density matrix $\tilde{\rho}$ that lives in Fock space.

### From Wigner to Probabilities

Before starting, make sure to review the accompanying theoretical notes, where we show that the Wigner function can be interpreted as the expectation value of a **displaced parity measurement**:

$$
W(\alpha) = \frac{2}{\pi}\text{Tr}[D(\alpha)PD^\dagger(\alpha)]
$$

This means that if we perform a measurement of the **parity operator** after displacing the state by $\alpha$, the expected outcome (which is ±1) corresponds to the Wigner function up to a scaling.

To turn this into a measurement probability, we define:

$$
E_\alpha = \frac{1}{2}(\mathbb{I} + D(\alpha)PD^\dagger(\alpha))
$$

Then, the probability of obtaining a +1 outcome from a displaced parity measurement at point $\alpha \in \mathbb{C}$ is:

$$
p_\alpha = \text{Tr}(E_\alpha\rho) = \frac{1}{2}(1+\frac{\pi}{2}W(\alpha))
$$

This provides the **mapping from Wigner data to probabilities**. This is crucial, as it allows us to formulate the reconstruction of $\rho$ as an optimization problem.

### **🔧 What to Do**

You need to implement a pipeline that:

1. **Takes as input** a Wigner function $W(x,p)$ measured on a grid $(x,p) \in \mathbb{R}^2$.
2.  **Choose a set of displacement points** $\alpha_k$

*Note*: Your measurement probabilities will depend on values $W(\alpha_k)$ and these may not be directly available on your original grid — you will need to address this yourself.

3. **Map Wigner values to measurement probabilities:**

Given the Wigner function at $\alpha_k$, compute:

$$
w_k = \frac{1}{2}(1+ \frac{\pi}{2}W(\alpha_k))
$$

These are the expected measurement outcomes for each displacement $\alpha_k$

4.  **Define the measurement operator $E_{\alpha_k}$:**

For each displacement $\alpha_k$ construct the observable:

$$
E_{\alpha_k} = \frac{1}{2}(\mathbb{I} + D(\alpha_k) P D^\dagger(\alpha_k))
$$

*💡 Notes and Tips: When constructing* $D(\alpha)$ *and* $P$ *use a **larger Fock-space dimension** and truncate the result back to the original size. This improves the numerical quality of the operator and avoids edge effects due to truncation.*

5.  **Fit the density matrix**

You now want to find a density matrix $\tilde{\rho}$ such that the predicted probabilities

$$
p_k = \text{Tr}({E_{\alpha_k}}\tilde{\rho})
$$

match the observed values $w_k$. This leads to the following least-squares optimization problem:

$$
\min_{\rho \in \mathcal{M}} \sum_k \left| \mathrm{Tr}(E_{\alpha_k} \rho) - w_k \right|^2
$$

Where $\mathcal{M}$ is the set of valid density matrices:

$$
\rho \succeq 0\text{  (positive semidefinite),          }\text{   Tr}(\rho) = 1
$$

This is a **convex optimization problem,** and you can solve it using your favorite optimization tool. 

6. **(Optional Task)  Explore Other Reconstruction Approaches:** While least-squares is simple and effective, it is not the only choice. You are encouraged to explore alternative reconstruction methods.

**How to Evaluate the Fit**

Once you have your fitted state $\tilde{\rho}$, you should evaluate how well it approximates the true $\rho$ using **fidelity:**

$$
F(\rho,\tilde{\rho}) = \left(\text{Tr} \sqrt{\sqrt{\rho}\tilde{\rho}\sqrt{\rho}}\right)^2
$$

Also compare the distribution of eigenvalues. What are other ways to correctness of the reconstruction? Explore.

Evaluate how well the reconstruction performs for different states, including **Fock states, coherent states, and cat states**.

### **📦 Deliverables**

- A **function** or **notebook** that demonstrates the reconstruction process from $W_\rho(x,p)$ to $\tilde{\rho}$.
- **Fidelity Tables or Plots** comparing $\rho$ vs $\tilde{\rho}$ for each state type (Fock, coherent, cats..).
- Compare $\rho$ and $\tilde{\rho}$ using **at least one additional method** beyond fidelity.

## **Task C: Robustness of the fit**

In practice, Wigner data obtained from experiments is never perfect, it often contains **measurement noise.** In some cases, the Wigner function may contain **many more phase-space points than your reconstruction algorithm can handle efficiently**, either due to computational limitations or because your fit function assumes a smaller number of displacement points.

In this task, you will explore how **robust** your reconstruction algorithm is by simulating noisy data, and you will also **test your method on real experimental data**.

### **🔧 What to Do**

1. **Gaussian Noise**
    - Add Gaussian noise $\eta(x,p) \sim \mathcal{N}(0,\sigma^2)$ 
     to $W_\rho(x,p)$
    - Reconstruct $\tilde{\rho}$  from $W_{noisy}$ and measure fidelity.
    - Evaluate the fidelity $F(\rho,\tilde{\rho})$ as a function of noise level $\sigma$.
    - Plot how performance degrades as noise amplitude grows.
2. **Test on Real Data**
    - Apply your reconstruction algorithm to a **provided real experimental Wigner dataset**.
    - Visualize the reconstructed state and compare it qualitatively to the expected features.
    - You will not have access to the ground truth $\rho$ but you can still analyze eigenvalue distributions, purity, or expected photon number.

*💡 Notes and Tips:  The real dataset may be oversampled, meaning it contains far more data points than necessary. → Tip: Interpolating or intelligently subsampling the Wigner grid may help your algorithm run faster without hurting fidelity.*

### **📦 Deliverables**

- Plots of fidelity $\text{F}(\rho,\tilde{\rho})$ vs. noise level $\sigma$ for different states.
- A comparison between reconstructions from simulated and real Wigner data.
- Optional: Additional evaluation metrics like purity, eigenvalue spectra, or expected photon number.

---

# **2. Advanced Exploration - Denoising**

Having gained experience generating Wigner functions and reconstructing density matrices  you are now equipped to address more realistic and *systematically challenging* issues: **affine distortions**, **complex noise**, and **time-consuming computational routines**. In this phase, you will **develop advanced denoising strategies** (including supervised machine learning) and **accelerate** your reconstruction algorithms, pushing quantum state tomography to larger system sizes and more demanding experimental conditions.

## **Task A: Correcting Affine Distortions and Background Noise**

Real-world Wigner function measurements often suffer from [**affine distortions**:](https://en.wikipedia.org/wiki/Affine_transformation)

$$
W_{measured}(x,p) = a W_{\rho}(x,p) + b + \text{noise}
$$

This affine transformation includes an **unknown scale** $a$, an **unknown offset $b$,** and some additive noise. Your objective is to **estimate** and **remove** these distortions, ensuring the “corrected” Wigner function is as close to the true $W_\rho(x,p)$ as possible. 

You are provided with a total of **16 noisy Wigner functions**, each representing a quantum state affected by affine distortions and/or noise. These are split into two categories:

- **8 noisy Wigner functions**: Use these to develop and test your denoising and affine correction methods.
- **8 noisy Wigner functions with the corresponding clean state $\rho$:** Use them to validate how well your method works, by comparing your output with the known clean Wigner functions. ⚠️ *Do not hardcode your solution based on these examples—each Wigner has different parameters and distortions, so solutions must be general.*
- **Experimental Wigner functions** from the previous task to give you a sense of how your method performs in a true lab setting.

### 🔧 What to Do

1. Estimate $a$ and $b$. 

To correct the affine distortion in your measured Wigner function:

- **Normalization Constraint:**

Remember that, a physical Wigner function satisfies:

$$
\int\int W(x,p)dxdp= 1
$$

- **Background Offset $b$**

In most physical states, at the edges of phase space $W(x,p) \rightarrow0$.  This should allow you to estimate the value of $b$.

Apply the correction:

$$
W_{\rm corrected}(x,p) = \frac{1}{a} \cdot (W_{\rm measured}(x,p) - b)
$$

2. Filter or De-Noise

Apply a **2D [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian_filter)** to reduce high-frequency noise in the Wigner distribution. This helps smooth out sharp, non-physical fluctuations caused by measurement error. There's no universally optimal choice for the filter width (standard deviation $\sigma$). Try different values and evaluate their effect on reconstruction fidelity. Small $\sigma$ retains detail but may leave noise; large $\sigma$ smooths aggressively but may wash out fine structure.

3. Validate and Benchmark:

Use **noisy Wigner functions with clean references** and apply your reconstruction to both the raw and the denoised version, then compare the results to the clean reference by computing the fidelities:

- $F_{\text{raw}} = F(\rho_{\text{ref}}, \tilde{\rho}_{\text{raw}})$
- $F_{\text{denoised}} = F(\rho_{\text{ref}}, \tilde{\rho}_{\text{denoised}})$

This gives a clear benchmark of improvement. You are encouraged to plot fidelity vs.  filter width $\sigma$ across different Wigner functions to better understand the robustness of your method.

Finally, test your denoiser on the **experimental Wigner data** from the previous task. These don’t have ground truth either, but they let you assess your method in real-world conditions.

### 📦 Deliverables

- Correction & Denoising Code: A script or notebook that performs:
    - Affine correction (estimation and removal of $a$ and $b$)
    - Gaussian filtering
- Benchmarking Module: A section that runs the full reconstruction pipeline on:
    - Raw noisy Wigner data
    - Corrected and/or denoised Wigner data
    - Then compares both against the clean reference using fidelity.
- Metrics & Plots
    - **Fidelity Comparison**: Table or plot showing $F_{\text{raw}}$ and $F_{\text{denoised}}$ for each test case.
    - **Fidelity vs. Noise**: Curves showing how performance degrades or improves with varying noise width $\sigma$, for different Wigners.
    - **Before vs. After Wigner Plots**: Visual side-by-side of $W_{\text{measured}}$  and
- Experimental Test Case: Apply your correction pipeline to the **experimental Wigner data** and include a short discussion or plot illustrating its effect (no fidelity comparison needed here).

## [Optional] Task B: Supervised Learning for Wigner Denoising

In many practical scenarios, conventional filtering methods (like Gaussian smoothing or affine correction) are insufficient to restore high-fidelity quantum states from heavily corrupted Wigner data. In this advanced task, you will **design, train, and evaluate** a **supervised machine learning model** to denoise Wigner functions, using synthetic noisy–clean data pairs. Your goal is to learn a mapping from noisy Wigner distributions to clean ones, ultimately improving density matrix reconstruction.

**For this Task you can ask for GPU access to accelerate your model training. Contact one of the Alice & Bob representatives for this.** 

### 🔧 What to Do

1. **Design Realistic Noise Models**

Create **multiple noise models** to simulate imperfections seen in experiments (investigate).

2. **Build a Supervised Dataset**
- **Generate Clean Wigners**: Simulate a variety of pure quantum states
- **Corrupt the Data**: Apply your noise models to generate paired samples

$$
\tilde{W}_i = \text{NoiseModel}(W_i)
$$

- **Split Your Data**: Create distinct **training**, **validation**, and **test** sets to assess generalization. Ensure that your test set includes Wigner functions of **new states not seen during training** — not just new noise samples on familiar ones.
3. **Train a Denoising Model**
- Choose a network architecture suited to 2D grid input:
- Use **supervised learning**: Train your model on pairs $(\tilde{W}, W)$ to minimize reconstruction loss, such as:

$$
\mathcal{L} = \sum_{x,p}|D(\tilde{W}(x,p)) - W(x,p)|^2
$$

Feel free to **experiment** with other losses—especially if your model fails to preserve **important quantum features** like Wigner negativity or interference fringes in cat states.

4. **Evaluate: Reconstruction and Fidelity**

After denoising a test sample:

- Apply your existing `fit` function to reconstruct the density matrix

$$
\tilde{\rho}_{\rm denoised} = \text{fit}(D(\tilde{W}))
$$

- Compare against the true state $\rho$ using fidelity:
    - Before denoising: $F_{\text{raw}} = F(\rho, \tilde{\rho}_{\text{raw}})$
    - After denoising: $F_{\text{denoised}} = F(\rho, \tilde{\rho}_{\text{denoised}})$
- Report $\Delta F$ (fidelity improvement) across your test set.
5. Test on Real Data

Once trained, apply your model to the **experimental Wigner functions** from the earlier tasks. These have no known ground truth, but visual inspection and smoother reconstructions can offer qualitative feedback.

### 📦 Deliverables

- **Code/Notebook**: Include the dataset generation, model definition, and training loop.
- **Plots & Metrics**:
    - Training/validation loss curves
    - Fidelity comparisons before vs. after denoising on the test set
    - Visual examples of noisy vs. denoised Wigner functions
    - Results from applying the denoiser to **real experimental data**

## [Optional] Task C: Accelerating the Reconstruction (“fit”) Function

If you’ve made it all the way here — congratulations! 

But there's one more challenge…

Your final (optional) task is to **speed up the reconstruction algorithm** so it can scale to **larger states, denser grids, or real-time applications.**

### 🔧 What to Do

- Investigate ways to accelerate your reconstruction algorithm.
- Explore techniques or tools that allow you to run the fit more efficiently (e.g., on larger grids or higher-dimensional states).
- Compare the performance (runtime and fidelity) of your optimized version against the original.

*💡 Notes and Tips: This is open-ended — how you speed things up is up to you. You can explore numerical tricks, code-level optimizations, or alternative solvers.*

**For this Task you can ask for GPU access to accelerate your model training. Contact one of the Alice & Bob representatives for this.** 

### 📦 Deliverables

- A faster version of your reconstruction function.
- A short comparison (e.g., a plot or table) showing speed-up and any trade-offs in reconstruction quality.

---
