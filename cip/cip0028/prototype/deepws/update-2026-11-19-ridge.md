## Update 2026-11-19: numerical findings and revised direction

> Evidence from a NumPy prototype (scripts in `cip/cip0028/prototype/`; summary below). Three changes: (1) correct \(K_{fu}\) to include \(\sqrt{p(\mathbf w)}\); (2) isotropic weight-space windows fail with input dimension, so replace them with **ridge windows in prior-whitened weight space**; (3) record design notes for a later deep extension.

### 1. Correction: \(K_{fu}\) must include \(\sqrt{p(\mathbf w)}\)

In the white-noise construction \(f(\mathbf x)=\sigma\int\phi(\mathbf w,\mathbf x)\sqrt{p(\mathbf w)}\,\mathrm dW(\mathbf w)\) with \(p=\mathcal N(\mathbf 0,\Sigma)\), the window multiplies \(\sqrt p\), which is itself Gaussian, \(\propto\mathcal N(\mathbf 0,2\Sigma)\). For isotropic windows of width \(s\) centred at \(\hat{\mathbf w}_m\):

\[
P=\tfrac12\Sigma^{-1}+s^{-2}I,\quad C=P^{-1},\quad \boldsymbol\mu_m=C\hat{\mathbf w}_m/s^2,
\]
\[
K_{fu}(\mathbf x,m)=\sigma A_m\,\operatorname{erf}\!\Big(\frac{\boldsymbol\mu_m^\top\tilde{\mathbf x}}{\sqrt{1+2\tilde{\mathbf x}^\top C\tilde{\mathbf x}}}\Big),\quad
A_m=(\pi s^2)^{-\frac{D'}4}(2\pi)^{\frac{D'}4}|\Sigma|^{-\frac14}|P|^{-\frac12}\exp\!\big(\tfrac12\boldsymbol\mu_m^\top P\boldsymbol\mu_m-\tfrac{\|\hat{\mathbf w}_m\|^2}{2s^2}\big),
\]
\[
K_{uu}[m,n]=\exp\!\big(-\|\hat{\mathbf w}_m-\hat{\mathbf w}_n\|^2/(4s^2)\big).
\]

The formula in Phase 1 above omits \(\sqrt p\); then neither \(K_{uu}\) nor \(K_{fu}\) depends on `weightVariance` / `biasVariance` (apart from \(\sigma\)), and they cannot be learnt.

Checks: closed form vs 2-D quadrature, worst relative error \(6.5\times10^{-14}\) over 12 random settings; \(\partial K_{fu}/\partial\mathbf x\) vs finite differences \(10^{-11}\); at a fixed window \(K_{fu}(h{=}1)=0.204,\,0.259,\,0.217\) for weight variance \(0.25,1,4\) while \(K_{uu}=1\).

### 2. Problem: isotropic windows do not scale with dimension

\(\sqrt p\) and each window are both \(L^2\)-normalised Gaussians in \(D'=d+1\) dimensions; their overlap falls roughly like \(e^{-cD'}\). Prior variance captured, \(Q(\mathbf x,\mathbf x)/k(\mathbf x,\mathbf x)\), best over widths with \(\kappa(K_{uu})\le10^4\), random centres:

| input dim \(d\) | isotropic, \(M=64\) | isotropic, \(M=1024\) | ridge (§3), \(M=64\) | ridge (§3), \(M=1024\) |
|---|---|---|---|---|
| 4 | 0.49 | 0.73 | 0.96 | 0.90\* |
| 8 | 0.24 | 0.60 | 0.97 | 0.99 |
| 16 | 0.02 | 0.23 | 0.87 | 0.97 |
| 32 | 0.0001 | 0.002 | 0.49 | 0.96 |

\*At \(d=4\), 1024 units in a 5-dimensional weight space are near-redundant, so the condition budget binds; fewer units do better.

At \(d=16\), \(M=256\), with the exact layer's hyperparameters, isotropic windows capture 0–2% of the variance; reaching unit per-layer gain needs \(\sigma^2\) scaled by \(\times6.6\) (\(\kappa=1\)) to \(\times600\) (\(\kappa=0.5\)).

Even in low dimension there is an accuracy/conditioning trade-off. With 81 windows on a grid in \(D'=2\):

| \(s\)/spacing | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| variance captured | 0.896 | 0.952 | 0.968 | 0.973 | 0.971 |
| \(\kappa(K_{uu})\) | \(1.1\times10^2\) | \(4.8\times10^3\) | \(2.5\times10^5\) | \(1.2\times10^7\) | \(1.4\times10^{10}\) |

So a kernel-free \(K_{uu}\) is not automatically well conditioned: conditioning is governed by window overlap, which is now a design choice. The Phase 1 gate "\(\kappa(K_{uu})\) flat in `weightVariance`" holds by construction and is uninformative; replace it with a variance-captured gate at a fixed condition budget (§5).

### 3. Proposed direction: ridge windows in prior-whitened weight space

**Whitening.** Let \(\boldsymbol\omega=\Sigma^{-1/2}\mathbf w\) and \(\hat{\mathbf x}=\Sigma^{1/2}\tilde{\mathbf x}\), and define \(u_m=\int\psi_m(\Sigma^{-1/2}\mathbf w)\,|\Sigma|^{-1/4}\,\mathrm dW(\mathbf w)\). Then \(K_{uu}=\langle\psi_m,\psi_n\rangle_{L^2(\boldsymbol\omega)}\) is kernel-free, and **all kernel dependence moves into the input scaling** \(\hat{\mathbf x}\), so ARD (`CMlpardKern`) costs nothing extra. The Williams kernel in these coordinates is \(\sigma^2\frac2\pi\arcsin\big(2\hat{\mathbf x}^\top\hat{\mathbf x}'/\sqrt{(1+2\|\hat{\mathbf x}\|^2)(1+2\|\hat{\mathbf x}'\|^2)}\big)\).

**Ridge window.** Unit direction \(\mathbf v_m\), offset \(a_m\), width \(s\):
\[
\psi_m(\boldsymbol\omega)=g_m(\mathbf v_m^\top\boldsymbol\omega)\,\sqrt{\mathcal N(\Pi_m^\perp\boldsymbol\omega;\mathbf 0,I)},\qquad
g_m(t)=(\pi s^2)^{-1/4}e^{-(t-a_m)^2/(2s^2)}.
\]
Across \(\mathbf v_m\) the window *is* the square root of the whitened prior, so that factor overlaps \(\sqrt p\) exactly and nothing decays with dimension. Each unit is a direction plus an offset, i.e. a neuron, rather than a point in weight space.

**Closed forms.** With \(c_s=2s^2/(2+s^2)\) and \(\mu_m=a_m c_s/s^2\):
\[
K_{fu}(\mathbf x,m)=\sigma A_m\operatorname{erf}\!\Big(\frac{\mu_m\,\mathbf v_m^\top\hat{\mathbf x}}{\sqrt{1+2\big(\|\hat{\mathbf x}\|^2-(1-c_s)(\mathbf v_m^\top\hat{\mathbf x})^2\big)}}\Big),\quad
A_m=(2\pi)^{-\frac14}(\pi s^2)^{-\frac14}\sqrt{2\pi c_s}\,e^{\mu_m^2/(2c_s)-a_m^2/(2s^2)}.
\]
\(K_{uu}[m,n]\) is a Gaussian integral with precision \(I+\alpha(\mathbf v_m\mathbf v_m^\top+\mathbf v_n\mathbf v_n^\top)\), \(\alpha=s^{-2}-\tfrac12\). It reduces to \(2\times2\) algebra in the cosine \(c=\mathbf v_m^\top\mathbf v_n\) (Sylvester / Woodbury), so it is \(O(M^2)\) with no \(D'\) cost, and the diagonal is exactly 1.

Checks: \(K_{uu}\) vs 2-D quadrature \(7.6\times10^{-14}\); \(K_{fu}\) vs Monte Carlo (\(10^7\) samples, \(D'=9\)) \(-0.145041\) vs \(-0.145074\pm0.000125\). Variance captured: see §2 table.

**Parameters and conditioning.** Directions \(\mathbf v_m\) live on the sphere and are learnable (initialise random or in orthogonal blocks; optimise with a sphere retraction or by normalising an unconstrained vector); offsets \(a_m\) are learnable (initialise \(n_{\text{off}}\) per direction on \([-2.5,2.5]\)); \(s\) is fixed by default, as in CIP-001C. Conditioning is controlled by \(s\), \(n_{\text{off}}\), and how many directions share a neighbourhood. The configurations above stay at \(\kappa(K_{uu})\le10^4\) with \(n_{\text{off}}=4\), \(s\in[0.5,0.8]\). In very low \(D'\), large \(M\) means near-parallel directions; use fewer units there.

**Relation to prior work.** Sun, Shi & Grosse (2020) and Dutordoir et al. (2021) also treat hidden units as inter-domain inducing variables for dot-product kernels, via RKHS projections onto activation-shaped functions computed with spherical harmonics; their \(K_{uu}\) is a Gegenbauer series depending on the kernel spectrum, and they recover exact activations. This proposal keeps the units-as-directions view but uses the VIK white-noise construction (Álvarez et al. 2010): \(K_{uu}\) is kernel-free and in closed form, inputs stay in \(\mathbb R^d\), and the features are smoothed rather than exact activations.

**Arc-cosine (Phase 2).** The same ridge construction applies, because \(\mathbb E[\max(0,z)]\) for Gaussian \(z\) has a closed form, \(m\Phi(m/\sqrt v)+\sqrt v\,\varphi(m/\sqrt v)\). So smoothed-ReLU ridge features should also be closed form (to derive and verify).

### 4. Deep extension: design notes (still a non-goal of this CIP)

End-to-end Jacobian \(J_L\cdots J_1\), \(D=16\), all stacks tuned to per-layer \(\chi\approx1\), log singular values computed exactly (160-digit arithmetic; the QR/Lyapunov estimate was off by up to 5.6). Spread = s.d. of log singular values within the spectrum:

| stack | spread, \(L=8\) | \(L=32\) | \(L=64\) |
|---|---|---|---|
| GP prior, erf, Gaussian \(W\) | 5.0 | 17.3 | 33.7 |
| GP prior, erf, orthogonal \(W\) | 5.0 | 17.8 | 34.1 |
| GP prior, ReLU | 5.1 | 18.2 | 35.3 |
| GP prior, each layer rescaled to \(|\det J_l|=1\) | 5.0 | 17.3 | 33.7 |
| weight-space prior, isotropic, \(M=256\) | 5.3 | 18.3 | 36.1 |
| mean path, tied \(A_l\propto W_l^\top\), ReLU | 0.54 | 1.23 | 1.91 |
| mean path, tied, erf, \(q^*\approx0.03\) | 0.10 | 0.28 | 0.41 |
| mean path, tied, erf, \(q^*\approx0.25\) | 0.80 | 3.17 | 6.32 |
| residual GP, \(\chi_f=1\) | 0.97 | 2.07 | 2.99 |
| residual GP, \(\chi_f=1/64\) | 0.17 | 0.35 | 0.48 |
| residual weight-space, \(\chi_f=1/64\) | 0.15 | 0.29 | 0.40 |

Conclusions for a deep version:

1. **\(\det J=1\) is the wrong target.** It fixes the mean log singular value (volume) but leaves the spread unchanged; the problem is the whole spectrum (dynamical isometry; Pennington et al. 2017).
2. **Deep GP prior samples cannot be made isometric.** Independent Gaussian output functions make each layer Jacobian approximately a Gaussian random matrix, so the spread grows roughly linearly in depth (log-log slope 0.92) whatever the activation, hidden-weight orthogonality, or ordered/critical/chaotic setting; criticality only moves the mean. This is the Duvenaud et al. (2014) pathology; cf. the effective-depth analysis of Dunlop et al. (2018).
3. **Use identity mean functions with per-layer variance \(\propto1/L\)** (Salimbeni & Deisenroth 2017 use linear means; Duvenaud et al. connect inputs). The spread then grows like \(\sqrt L\) (slope 0.50) and stays below 0.5 at \(L=64\), including for weight-space layers.
4. **Isometry is achievable on the posterior-mean path** if the next layer's centres and this layer's output weights are aligned (the \(W_{l+1}B_l\) coupling of Dutordoir et al. 2021), with erf near the linear regime (small stationary scale \(q^*\)). Unlike square networks, ReLU here is not hopeless (1.9 at \(L=64\)): each layer averages over many hidden units through a \(D\)-dimensional bottleneck. Treat that as an observation, not a theorem.
5. Smoothing windows reduce the effective slope of the activation and push layers towards the ordered phase; criticality must be computed for the windowed layer, not taken from the exact kernel.

Caveats: prior samples and untrained mean paths only; exact layers approximated with 512 random features; \(D=16\); bisection noise about \(\pm3\%\) in \(\chi\).

### 5. Revised plan and gates

- **Phase 0:** unchanged (`domainDim` ownership; weight-space centres become *directions + offsets* for the ridge family).
- **Phase 1 (revised): ridge erf windows in whitened weight space** replace isotropic windows. Keep isotropic windows only as a \(d\le2\) test oracle, if at all. Gates:
  - closed forms vs quadrature (\(D'=2\)) and Monte Carlo (\(D'\ge8\));
  - finite-difference gradients for \(\mathbf v_m\) (tangent to the sphere), \(a_m\), \(s\), and kernel parameters through \(\hat{\mathbf x}\);
  - **variance-captured gate** at \(\kappa(K_{uu})\le10^4\): \(\ge0.80\) at \(d=16,M=64\) (measured 0.87) and \(\ge0.95\) at \(d=16,M=1024\) (measured 0.97);
  - DTCVAR \(M=N\)-style sanity check where applicable, and stream round-trip.
- **Phase 1b:** ARD falls out of \(\hat{\mathbf x}=\Sigma^{1/2}\tilde{\mathbf x}\); test against `CMlpardKern`.
- **Phase 2:** arc-cosine ridge features (derive \(\mathbb E[\mathrm{ReLU}]\) closed form; same gates).
- **Deep:** separate future CIP using §4; identity mean plus \(1/L\) variance as the default; isometry diagnostics (exact log singular values of \(J_L\cdots J_1\)) as a test utility.

### References (additions)

- S. Sun, J. Shi, R. Grosse (2020). *Neural Networks as Inter-Domain Inducing Points.* AABI.
- V. Dutordoir, N. Durrande, J. Hensman (2020). *Sparse Gaussian Processes with Spherical Harmonic Features.* ICML.
- L. C. Tiao, V. Dutordoir, V. Picheny (2023). *Spherical Inducing Features for Orthogonally-Decoupled Gaussian Processes.* ICML.
- D. Duvenaud, O. Rippel, R. Adams, Z. Ghahramani (2014). *Avoiding Pathologies in Very Deep Networks.* AISTATS.
- M. Dunlop, M. Girolami, A. Stuart, A. Teckentrup (2018). *How Deep Are Deep Gaussian Processes?* JMLR 19(54).
- J. Pennington, S. Schoenholz, S. Ganguli (2017). *Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry.* NeurIPS.
- H. Salimbeni, M. Deisenroth (2017). *Doubly Stochastic Variational Inference for Deep Gaussian Processes.* NeurIPS.
