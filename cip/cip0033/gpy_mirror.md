# CIP-0033 — GPy mirror locks

GPy `GPClassification` defaults: **RBF kernel**, **Bernoulli + Probit link**, **EP** inference (`GPy.inference.latent_function_inference.expectation_propagation.EP`). We mirror that decomposition, not IVM’s White softener.

## Site parameterisation (GPy `gaussianApproximation`)

Unnormalised Gaussian sites in **natural parameters**:

- \(\tilde\tau_i\) = `tau` (precision)
- \(\tilde\nu_i\) = `v` (precision × mean)

Cavity (fractional EP with damping \(\eta\); default \(\eta=1\)):

\[
\tau_{-i} = \Sigma_{ii}^{-1} - \eta\,\tilde\tau_i,\qquad
\nu_{-i} = \mu_i\Sigma_{ii}^{-1} - \eta\,\tilde\nu_i.
\]

## Bernoulli–probit `moments_match_ep` (GPy verbatim)

For label \(y\in\{0,1,-1\}\) with \(\mathrm{sign}=+1\) if \(y=1\) else \(-1\):

\[
z = \mathrm{sign}\,\frac{\nu_{-}}{\sqrt{\tau_{-}^2+\tau_{-}}},\quad
\frac{\phi}{\Phi}=\texttt{gradLnCumGaussian}(z),
\]

\[
\hat\mu = \frac{\nu_{-}}{\tau_{-}} + \mathrm{sign}\,\frac{\phi/\Phi}{\sqrt{\tau_{-}^2+\tau_{-}}},
\]

\[
\hat\sigma^2 = \frac{1}{\tau_{-}} - \frac{\phi/\Phi}{\tau_{-}^2+\tau_{-}}\,(z+\phi/\Phi).
\]

Site update (GPy `gaussianApproximation._update_i`, damping \(\delta\)):

\[
\Delta\tilde\tau = \frac{\delta}{\eta}\Big(\hat\sigma^{-2}-\Sigma_{ii}^{-1}\Big),\qquad
\Delta\tilde\nu = \frac{\delta}{\eta}\Big(\hat\mu\hat\sigma^{-2}-\mu_i\Sigma_{ii}^{-1}\Big),
\]

with \(\tilde\tau\leftarrow\max(\tilde\tau,\varepsilon)\).

## Posterior recompute (GPy `posteriorParams._recompute`)

With \(\mathrm{S}^{1/2}=\mathrm{diag}(\sqrt{\tilde\tau})\):

\[
B = I + S^{1/2} K S^{1/2},\quad
L=\mathrm{chol}(B),\quad
V=L^{-1}(S^{1/2}K),\quad
\Sigma = K - V^\top V,
\]

\[
\alpha = \tilde\nu - S^{1/2} B^{-1}\big(S^{1/2}(K\tilde\nu + m_0)\big),\quad
\mu = K\alpha + m_0.
\]

## Locked product defaults (v1)

| Piece | GPy | GPc CIP-0033 |
|---|---|---|
| Likelihood | Bernoulli + Probit | Same moments via `ndlnoise::bernoulliProbitEpMoments`; labels \(\{0,1\}\) or \(\{\pm1\}\` |
| Inference | `EP()` | `CEpInference` (FTC first) |
| Kernel default (demos) | RBF | RBF; **no auto White** |
| Sparse EP | `EPDTC` exists | **Out of v1** (FTC-first Accept lock) |
| Laplace | separate inference class | Phase C on same likelihood primitives |

## Not mirrored (on purpose)

- `paramz` / mutable attribute tree
- `Y_metadata` side channels
- Storing sites on the likelihood object
