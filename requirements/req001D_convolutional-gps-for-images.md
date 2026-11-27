---
id: "001D"
title: "GP models can use convolutional covariance functions on image inputs"
status: "Proposed"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
related_tenets: ["assert-and-verify", "compose-through-interfaces", "flexibility-with-performance", "lapack-owns-linear-algebra"]
stakeholders: ["maintainers", "modellers", "GPy users", "teachers"]
tags: ["convolutional", "kernels", "inducing", "images", "deep-gp", "svgp"]
---

# REQ-001D: GP models can use convolutional covariance functions on image inputs

## Description

Users can fit sparse and deep GP models to images with a convolutional covariance function: a weighted sum, over image patches, of a covariance function on patches (van der Wilk, Rasmussen and Hensman, 2017). Inducing variables live in patch space, either as inducing patches or as weight-space windows over filters, so that training costs do not depend on the number of training images.

**Why this matters.** On MNIST, GPc's deep GP with an isotropic covariance on raw pixels reaches 2.2% test error against 0.99% for a small convolutional network; the gap is mostly architecture (translation structure), not inference. A convolutional covariance gives GP models that structure and makes the comparison with convolutional networks like for like. With an MLP covariance on patches, the weight-space construction of CIP-0028 gives an infinitely wide convolutional layer whose inducing variables are smoothed filters.

**Who benefits.** Modellers with image or other grid-structured inputs; GPy and GPflow users who expect convolutional kernels; the deep GP paper's experiments.

## Acceptance Criteria

- [ ] A convolutional covariance function on flattened images with configurable image shape, patch shape and stride, learned patch weights, and any existing covariance function on patches
- [ ] Inducing patches as an inducing family, and weight-space (ridge and Hermite) inducing variables when the patch covariance is an MLP-family kernel
- [ ] With one patch covering the image and unit weight, every quantity equals the patch covariance's own; gradients pass finite-difference checks
- [ ] Usable as the first layer of `SparseGPRegression`, `DeepGP` and the classification models through the Python façade
- [ ] Minibatch step cost independent of the number of training images; the cost of the patch-pair diagonal is stated and can be reduced (stride, or an unbiased estimator where the bound allows it)
- [ ] Non-goals are explicit: image-shaped hidden layers (deep convolutional GPs) and translation-insensitive variants are follow-on phases; no GPU backend

## Notes (Optional)

HOW is [CIP-0040](../cip/cip0040.md), which follows [CIP-003F](../cip/cip003F.md) (bulk weight-space ridge path) and [CIP-0041](../cip/cip0041.md) (per-layer façade).

## References

- **Related Tenets**: assert-and-verify, compose-through-interfaces, flexibility-with-performance, lapack-owns-linear-algebra
- **External Links**: van der Wilk, Rasmussen & Hensman (2017), *Convolutional Gaussian Processes*; Dutordoir, van der Wilk, Artemev & Hensman (2020), *Bayesian Image Classification with Deep Convolutional Gaussian Processes*; Blomqvist, Kaski & Heinonen (2019), *Deep Convolutional Gaussian Processes*

## Progress Updates

### 2026-11-27
Requirement drafted as Proposed alongside CIP-0040.
