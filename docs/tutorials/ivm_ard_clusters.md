# IVM with ARD (Gaussian clusters)

**GPy counterpart:** ARD / input-relevance demos (e.g. parts of `toy_ARD`)  
**Data:** `examples/ard_gaussian_clusters.svml`

## Build

```sh
make ivm
```

## CLI tutorial

```sh
./ivm learn -a 100 -k rbf -i 1 examples/ard_gaussian_clusters.svml ard_gaussian_clusters.model
./ivm display ard_gaussian_clusters.model
./ivm gnuplot examples/ard_gaussian_clusters.svml ard_gaussian_clusters.model ard_gaussian_clusters
gnuplot ard_gaussian_clusters_plot.gp
```

`-i 1` selects an ARD RBF so irrelevant input scales can shrink. Display the model to see per-dimension `inputScale` values.

## Notes

- Path façade naming for ARD leaves follows CIP-0014 (`inputScale_k`; not fake GPy `lengthscale_i` on RBF-ARD).
