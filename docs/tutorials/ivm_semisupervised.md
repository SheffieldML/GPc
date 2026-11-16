# IVM semi-supervised crescent (null-category noise)

**GPy counterpart:** no direct EP notebook; GPc feature from Lawrence & Jordan null-category noise (`ncnm`)  
**Data:** `examples/semisupercrescent.svml` (and labelled-only `semisupercrescent_labelled.svml`)

## Build

```sh
make ivm
```

## CLI tutorial

```sh
./ivm learn -a 100 -k rbf examples/semisupercrescent.svml semisupercrescent.model
./ivm display semisupercrescent.model
./ivm gnuplot examples/semisupercrescent.svml semisupercrescent.model semisupercrescent
gnuplot semisupercrescent_plot.gp
```

Compare with labelled-only probit training using `examples/semisupercrescent_labelled.svml` if you want the paper-style contrast (see root README figures).

## Notes

- Noise type in the model display should be Ncnm (null category), with missing-label probabilities for each class.
