# IVM classification on the unit square

**GPy counterpart:** `basic_classification` uses Bernoulli + EP. GPc’s documented substitute here is the **IVM** with a probit noise model.  
**Data:** `examples/unitsquaregp.svml` (2D inputs, binary labels)

## Build

```sh
make ivm
```

## CLI tutorial

```sh
./ivm -v 3 learn -a 200 -k rbf examples/unitsquaregp.svml unitsquaregp.model
./ivm display unitsquaregp.model
./ivm gnuplot examples/unitsquaregp.svml unitsquaregp.model unitsquaregp
gnuplot unitsquaregp_plot.gp
# or: gnuplot -e "GPLOT_PNG='unitsquaregp_plot.png'" unitsquaregp_plot.gp
```

`-a 200` sets the active-set size. Default compound kernel is RBF + bias + white.

## Notes

- This is not EP inference inside `CGp`. For teaching, treat IVM+probit as the GPc classification path until a future CIP adds EP/Laplace.
- Active points (dark red in gnuplot) typically lie near the decision boundary.
