# Maintainer tools

Diagnostic drivers (`diag*`) and fixture stampers (`genTest*`) used during
sparse-GP / inducing work. Build targets keep the historical binary names at
the repository root:

```sh
make USE_MATIO=1 diagDtcvar && ./diagDtcvar
make USE_MATIO=1 genTestGpDtcvar && ./genTestGpDtcvar
```

These programs are not part of `make run-matio-tests`.
