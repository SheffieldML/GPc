# Legacy platform make fragments

Historical `make.*` files kept for archaeology. Blessed contemporary paths are
still at the repository root:

- `make.linux` — Ubuntu / GitHub Actions default
- `make.osx` — Homebrew macOS

Copy one of those to `make.inc` (gitignored) for a local override. Do not
expect CI to exercise anything in this directory.

Also archived here:

- `CMakeLists.txt.archive-2020` — pre-CIP-0020 stub that pointed at an external
  `mlprojects` tree and `ndlfortran.c`. Do not use. Living thin CMake (when
  present) is the root `CMakeLists.txt` from [CIP-0020](../../cip/cip0020.md).
