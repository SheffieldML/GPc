# Legacy platform make fragments

Historical `make.*` files kept for archaeology. Blessed contemporary paths are
still at the repository root:

- `make.linux` — Ubuntu / GitHub Actions default
- `make.osx` — Homebrew macOS

Copy one of those to `make.inc` (gitignored) for a local override. Do not
expect CI to exercise anything in this directory.
