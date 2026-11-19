#!/usr/bin/env bash
# Retry apt update/install for flaky Ubuntu mirrors on GitHub Actions.
# Usage: ./scripts/ci_apt_install.sh <package> [package ...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <package> [package ...]" >&2
  exit 2
fi

for attempt in 1 2 3 4 5; do
  echo "::group::apt attempt ${attempt}/5"
  if sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends "$@"; then
    echo "::endgroup::"
    exit 0
  fi
  echo "::endgroup::"
  echo "apt attempt ${attempt} failed; cleaning and retrying..."
  sudo apt-get clean || true
  sleep $((attempt * 20))
done

echo "ERROR: apt install failed after 5 attempts: $*" >&2
exit 1
