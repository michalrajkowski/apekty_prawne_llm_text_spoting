#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Syncing submodule URLs..."
git submodule sync --recursive

echo "Initializing and updating submodules..."
git submodule update --init --recursive

echo "Submodules are initialized."
