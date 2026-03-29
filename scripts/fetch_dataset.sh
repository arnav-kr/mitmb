#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/nuscenes"
ARCHIVE="${DATA_DIR}/v1.0-mini.tgz"
URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

mkdir -p "${DATA_DIR}"

echo "[setup] Downloading nuScenes mini archive to ${ARCHIVE}"
if [[ ! -f "${ARCHIVE}" ]]; then
  curl -L "${URL}" -o "${ARCHIVE}"
else
  echo "[setup] Archive already exists, skipping download"
fi

echo "[setup] Extracting archive"
tar -xzf "${ARCHIVE}" -C "${DATA_DIR}"

echo "[setup] Done. Expected structure:"
find "${DATA_DIR}" -maxdepth 2 -type d | sort
