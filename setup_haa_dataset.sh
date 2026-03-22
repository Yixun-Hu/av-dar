#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_haa_dataset.sh [DATA_ROOT]
#
# Example:
#   bash setup_haa_dataset.sh ./data
#
# This script downloads only the Hearing Anything Anywhere (HAA / DIFFRIR)
# dataset from Zenodo and extracts it.

ROOT_ARG="${1:-./data}"
DATA_ROOT="$(python3 - <<'PY' "$ROOT_ARG"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

HAA_ROOT="$DATA_ROOT/HAA"
mkdir -p "$HAA_ROOT"

echo "[1/2] Downloading HAA (DIFFRIR) from Zenodo into: $HAA_ROOT"
HAA_FILES=(
  classroomBase.zip
  complexBase.zip
  complexRotation.zip
  complexTranslation.zip
  dampenedBase.zip
  dampenedPanel.zip
  dampenedRotation.zip
  dampenedTranslation.zip
  hallwayBase.zip
  hallwayPanel1.zip
  hallwayPanel2.zip
  hallwayPanel3.zip
  hallwayRotation.zip
  hallwayTranslation.zip
  mic_calibrations.zip
)

for f in "${HAA_FILES[@]}"; do
  url="https://zenodo.org/records/11195833/files/${f}?download=1"
  echo "  - $f"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$HAA_ROOT/$f" "$url"
done

echo "[2/2] Extracting HAA zips into per-scene folders"
for z in "$HAA_ROOT"/*.zip; do
  base="$(basename "$z" .zip)"
  mkdir -p "$HAA_ROOT/$base"
  unzip -o "$z" -d "$HAA_ROOT/$base"
done

echo "Done."
echo "HAA dataset is under: $HAA_ROOT"