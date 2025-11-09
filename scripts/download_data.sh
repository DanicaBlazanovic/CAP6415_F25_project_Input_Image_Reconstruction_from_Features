#!/usr/bin/env bash
set -euo pipefail

# Example template: download & unpack your dataset here.
# Replace the placeholder URL(s) with your actual source(s).

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo ">> Downloading dataset (edit this script with real links)..."
# curl -L -o "$DATA_DIR/dataset.zip" "https://example.com/path/to/dataset.zip"
# unzip -q "$DATA_DIR/dataset.zip" -d "$DATA_DIR"
# rm "$DATA_DIR/dataset.zip"

echo ">> Done. Place any manual files under: $DATA_DIR"
