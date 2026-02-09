#!/bin/bash
set -euo pipefail

DIST_FOLDER="${1:?dist folder missing}"
mkdir -p "$DIST_FOLDER"

# If wheel already present, do nothing
if [[ -f "$DIST_FOLDER/tensorflow.whl" ]]; then
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TF_REPO="${TF_WHEEL_REPO:-https://github.com/tensorflow/tensorflow.git}"
TF_REF="${TF_WHEEL_REF:-v2.18.0}"
TF_DIR="${TF_WHEEL_SRC_DIR:-$ROOT_DIR/target/tensorflow-src}"
BAZELISK="${TF_WHEEL_BAZELISK:-$HOME/bin/bazelisk}"
BAZELRC="${TF_WHEEL_BAZELRC:-$ROOT_DIR/.tf_configure.bazelrc}"

mkdir -p "$(dirname "$TF_DIR")"

[[ -x "$BAZELISK" ]] || { echo "bazelisk not found"; exit 1; }
[[ -f "$BAZELRC" ]] || { echo "bazelrc not found"; exit 1; }

if [[ ! -d "$TF_DIR/.git" ]]; then
  git clone "$TF_REPO" "$TF_DIR"
fi

cd "$TF_DIR"
git fetch --all --tags
git checkout -f "$TF_REF"

"$BAZELISK" --bazelrc="$BAZELRC" \
  build --noenable_bzlmod \
  //tensorflow/tools/pip_package:wheel \
  --repo_env=USE_PYWRAP_RULES=1 \
  --repo_env=WHEEL_NAME=tensorflow \
  --config=cuda --config=cuda_wheel

cp -f bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl \
  "$DIST_FOLDER/tensorflow.whl"
