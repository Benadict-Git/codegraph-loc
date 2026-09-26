#!/usr/bin/env bash
# Upload the current package source as a private Kaggle dataset (benadictinfanta/cgloc-code).
set -euo pipefail
cd "$(dirname "$0")/.."
stage=kaggle/.code_dataset
rm -rf "$stage" && mkdir -p "$stage/src"
cp -r codegraph cgloc pyproject.toml README.md "$stage/src/"
find "$stage" -name __pycache__ -prune -exec rm -rf {} +
cat > "$stage/dataset-metadata.json" <<'EOF'
{"title": "cgloc-code", "id": "benadictinfanta/cgloc-code", "licenses": [{"name": "apache-2.0"}]}
EOF
if kaggle datasets status benadictinfanta/cgloc-code >/dev/null 2>&1; then
  kaggle datasets version -p "$stage" -m "${1:-update}" -r zip -q
else
  kaggle datasets create -p "$stage" -r zip -q
fi
