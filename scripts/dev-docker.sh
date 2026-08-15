#!/usr/bin/env bash
# ホストに .venv を作らずに pytest / ruff / Synthetic Benchmark を回す。
#
#   scripts/dev-docker.sh                       # pytest
#   scripts/dev-docker.sh python -m pytest tests/evaluation -q
#   scripts/dev-docker.sh ruff check
#   scripts/dev-docker.sh python scripts/generate_benchmark.py --cases 20
#
# 作成されるファイルはホストの実行ユーザー所有になる（--user）。
# data/ は tmpfs に逃がすので、リポジトリの data/ を汚さない。
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-matuge-change-dev}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE ..." >&2
  docker build -f Dockerfile.dev -t "$IMAGE" .
fi

# MediaPipe のモデルはイメージに焼いてあるので、ホストへ取り出してマウントする
# （リポジトリを /app にマウントすると、イメージ側の /app/models は隠れてしまう）。
if [ ! -s models/face_landmarker.task ]; then
  echo "extracting models/face_landmarker.task ..." >&2
  mkdir -p models
  docker run --rm --entrypoint cat "$IMAGE" /app/models/face_landmarker.task >models/face_landmarker.task
fi

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/app" \
  --tmpfs /app/data:exec,mode=1777 \
  --workdir /app \
  --env PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest_cache" \
  "$IMAGE" "${@:-python -m pytest}"
