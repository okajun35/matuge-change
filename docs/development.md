# 開発環境と品質チェック

## 推奨環境: Docker / WSL

WindowsのWSL2またはDocker Desktopでは、Python環境やMediaPipeモデルをホストへ用意せずに
起動できる。

```bash
git clone https://github.com/okajun35/matuge-change.git
cd matuge-change
docker compose up --build
```

ブラウザで <http://localhost:8000> を開く。初回ビルドでは依存関係のダウンロードに数分掛かり、
起動後もMediaPipeとnumbaの読み込みに30〜60秒掛かる場合がある。`docker compose ps` が
`healthy` になれば準備完了である。

- 抽出結果とカタログはホスト側の `./data` に残る
- 停止: `docker compose down`
- コード更新後の再ビルド: `docker compose up --build`
- Supabaseを使う場合は `.env` をcomposeが読み込む。未設定ならローカル実装で動く

テストやLintだけをDockerで実行する場合:

```bash
scripts/dev-docker.sh
scripts/dev-docker.sh ruff check
scripts/dev-docker.sh python -m pytest tests/evaluation -q
```

## ローカルPython環境

Python 3.11と[uv](https://docs.astral.sh/uv/)を使う。

```bash
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install -r requirements.txt
curl -sL -o models/face_landmarker.task --create-dirs \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

サーバーを起動する。

```bash
. .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

uvicornは自動リロードしない。ルート追加やpullの後はサーバーを再起動する。

## テストとLint

このリポジトリでは新機能とバグ修正をTDDで進める。期待する振る舞いを表す失敗テストを先に
追加し、最小実装で通した後にリファクタリングする。

```bash
uv pip install -r requirements-dev.txt
python -m pytest
ruff check
ruff format
pre-commit run --all-files
```

`.venv` にruffまたはpre-commitがない場合:

```bash
uvx ruff check
uvx pre-commit run --all-files
```

GitHub ActionsでもPRとmainへのpushに対して `ruff check`、`ruff format --check`、`pytest` を
実行する。ruffのバージョンは次の3箇所で揃える。

- `requirements-dev.txt`
- `.pre-commit-config.yaml` の `rev`
- `.github/workflows/ci.yml` の `version`

## Synthetic Benchmark

正解MaskとAlphaが既知の合成データで、productionの抽出・保持・再合成コードを測定できる。

```bash
python scripts/generate_benchmark.py --cases 100 --clean
python scripts/run_evaluation.py --output evaluation-results
```

Benchmarkの目的、指標、数値の引用ルールは [../evaluation/README.md](../evaluation/README.md)、
現行実装で見つかった問題は [benchmark-findings.md](benchmark-findings.md) を参照。

## 開発時に読む文書

- [../AGENTS.md](../AGENTS.md) — TDD、コマンド、設計上の不変条件
- [handover.md](handover.md) — 採用・却下した方式、既知の落とし穴、未検証事項
- [README.md](README.md) — 技術文書の索引
