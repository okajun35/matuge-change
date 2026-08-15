# AGENTS.md — 開発エージェント向けメモリ

このリポジトリで作業するAIエージェント・開発者が毎回最初に読むべきルール。

**セッションを引き継いだ場合はまず `docs/handover.md` を読む** — これまでに採用/却下した方式とその理由、既知の落とし穴、未検証事項がまとまっている。

## 開発方針: TDD（テスト駆動開発）

新機能・バグ修正は必ずTDDで進める。

1. **Red**: 期待する振る舞いを表す失敗するテストを先に書く（`tests/` に pytest）
2. **Green**: テストを通す最小限の実装を書く
3. **Refactor**: テストが通る状態を保ったままリファクタリングする

- テストを先に書かずに実装を書かない
- 実装に合わせてテストを後から緩めない（仕様が変わった場合のみテストを変更）
- 顔検出など重い依存はfixture（`tests/conftest.py`）や合成データで代替してよい

## コマンド

```bash
. .venv/bin/activate
python -m pytest          # テスト実行
ruff check                # Lint
ruff format               # フォーマット
pre-commit run --all-files
```

ローカルに Python 環境を作らない場合は Docker で同じことができる（`Dockerfile.dev`）:

```bash
scripts/dev-docker.sh                                    # pytest
scripts/dev-docker.sh ruff check
scripts/dev-docker.sh python -m pytest tests/evaluation -q
```

## プロジェクト構成

- `backend/lash_extraction/` — 抽出ドメイン（landmark / ROI / alignment / evidence / matting）
- `backend/sessions/` — セッションの永続化とユースケース
- `backend/catalog/` — 商品アセットカタログと形状記述子
- `backend/jobs/` — Matting 非同期ジョブ
- `backend/strokes/` — ブラシストローク（ベクタ保存・ラスタライズ）
- `backend/infrastructure/` — Supabase アダプタ（未設定ならローカル実装）
- `backend/video.py` — 動画モード（ベストフレーム選択・目元領域差し替え）
- `backend/api/` — FastAPI ルータと合成ルート（`container.py`）
- `backend/app.py` — FastAPI エントリポイント（ルータ登録のみ）
- `frontend/index.html` — カタログ画面（`/`）
- `frontend/extract.html` — 静止画モード（Canvasブラシ補正UI）
- `frontend/video.html` — 動画モード
- `frontend/common.css` — 3ページ共通スタイル
- `evaluation/` — Synthetic Benchmark（既存アルゴリズムの**計測専用**。`backend/` の挙動は変えない）
- `scripts/` — CLI（`generate_benchmark.py` / `run_evaluation.py` / `dev-docker.sh`）
- `tests/` — pytest（API + ドメイン + evaluation）
- `models/face_landmarker.task` — MediaPipeモデル（コミットしない）
- `data/` — セッションデータ（コミットしない）
- `evaluation-data/` `evaluation-results/` — Benchmarkの入出力（コミットしない）

## 設計原則

- 商品（まつ毛）ピクセルは **Generative Product Preserve**: 生成AIに描き直させず元画像から抽出・再合成する
- 商品領域に許す変換は相似変換（移動・縦横比固定の拡縮・回転・反転）のみ。自由変形・パースペクティブ・非相似warpは不可
- 「ピクセル単位で元画像と一致する」とは謳わない（静止画は前景色推定と補間、動画は出力エンコードを経る）。保持強度の定義は README「商品保持の考え方」
- 生成AIは Model Editing のみに使い、Adapter構造で特定サービスに依存しない（docs/ai-editing-api.md）
- 動画は「ベストフレーム1枚をAI加工 → 元動画の目元領域をランドマーク追従で差し替え合成」方式（docs/video-approach.md）
- 却下済み方式を再提案しない: 抽出レイヤーの warp 追従（回転＋縦圧縮の非相似変形で商品の形状が変わるためNG）／毎フレーム独立のAlpha抽出（ちらつく）。経緯は docs/handover.md §3

## フロントエンドの不変条件（静止画モード）

- レイヤーは3系統。`roi_*` などサーバ側レイヤー（目元ROI解像度・ブラシ対象）、`source_*`（アップロード済み元画像・表示専用）、`local_*`（解析前のローカルプレビュー・表示専用）
- 表示専用レイヤーでは `paintCanvas` のサイズを変えない（ブラシ座標＝制約PNGの座標系が壊れる）
- ズームは `#canvasWrap` の `transform: scale()`。ブラシ座標は `state.zoom` で割って画像座標へ戻す
- レイヤー一覧の再構築でレイヤーを落とさない（再開時の `roi_b`、Matting再実行時の `composite_on_edited`）

## Benchmark（`evaluation/`）のルール

- **数値を良くするために production を変えない。** probability / matting / ROI / threshold /
  evidence / alignment を触ってよいのは「アルゴリズムを改善する」PRだけで、計測PRでは触らない
- 見つかった問題は修正せず `docs/benchmark-findings.md` に記録する
- 合格ライン（Dice >= 0.9 等）は設けない。目的はベースラインの取得と回帰検出
- **数値は A / B / C の層に分けて引用する**（`evaluation/README.md` §0）。
  A=コードパスの性質（実写でも有効）、B=相対・頑健性の傾向、C=絶対スコア（回帰検出のみ）。
  「本システムの精度は Dice 0.52」のような C 層の引用をしてはいけない
- 合成データは実写性能を証明しない。甘い点・厳しい点は両方向にあり `evaluation/README.md` §8 に列挙
- `warp_product(interpolation="linear", premultiply=False)` が本番 `recompose_onto` と
  ビット一致することをテストで担保している。本番を変えたらこのテストが落ちる（＝計測対象のズレ検知）

## その他ルール

- PRを作る前に `pytest` / `ruff check` / `pre-commit run --all-files` を全て通す
- FastAPIの `File(...)/Form(...)` 引数デフォルトは許容（ruff B008 はignore済み）
- 既存の静止画モードの機能は壊さない・削除しない
- uvicorn は自動リロードしないので、ルート追加・pull後はサーバを再起動する（しないと404を誤診する）
- `.venv` に ruff/pre-commit が無い環境では `uvx ruff check` / `uvx pre-commit run --all-files` を使う
- **ruff のバージョンは3箇所で必ず揃える**: `requirements-dev.txt` / `.pre-commit-config.yaml` の `rev` /
  `.github/workflows/ci.yml` の `version`（現在 0.16.3）。版によって規則が違うため、割れていると
  手元の `ruff check` が通るのに CI だけ落ちる（実例: 0.9.6 のみ UP038 を出し、
  `isinstance(x, (int, float))` を `int | float` に直させる）。上げるときは3箇所同時に上げる
