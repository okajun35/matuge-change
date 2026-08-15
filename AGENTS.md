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
- `tests/` — pytest（API + ドメイン）
- `models/face_landmarker.task` — MediaPipeモデル（コミットしない）
- `data/` — セッションデータ（コミットしない）

## 設計原則

- 商品（まつ毛）ピクセルはLevel 3（Pixel Preserve）: AIに描き直させず元画像から抽出・保持・再合成する
- 生成AIは Model Editing のみに使い、Adapter構造で特定サービスに依存しない（docs/ai-editing-api.md）
- 動画は「ベストフレーム1枚をAI加工 → 元動画の目元領域をランドマーク追従で差し替え合成」方式（docs/video-approach.md）
- 却下済み方式を再提案しない: 抽出レイヤーの warp 追従（商品ピクセルを変形するためNG）／毎フレーム独立のAlpha抽出（ちらつく）。経緯は docs/handover.md §3

## フロントエンドの不変条件（静止画モード）

- レイヤーは3系統。`roi_*` などサーバ側レイヤー（目元ROI解像度・ブラシ対象）、`source_*`（アップロード済み元画像・表示専用）、`local_*`（解析前のローカルプレビュー・表示専用）
- 表示専用レイヤーでは `paintCanvas` のサイズを変えない（ブラシ座標＝制約PNGの座標系が壊れる）
- ズームは `#canvasWrap` の `transform: scale()`。ブラシ座標は `state.zoom` で割って画像座標へ戻す
- レイヤー一覧の再構築でレイヤーを落とさない（再開時の `roi_b`、Matting再実行時の `composite_on_edited`）

## その他ルール

- PRを作る前に `pytest` / `ruff check` / `pre-commit run --all-files` を全て通す
- FastAPIの `File(...)/Form(...)` 引数デフォルトは許容（ruff B008 はignore済み）
- 既存の静止画モードの機能は壊さない・削除しない
- uvicorn は自動リロードしないので、ルート追加・pull後はサーバを再起動する（しないと404を誤診する）
- `.venv` に ruff/pre-commit が無い環境では `uvx ruff check` / `uvx pre-commit run --all-files` を使う
