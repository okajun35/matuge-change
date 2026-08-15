# AGENTS.md — 開発エージェント向けメモリ

このリポジトリで作業するAIエージェント・開発者が毎回最初に読むべきルール。

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
- `frontend/index.html` — SPA（Canvasブラシ補正UI・動画モード）
- `tests/` — pytest（API + ドメイン）
- `models/face_landmarker.task` — MediaPipeモデル（コミットしない）
- `data/` — セッションデータ（コミットしない）

## 設計原則

- 商品（まつ毛）ピクセルはLevel 3（Pixel Preserve）: AIに描き直させず元画像から抽出・保持・再合成する
- 生成AIは Model Editing のみに使い、Adapter構造で特定サービスに依存しない（docs/ai-editing-api.md）
- 動画は「ベストフレーム1枚をAI加工 → 元動画の目元領域をランドマーク追従で差し替え合成」方式（docs/video-approach.md）

## その他ルール

- PRを作る前に `pytest` / `ruff check` / `pre-commit run --all-files` を全て通す
- FastAPIの `File(...)/Form(...)` 引数デフォルトは許容（ruff B008 はignore済み）
- 既存の静止画モードの機能は壊さない・削除しない
