# Supabase Phase B 実装記録

対象: 優先度「中 — プロダクト価値を伸ばす」の 4 機能。

- B-1 抽出済み商品まつ毛アセットのカタログ
- B-2 pgvector による形状類似検索
- B-3 Matting の非同期ジョブ化と Realtime 進捗通知
- B-4 ブラシストロークの永続化と作業再開

Supabase 未設定でも PoC がそのまま動くことを前提条件にした。すべての機能はローカル
アダプタ（`data/` 配下の JSON / PNG / インメモリ）でも成立し、環境変数が揃ったときだけ
Supabase 実装に差し替わる。

## ディレクトリ構成（ミノ駆動: ドメインごとにパッケージを分ける）

```
backend/
  app.py                     FastAPI 本体（ルータ登録だけ）
  api/
    container.py             合成ルート（Supabase or ローカルのアダプタ選択）
    errors.py                ドメイン例外 → HTTP ステータス変換
    config_routes.py         /api/config
    sessions_routes.py       /api/session, /api/matte, /api/matte/jobs, /api/recompose, /api/image
    catalog_routes.py        /api/assets ...
    stroke_routes.py         /api/sessions/{id}/strokes
  lash_extraction/           画像処理ドメイン（旧 backend/pipeline.py を分割）
    landmarks.py  roi.py  alignment.py  evidence.py  matting.py
  sessions/
    store.py                 セッション成果物のファイル配置
    service.py               ユースケース（作成 / Matting / 再合成）
    errors.py                SessionNotFound, FaceNotDetected, MatteNotReady ...
  catalog/
    descriptor.py            LashDescriptor（64 次元形状記述子）, alpha_coverage
    asset.py                 AssetDraft / ProductAsset
    service.py               CatalogService とポート定義（Repository / Storage）
    local.py                 ローカル JSON + 総当たりコサイン検索（pgvector の代替）
  jobs/
    job.py                   MatteJob（状態機械）
    repository.py            InMemory / Supabase ミラーリング
    runner.py                MatteJobRunner（ThreadPool + 進捗通知）
  strokes/
    stroke.py                BrushTool / Stroke / StrokeSet（値オブジェクト）
    constraints.py           Canvas PNG → 制約マップ
    repository.py service.py 保存・復元
  infrastructure/
    supabase_gateway.py      Supabase クライアントと各ポートの実装
```

依存の向きは常に `api → service → domain`、外部 I/O は `infrastructure` と各
`repository` に閉じている。`lash_extraction` は Supabase を一切知らない。

## TDD（t-wada 方式）で進めた順序

1. Red: `tests/catalog/test_descriptor.py`（次元数 / 空 alpha / スケール・平行移動不変性）
2. Green: `backend/catalog/descriptor.py`
3. Red → Green: `tests/strokes/test_strokes.py` → `stroke.py`
4. Red → Green: `tests/jobs/test_matte_job.py`, `test_runner.py` → `job.py` / `runner.py`
5. Red → Green: `tests/catalog/test_service.py`（インメモリのテストダブル）→ `service.py`
6. Refactor: `backend/pipeline.py` を `lash_extraction/` へ分割、`app.py` を薄いルータへ
7. Red → Green: `tests/api/`（カタログ / ジョブ / ストローク / セッション）→ 各ルータ

最終: 101 tests passed / `ruff check --select E4,E7,E9,F,I` clean。

## B-1 / B-2 カタログと類似検索

`product_assets` に Product RGBA のメタデータと 64 次元 embedding を保存する。
PNG 本体は Storage（private bucket `product-assets`）へ、未設定時は `data/assets/`。

記述子 `LashDescriptor.from_alpha` は alpha のバウンディングボックスを 128×128 に
正規化したうえで、16 次元 × 4 ブロックを L2 正規化して連結する。

| ブロック | 内容 | 意味 |
| --- | --- | --- |
| 列方向密度 | 各列の alpha 合計 | まつ毛ラインに沿った広がり |
| 行方向密度 | 各行の alpha 合計 | 長さの分布 |
| alpha ヒストグラム | 値の分布 | 密度・柔らかさ |
| 勾配方向ヒストグラム | Sobel の向き × 強度 | カールの向き |

学習済みモデルの埋め込みではなく hand-crafted な記述子である。スケール・平行移動には
不変で、カール違いは類似度が下がることをテストで固定している。

検索は Supabase では `match_product_assets` RPC（コサイン距離 + ivfflat）、ローカルでは
総当たりコサインで同じインターフェースを満たす。

## B-3 非同期ジョブと Realtime

`POST /api/matte/jobs` は 202 とジョブ ID を返し、`MatteJobRunner` がワーカースレッドで
Matting を実行する。状態遷移は `queued → running → done|failed`、進捗は
`20 trimap / 45 alpha / 85 foreground / 100 done`。

Supabase 設定時は `MirroringJobRepository` が更新のたびに `matte_jobs` を upsert し、
フロントエンドは `@supabase/supabase-js@2.45.4` で `postgres_changes` を購読する。
未設定時は同じ UI が `GET /api/matte/jobs/{id}` のポーリングにフォールバックする。
ミラー失敗は Matting 本体を止めない（進捗通知は補助機能）。

既存の同期 `POST /api/matte` は互換性のため残している。

## B-4 ブラシストロークの永続化

Canvas のラスタ PNG に加えて、ストロークをベクタで保存する。

```json
[{ "tool": "fg", "radius": 12, "points": [[10, 20], [11, 21]] }]
```

UI のモード名との対応は `add → fg` / `unknown → unknown` / `remove → bg`。
`StrokeSet.rasterize()` が Trimap 制約マップ（+1 / 2 / -1）を再生成するので、
`POST /api/matte?use_saved_strokes=true` で保存済み補正のまま再実行できる。
UI ではセッション ID を入力して「再開」するとストロークを再描画する。

## API

| メソッド | パス | 説明 |
| --- | --- | --- |
| GET | `/api/config` | ブラウザ向け設定（publishable key のみ、service role key は返さない） |
| GET | `/api/sessions/{id}` | セッションのメタ情報と存在するレイヤー |
| POST | `/api/matte/jobs` | Matting をジョブとして投入（202） |
| GET | `/api/matte/jobs/{job_id}` | ジョブ状態・進捗・結果 |
| PUT | `/api/sessions/{id}/strokes` | ブラシストローク保存 |
| GET | `/api/sessions/{id}/strokes` | ブラシストローク復元 |
| POST | `/api/assets` | 抽出結果を商品として登録 |
| GET | `/api/assets` | カタログ一覧 |
| GET | `/api/assets/{id}` | 詳細 |
| GET | `/api/assets/{id}/image` | Product RGBA PNG |
| GET | `/api/assets/{id}/similar` | pgvector 類似検索 |

## Supabase スキーマ（適用済み: project `ezlkjkeectohhoykdntd`）

- `product_assets`（`embedding vector(64)` + ivfflat コサインインデックス）
- `matte_jobs`（`supabase_realtime` publication に追加）
- `session_strokes`
- RPC `match_product_assets(query_embedding, match_count, exclude_id)`
- Storage private bucket `product-assets`

## 環境変数

```bash
export SUPABASE_URL=https://<project-ref>.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=...   # サーバー専用。ブラウザへ渡さない
export SUPABASE_PUBLISHABLE_KEY=...    # Realtime 購読用（ブラウザへ渡す）
```

未設定なら自動的にローカルアダプタで動作する（`data/assets/`, `data/<session>/strokes.json`,
インメモリジョブ）。

## 付随して直したこと

`_norm_percentile` のゼロ幅判定を `1e-6` → `1e-3` にした。同一画像同士の Difference Map で
Sobel の丸め誤差（約 1.5e-5）が「信号」として正規化され、
`tests/.../test_identical_images_give_zero` が環境によって落ちていた（main でも再現）。

## 未実装 / 今後

- Auth 未実装のため RLS は暫定。`matte_jobs` の匿名 read ポリシーとセッション画像 API は
  ユーザー所有権ベースに閉じる必要がある。
- `product_assets` / Storage オブジェクトのポリシー未整備。
- ジョブのキャンセル・リトライ・多重実行防止、ワーカーのプロセス分離。
- セッション成果物そのものの Storage 移行（現状はローカル `data/`）と保持期間管理。
- embedding の学習ベース化（現状は hand-crafted）。
