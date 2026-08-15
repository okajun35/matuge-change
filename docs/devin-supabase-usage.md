# Devin と Supabase の使い方（開発体制の記録）

このリポジトリを「誰が・どうやって」作ったかを、git 履歴から検証できる形で残す。
数値はすべて `main`（初コミット `573ea7b` 〜 PR #16 マージ時点）を対象に、下記コマンドで取得したもの。

```bash
git log --pretty='%h|%ad|%an|%s' --date=short main   # コミット一覧
git shortlog -sne main                               # 著者別コミット数
git branch -r | grep devin/                          # Devin が作ったブランチ
git diff --shortstat 573ea7b main                    # 総変更量
gh pr list --state all --limit 30                    # PR 一覧
```

## 1. Devin をどう使ったか

### 全体像

| 指標 | 値 | 取得元 |
| --- | --- | --- |
| 期間 | 2026-08-14 15:17 UTC 〜 2026-08-15（約 1.5 日） | 初コミットと最新コミットの `%ad` |
| コミット数 | 66 | `git log --oneline main \| wc -l` |
| 総変更量 | 91 files / +7,159 行 | `git diff --shortstat 573ea7b main` |
| PR 数 | 17 | `gh pr list --state all` |
| うち `devin/*` ブランチ | 16 | `git branch -r \| grep -c devin/` |

著者別コミット数（`git shortlog -sne main`）:

| 著者 | コミット | 役割 |
| --- | --- | --- |
| `okazakijun54392` | 45 | Devin セッションによる実装コミット |
| `Jun okazaki` | 16 | 人間によるマージコミット（＝レビューして取り込む側） |
| `okajun35` | 3 | 人間による直接修正（PR #16: Render のポートバインディング修正・依存ピン留め） |
| `devin-ai-integration[bot]` | 1 | Devin のスキル更新 PR（#7） |

つまり **実装は Devin、レビュー・マージ判断は人間**という分担が git 上にそのまま出ている。
デプロイ環境固有の調整（Render のポート）だけ人間が直接コミットした。

### 1 セッション = 1 トピック = 1 PR

ブランチ名は `devin/<unixtime>-<topic>` で、セッションごとに 1 本ずつ切られている。
`main` へは squash せずマージコミットで入れているので、`git log --merges` でセッション単位の
作業境界がそのまま辿れる。

| PR | ブランチ | 内容 | 種別 |
| --- | --- | --- | --- |
| #1 | `devin/1786720643-lash-alpha-poc` | まつ毛 Alpha 抽出 PoC（差分推定＋ブラシ補正＋Matting） | 機能 |
| #2 | `devin/1786743223-docs-lint-ci` | 設計ドキュメント / ruff / GitHub Actions CI / pre-commit | 基盤 |
| #3 | `devin/1786745432-video-mode` | 動画モード（目元領域まるごと差し替え） | 機能 |
| #4 | `devin/1786749415-supabase-phase-b` | Supabase 連携 Phase B（カタログ / pgvector / 非同期ジョブ / ストローク） | 機能 |
| #5 | `devin/1786760000-page-split` | UI をページ分割（カタログ / 静止画 / 動画） | 機能 |
| #6 | `devin/1786762671-asset-mask-download` | カタログから RGBA / マスク PNG をダウンロード | 機能 |
| #7 | `devin/update-skills-1786763319` | `testing-lash-poc` スキルにカタログ画面のテスト手順を追記 | 開発基盤 |
| #8 | `devin/1786764305-docker-wsl` | WSL/Windows から `docker compose up` で動かせるように | 基盤 |
| #9 | `devin/1786765796-fix-upload-stuck` | アップロード失敗時に「解析中…」で固まる不具合 | バグ修正 |
| #10 | `devin/1786767932-source-layers` | 元画像（装着 / 未装着 / AI 加工済み）のレイヤー表示 | 機能 |
| #11 | `devin/1786770950-preview-zoom` | アップロード直後のプレビューとズーム / スクロール | 機能 |
| #12 | `devin/1786771849-session-archive` | セッションを Supabase Storage へ退避 / 取り込み | 機能 |
| #13 | `devin/1786773577-restore-strokes-paint` | 復元直後にブラシ跡が出ない不具合 | バグ修正 |
| #14 | `devin/1786775198-manual-roi-mode` | 手動 ROI モード（ROI-A 抽出 / ROI-B 貼付） | 機能 |
| #15 | `devin/1786790841-readme-overview` | README にプロジェクト概要を追記 | ドキュメント |
| #17 | `devin/1786803734-preserve-wording` | 「Pixel Preserve」表現を実装どおりに修正 | ドキュメント |

（#16 のみ人間の `fix/render-port-binding`）

### セッションを跨いで品質を保つための仕組み

Devin は毎回コンテキストが空から始まるので、「判断の履歴」をリポジトリ側に置いている。
これらのファイル自体も Devin が更新している。

- `AGENTS.md` — 毎セッション最初に読むルール。TDD 必須、PR 前に `pytest` / `ruff check` /
  `pre-commit run --all-files` を通す、フロントエンドの不変条件、**却下済み方式を再提案しない**
- `docs/handover.md` — 採用 / 却下した方式とその理由、踏んだ落とし穴（PR #11 のセッションで新規作成）。
  「コードを読むだけでは分からないこと」専用
- `.agents/skills/testing-lash-poc/SKILL.md` — UI テスト手順。Devin 自身が PR #7 で追記した
- `docs/*.md` — 設計判断の単位でファイルを分けている（`video-approach.md` は却下案とその理由も残す）

この形にした効果が履歴に出ている例:

- PR #5 で `Revert "feat: 商品カタログを別画面化（ハッシュルーティング）"` → 別ページ案へ切り替え。
  この経緯を `handover.md` §5 に書いたので、以降のセッションでハッシュルーティングが再提案されていない
- PR #14 は 06:35 作成 → 11:17 マージで、`Address manual ROI review feedback` /
  `Fix manual ROI UI synchronization` / `Keep fit controls available on image load errors` と
  レビュー指摘対応コミットが積まれている（人間のレビュー → Devin が同ブランチで修正、のループ）
- PR #17 は「README の Pixel Preserve という表現がコードの実装より強い」という人間の指摘を受けて、
  Devin が実装（`warpAffine` の補間、`estimate_foreground_ml` の前景推定）を読み直して用語を修正した

### 典型的なセッションの流れ

1. 人間が課題・不具合・仕様変更を投げる
2. Devin が `AGENTS.md` → `docs/handover.md` → 該当 `docs/*.md` を読む
3. 失敗するテストを `tests/` に書く（Red）→ 実装（Green）→ リファクタ
4. `pytest` / `ruff check` / `pre-commit run --all-files`
5. `devin/<unixtime>-<topic>` を push して PR、必要なら UI をブラウザ操作で検証
6. 人間がレビュー → 指摘は同ブランチで修正 → マージ
7. 新しく分かった落とし穴・却下した方式を `handover.md` / `AGENTS.md` へ書き戻す

## 2. Supabase をどう使っているか

実装は `backend/infrastructure/supabase_gateway.py` に集約し、詳細な設計記録は
[supabase-phase-b.md](supabase-phase-b.md) にある。ここでは用途の要約を示す。

### 設計方針: 常に「任意の依存」

`SUPABASE_URL` と `SUPABASE_SERVICE_ROLE_KEY` が揃ったときだけ Supabase 実装に差し替わり、
未設定なら `data/` 配下のローカル実装（JSON / PNG / インメモリ）で同じ機能が動く。
切り替えは `backend/api/container.py` の合成ルート 1 箇所だけで、`lash_extraction`
（画像処理ドメイン）は Supabase を一切知らない。

```python
# backend/infrastructure/supabase_gateway.py
def is_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
```

### 機能ごとの使い分け

| Supabase 機能 | 用途 | 実装 | ローカル代替 |
| --- | --- | --- | --- |
| Postgres `product_assets` | 抽出済み商品まつ毛アセットのメタデータ（名前・ブランド・alpha 被覆率・再合成誤差） | `SupabaseAssetRepository` | `data/assets/*.json` |
| pgvector + RPC | 形状類似検索。64 次元 hand-crafted 記述子を `vector(64)` に保存し、`match_product_assets` RPC（コサイン + ivfflat）で検索 | 同上 `.rpc(...)` | 総当たりコサイン（`catalog/local.py`） |
| Postgres `matte_jobs` + Realtime | Matting 非同期ジョブの進捗をミラーし、ブラウザが `postgres_changes` を購読して進捗表示 | `SupabaseJobMirror` / `frontend/extract.html`（`@supabase/supabase-js@2.45.4`） | `GET /api/matte/jobs/{id}` のポーリング |
| Postgres `session_strokes` | ブラシストロークをベクタ（`[{tool, radius, points}]`）で保存し、セッション再開で再描画 | `SupabaseStrokeRepository` | `data/<session>/strokes.json` |
| Storage `product-assets`（private） | Product RGBA の PNG 本体 | `SupabaseAssetStorage` | `data/assets/` |
| Storage `sessions` | セッションディレクトリ丸ごとの退避 / 取り込み（別マシンへの作業引き継ぎ、PR #12） | `SupabaseSessionArchive` / `backend/sessions/archive.py` | なし（`ArchiveUnavailable` を返すだけ） |

### 鍵の扱い

- `SUPABASE_SERVICE_ROLE_KEY` はサーバー専用。ブラウザへは渡さない
- `GET /api/config` は `publishable_key` と `realtime` フラグだけ返す（`public_config()`）
- 進捗ミラーの失敗は `contextlib.suppress(Exception)` で握り、Matting 本体を止めない
  （通知は補助機能という位置づけ）

### スキーマの適用も Devin から

スキーマは Supabase MCP 経由でマイグレーションとして適用した（project `ezlkjkeectohhoykdntd`）。

| version | name |
| --- | --- |
| 20260814231424 | `b_phase_asset_catalog_jobs_strokes` |
| 20260814231433 | `matte_jobs_realtime_and_read_policy` |
| 20260814231455 | `match_product_assets_rpc_and_bucket` |

（`20260814215335 create_mcp_demo_items` は接続確認用のデモテーブル）

### 未整備

- Auth 未実装のため RLS は暫定（`matte_jobs` の匿名 read ポリシー等）。所有権ベースに閉じる必要がある
- `product_assets` / Storage オブジェクトのポリシー未整備
- ジョブのキャンセル・リトライ・多重実行防止、ワーカーのプロセス分離
