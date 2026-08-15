# matuge-change

商品まつ毛保持型 AIモデル画像加工システムの PoC。

まつ毛（商品＋商品に隠された/動かされた自まつ毛を含む「装着後の目元まつ毛外観」）を
Alpha Matte として抽出し、後段の AI モデル加工画像へ再合成することを目標とする。

## PoC の範囲 (Phase 1–2)

- 装着画像（＋任意で未装着画像）から目元 ROI を自動検出
- 未装着画像がある場合: 位置合わせ（Landmark Affine + ECC）→ Difference Map
- 未装着画像がない場合: 局所暗部検出によるフォールバック
- 初期 Probability 推定 → ユーザーがブラシ（＋商品 / ？中間 / −背景）で補正
- Soft Trimap → Closed-form Alpha Matting + Foreground 推定 (pymatting)
- Product RGBA 出力・未装着画像への再合成・再合成誤差の表示
- 抽出済みまつ毛の AI 編集済み画像への再合成（Landmark ベース位置合わせ + Alpha 合成）
- 抽出済み商品まつ毛のカタログ登録と形状類似検索（64 次元記述子 / pgvector）
- Matting の非同期ジョブ化と進捗表示（Supabase Realtime、未設定時はポーリング）
- ブラシストロークのベクタ保存とセッション再開

詳細は [docs/supabase-phase-b.md](docs/supabase-phase-b.md)。

アップロードした元画像・AI 加工済み画像・合成結果と Matting 実行履歴（`runs.json`）は
セッションディレクトリに残る（[docs/session-provenance.md](docs/session-provenance.md)）。

## 動画モード

顔固定・まばたき動画向けの「目元領域まるごと差し替え方式」（[docs/video-approach.md](docs/video-approach.md)）。

- 動画をフレーム分解し、目が開いていて鮮明なベストフレームを選択・出力
- ベストフレームを外部 AI で加工（目元は変えないことを推奨）してアップロード
- 各フレームへ AI 加工画像をランドマーク位置合わせし、元フレームの目元領域
  （まつ毛・まぶた・まばたきの動きごと）をフェザー付きで貼り戻して MP4 を出力
- 商品ピクセルは一切変形・再生成しない（Level 3 Pixel Preserve）

## セットアップ

```bash
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install -r requirements.txt
curl -sL -o models/face_landmarker.task --create-dirs \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## テスト・Lint

```bash
uv pip install -r requirements-dev.txt
python -m pytest          # テスト
ruff check                # Lint
ruff format               # フォーマット
```

pre-commit（commit 時に自動で lint / format / 各種チェック）:

```bash
pre-commit install
pre-commit run --all-files   # 手動で全ファイルにかける場合
```

GitHub Actions（`.github/workflows/ci.yml`）で PR / main push 時に
`ruff check` / `ruff format --check` と `pytest` が実行される。

## ドキュメント

- [docs/ai-editing-api.md](docs/ai-editing-api.md) — AI モデル加工（Gemini / FLUX 等）の API 選定と Adapter 設計、目元保護マスク
- [docs/video-approach.md](docs/video-approach.md) — 顔固定・まばたき動画への対応方針（目元領域まるごと差し替え方式、スタビライズ）
- [docs/session-provenance.md](docs/session-provenance.md) — 元画像 / 合成元 / 合成結果と実行履歴のローカル保存

## 起動

```bash
. .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

ブラウザで http://localhost:8000 を開く。画面はページごとに分かれている。

| URL | 画面 |
| --- | --- |
| `/` | 商品カタログ（登録済み商品の一覧・類似検索） |
| `/extract.html` | 静止画モード（まつ毛 Alpha 抽出・再合成） |
| `/video.html` | 動画モード（ベストフレーム選択・目元領域差し替え合成） |

### Supabase 連携（任意）

環境変数が揃っているときだけ、カタログ・類似検索・ジョブ進捗が Supabase
（Postgres + pgvector + Storage + Realtime）に切り替わる。未設定なら `data/` 配下の
ローカル実装で同じ機能が動く。

```bash
export SUPABASE_URL=https://<project-ref>.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=...   # サーバー専用（ブラウザへは渡さない）
export SUPABASE_PUBLISHABLE_KEY=...    # Realtime 購読用
```

## 使い方（静止画モード: `/extract.html`）

1. 装着画像（必要なら未装着画像も）をアップロードして「解析開始」
2. 表示レイヤーを Probability に切り替えて初期推定を確認
3. ＋商品 / ？中間 / −背景 ブラシで必要な箇所だけ補正（中間は Trimap の Unknown 帯に強制し Matting に境界判定を任せる）。ストローク単位で Undo/Redo 可能（↶/↷ ボタン or Ctrl+Z / Ctrl+Y）
4. 「Matting実行」→ Alpha / Product RGBA / 未装着画像への再合成を確認
5. FG/BG 閾値スライダーで Trimap を調整して再実行可能
6. AI 加工済みのモデル画像を「編集済み画像」に選択して「再合成」→ 抽出まつ毛を位置合わせして合成
7. 「表示中レイヤーを保存」で表示中の画像を PNG ダウンロード
8. 商品名を入れて「この抽出結果を商品として登録」→ カタログ（`/`）のカードをクリックすると形状の近い商品を表示
9. ブラシ補正はストローク単位で自動保存される。セッションID を入力して「再開」すると復元できる

## 使い方（動画モード: `/video.html`）

1. 動画を選択して「解析開始」
2. 選択されたベストフレームを「ベストフレームを保存」でダウンロード
3. 外部 AI でベストフレームを加工（目元は変えない指示を推奨）
4. 加工済み画像を「AI加工済み画像」に選択し、必要なら「目元領域の広さ」を調整して「動画合成」
5. 合成結果をプレビューし、「合成動画を保存」で MP4 ダウンロード
