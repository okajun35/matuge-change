# matuge-change

商品まつ毛保持型 AIモデル画像加工システムの PoC。

まつ毛（商品＋商品に隠された/動かされた自まつ毛を含む「装着後の目元まつ毛外観」）を
Alpha Matte として抽出し、後段の AI モデル加工画像へ再合成することを目標とする。

## プロジェクト概要

### 1. 作りたいもの

**EC販売者向けに、商品の着用画像からAIモデルの商品紹介動画を簡単に作れるWebアプリです。**

販売者本人やスタッフなどが、実際の商品を着用して撮影した画像を素材として使用します。

その画像から、**着用している商品そのものの見た目をできるだけ維持したまま、着用人物だけをAIモデルへ変更**し、さらにその画像をもとに短い商品紹介動画を生成します。

#### コンセプト

> **「商品はそのまま、人だけAIモデルへ。」**

AIで商品そのものを新しく生成するのではなく、実際に販売している商品の着用状態を基準として、人物部分をAIモデルへ置き換えることを目指します。

### 2. 解決したい課題

AmazonなどのECモールで商品を販売する際、商品単体の画像だけではなく、

- 実際に商品を着用した状態
- 使用したときの見え方
- 動きのある商品紹介動画

などを見せることで、購入者に商品の特徴をより分かりやすく伝えることができます。

一方で、着用画像や動画を制作するためには、

- モデルの手配
- 撮影
- 撮影場所の準備
- 動画制作・編集

などが必要になり、小規模なEC販売者にとっては時間・費用・手間が大きな負担になります。

この課題は、実際にAmazonで自己ブランド商品を販売する中で感じてきたものです。

そこで、

**販売者自身が商品を着用してスマートフォンなどで撮影するだけで、AIモデルによる商品紹介画像・動画を作れる仕組み**

を目指します。

### 3. 今回の対象商品

今回のハッカソンでは対象を広げず、**つけまつげ1種類**に絞って実装・検証します。

実際にECで販売している商品を使用し、

- 実物の商品
- 商品画像
- 実際の着用画像

を用いて検証します。

まずは1商品で、**実際の商品をどこまで維持しながら人物だけを変更できるか**を重視します。

### 4. 最も重要な要件

このプロジェクトで最も重要なのは、

> ### **人物は変えても、商品は変えない**

という点です。

目的は、AIで単に「つけまつげを着けたAI美女」を生成することではありません。

元の着用画像に写っている**実際に販売する特定の商品**について、可能な限りその特徴を維持する必要があります。

つけまつげの場合、特に以下の要素を重要視します。

- まつげの長さ
- 毛束の形
- 毛束の間隔
- ボリューム
- カール
- 毛先の方向
- 目元への装着位置
- 着用したときの全体的な見え方

AIモデルへ人物を変更した結果、元の商品とは異なる「似たような別のつけまつげ」が生成されてしまうと、ECの商品紹介素材として正確ではありません。

そのため本プロジェクトでは、**人物の美しさや生成画像としての完成度だけを追求するのではなく、商品の再現性を優先します。**

### 5. 目指す体験

最終的には、EC販売者が難しい画像編集や動画制作を意識することなく、

**商品を着用する → 撮影する → AIモデルを選ぶ → 商品紹介素材ができる**

というシンプルな流れで利用できることを目指します。

今回のハッカソンでは、その中でも特に、

**「実際の商品を維持したまま、着用人物をAIモデルへ変更できるか」**

というコア部分の実現と検証に取り組みます。

## 商品保持の考え方（Generative Product Preserve）

本プロジェクトの不変の目標は「商品を生成AIに描き直させない」こと。商品領域は実物の
着用画像・動画から取り出したピクセルだけを使う。ただし保持の強さはモードで異なる。

- 動画モード: 元フレームの目元領域をそのまま貼り戻す（幾何変換はAI加工画像側にのみ掛ける）。
  合成段階では商品ピクセルは無変形で、劣化は境界のフェザー帯と出力の再エンコードに限られる。
- 静止画モード: 抽出した商品領域を相似変換（移動・縦横比固定の拡縮・回転・反転）＋補間で貼る。
  形状（長さ・毛束・間隔・カール・毛先方向）と縦横比は保持するが、Matting の前景色推定と
  補間リサンプルを通るため、元画像とピクセル単位で一致するわけではない。
  自由変形・パースペクティブ・生成AIによる描き直しは行わない。

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
- 商品ピクセルは幾何変換せず、元フレームの目元領域をそのまま貼り戻す（変形するのはAI加工画像側）。
  最終 MP4 は H.264 再エンコードを経る

## セットアップ（Docker / WSL 推奨）

Windows の WSL2（Docker Desktop でも可）で動かす場合は Docker だけで完結する。
Python も MediaPipe モデルも用意不要。

```bash
git clone https://github.com/okajun35/matuge-change.git
cd matuge-change
docker compose up --build
```

ブラウザ（Windows 側）で http://localhost:8000 を開く。

- 初回ビルドは依存のDLで数分かかる。起動後も MediaPipe / numba のインポートに
  30〜60 秒かかるため、すぐ開けない場合は少し待つ（`docker compose ps` が healthy になれば準備完了）
- 抽出結果・カタログはホスト側の `./data` に残る（コンテナを作り直しても消えない）
- 停止は `docker compose down`、コード更新後は `docker compose up --build`
- Supabase を使う場合は `.env` に `SUPABASE_URL` 等を書けば compose が読み込む（未設定なら `data/` のローカル実装で動く）

## セットアップ（ローカル Python 環境）

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

- [docs/handover.md](docs/handover.md) — 引き継ぎメモ（採用/却下した方式とその理由、既知の落とし穴、未検証事項）
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
