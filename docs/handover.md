# 引き継ぎメモ（セッションをまたぐ開発者・エージェント向け）

これまでのセッションで得た「コードを読むだけでは分からない」判断の経緯・落とし穴をまとめる。
新しいセッションを始めるときは `AGENTS.md` → 本ファイル → 該当する `docs/*.md` の順に読むとよい。

## 1. 何を作っているか（不変の目標）

- 生成AIに商品（つけまつ毛）を描き直させない。**商品領域を元画像から抽出・保持し、AI加工後の画像へ貼り戻す**
- 抽出対象は「商品＋商品に隠された/動かされた自まつ毛」。あとでAI加工画像に貼るので、その範囲があればよい
- 目標は **Level 3 = Pixel Preserve**: 束数・長さ・カール・密度・根元位置・左右差・毛先形状、そして可能な限り元RGBそのものを保持する
- 合成モデルは単純なアルファ合成:

  ```text
  out = alpha * original + (1 - alpha) * edited
  ```

## 2. 現在の機能一覧（実装済み）

| 機能 | 場所 | 備考 |
| --- | --- | --- |
| 静止画抽出（差分推定→Trimap→Matting→Product RGBA） | `backend/lash_extraction/`, `/extract.html` | 未装着画像なしでも暗部ベースで動く |
| 3値ブラシ（＋商品 / ？中間 / −背景） | `frontend/extract.html` | 中間は Trimap の Unknown=128 を強制 |
| Undo/Redo | `frontend/extract.html` | DB不要。Canvasスナップショットをメモリに積むだけ |
| セッション永続化・再開 | `backend/sessions/` | `data/<session-id>/`。ストロークはベクタ保存 |
| 編集済み画像への再合成 | `/api/recompose` | landmark affine で位置合わせ。**新しいセッションのみ**（旧セッションは `landmarks.npy` 無し） |
| カタログ（商品一覧・類似検索・RGBA/マスクDL） | `frontend/index.html`, `backend/catalog/` | pytest が作るダミー資産が残るのでUI確認しやすい |
| 動画モード（目元領域まるごと差し替え） | `backend/video.py`, `/video.html` | 下記 §4 |
| 元画像レイヤー表示（装着/未装着/AI加工済み） | `frontend/extract.html` | `source_*` |
| アップロード直後のプレビュー＋ズーム/スクロール | `frontend/extract.html` | `local_*` 擬似レイヤー、`#canvasWrap` の `scale()` |
| Docker / WSL 起動 | `Dockerfile`, `docker-compose.yml` | `docker compose up --build` → http://localhost:8000 |
| Lint / CI / pre-commit | `pyproject.toml`, `.github/workflows/ci.yml` | CI の test ジョブは MediaPipe 用に `libegl1` / `libgles2` を apt する |

## 3. 設計上の決定と、その理由（覆さないための記録）

- **AI画像加工は自前実装しない**: 外部API（Gemini 画像編集 / FLUX Kontext 等）を Adapter 経由で呼ぶ前提。抽出済みAlphaを膨張させた保護マスクで目元を編集対象外にする → `docs/ai-editing-api.md`
- **動画生成AIは使わない**: 画像編集APIは静止画専用。全フレームを個別にAI加工すると顔・背景がフレームごとに揺れてちらつくため非推奨
- **却下: 抽出したまつ毛レイヤーをランドマークで warp して各フレームに貼る方式**
  - 商品ピクセルを変形してしまい Level 3（Pixel Preserve）に反する
  - まばたきの形状変化に追従できない
- **却下: 毎フレーム独立にAlpha抽出**
  - 閉眼中は差分が「まぶたの動き」に支配され破綻し、フレーム間でAlphaが揺れてちらつく
  - どうしてもやるなら時間方向の一貫性を持つ video matting（SAM2 / MatAnyone 系）が必要で重い → `docs/video-expression-matting.md`
- **採用: 目元領域まるごと差し替え方式** → §4
- **スタビライズは「顔ランドマーク基準」で行う**: 全フレームをベストフレームの顔位置へ affine で揃えると、手ブレ除去と貼り付け位置合わせが同じ変換で済む。ffmpeg vidstab 等の汎用スタビライザーは背景基準なので顔が動き残る（前処理としてなら有用）

## 4. 動画モードの方式（現状の基本形）

1. 動画からベストフレーム（目が最も開いたフレーム）を1枚出力し、外部AIで加工する（目元は変えない指示）
2. 各フレームについて、AI加工画像を顔ランドマークで位置合わせし、その上に**元動画のそのフレームの目元領域**（まつ毛・自まつ毛・まぶた・まばたきの動きごと）をフェザー付きマスクで貼り戻す
3. 再エンコードして動画にする

- 商品ピクセルは無加工のまま、まばたきの形状変化は元動画そのものなので追従問題が起きない
- 弱点: まぶたの肌も元動画のままなので、AI加工で肌色を大きく変えると境界に違和感が出る。境界のフェザー＋color transfer、または「目元周辺の肌色を変えない」加工指示で運用する
- 表情変化が大きい動画向けの発展方針は `docs/video-expression-matting.md`

## 5. フロントエンドの構造と注意点

- ページは3つ: `/`（カタログ）、`/extract.html`（静止画）、`/video.html`（動画）。共通スタイルは `frontend/common.css`
  - 過去にハッシュルーティング案と別ページ案が競合し、**別ページ案を採用**（URL共有・ページ毎にJSが分離できる）
- `extract.html` のレイヤーは3種類あり、混同するとバグる:
  - サーバ側レイヤー（`roi_a` / `difference` / `probability` / `trimap` / `alpha` / `product_rgba` / `composite_*`）… 目元ROI解像度。ブラシ対象
  - `source_*` … アップロード済み元画像（フル解像度）。表示専用
  - `local_*` … 解析前に選択したファイルの `URL.createObjectURL()` プレビュー。セッション不要・表示専用
- **表示専用レイヤーで `paintCanvas` のサイズを変えてはいけない**（ブラシ座標＝制約PNGの座標系が壊れる）。`isViewOnlyLayer()` で分岐し、ペイントキャンバスを非表示＋ブラシOFFにしている
- ズームは `#canvasWrap` の `transform: scale()` で行い、`#canvasWrap` の実寸を `canvas幅 * zoom` に設定してスクロール範囲を合わせる。ブラシ座標は `(clientX - rect.left) / state.zoom` で画像座標へ戻す
- レイヤー一覧を再構築するとき（Matting後・再開後）、`layers` に無いものは消えるので注意:
  - セッション再開時はファイル入力が空でも `roi_b` を残す
  - Matting再実行で `composite_on_edited` を落とさない
  - 解析・Matting後は `local_*` を選択中でも結果レイヤーへ切り替える

## 6. 過去に踏んだ落とし穴（再発防止）

- **uvicorn は自動リロードしない**: 新しいルートを追加/pullしたらサーバを再起動する。していないと 404 を「バグ」と誤診する
- MediaPipe は `libEGL.so.1` / GLES を要求する → Docker/CI に `libegl1` `libgles2` を入れる
- `fetch` の例外を catch しないと、アップロード失敗時にUIが「解析中…」のまま固まる（PR #9 の `postForm()`）
- 0バイト画像アップロードは 500 ではなく 400 を返す
- Product RGBA 表示後に他レイヤーへ切り替えると何も描画されない不具合があった（stage の id を書き換えていたため）。今は `.checker` クラスのトグル
- カタログカードの透過チェッカーがCSSの宣言順で効いていなかった
- ローカル環境では `.venv/bin/ruff` が無いことがある → `uvx ruff check` / `uvx pre-commit run --all-files` で代用できる

## 7. 未検証・今後の課題

- 実際の「まばたきする実動画」での動画モード検証（合成動画では数値確認済み）
- AI加工API（Gemini/FLUX）の Adapter 実装 — APIキー未提供
- 抽出Alphaに虹彩付近が混ざることがある（`−背景`ブラシで除外して再抽出する運用）
- 表情変化が大きい動画向け Phase 2/3（マスク伝播・temporal smoothing・SAM2/MatAnyone）
