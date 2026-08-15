# 引き継ぎメモ（セッションをまたぐ開発者・エージェント向け）

これまでのセッションで得た「コードを読むだけでは分からない」判断の経緯・落とし穴をまとめる。
新しいセッションを始めるときは `AGENTS.md` → 本ファイル → 該当する `docs/*.md` の順に読むとよい。

## 1. 何を作っているか（不変の目標）

- 生成AIに商品（つけまつ毛）を描き直させない。**商品領域を元画像から抽出・保持し、AI加工後の画像へ貼り戻す**
- 抽出対象は「商品＋商品に隠された/動かされた自まつ毛」。あとでAI加工画像に貼るので、その範囲があればよい
- 目標は **Generative Product Preserve**（生成AIに商品を描き直させない）: 束数・長さ・カール・密度・根元位置・左右差・毛先形状を保持し、RGBは実物画像由来のピクセルのみを使う
- 「ピクセル単位で元画像と一致させる」という意味ではない。静止画は相似変換＋補間、動画は貼り戻し（合成段階は無変形）。保持強度の定義は README「商品保持の考え方」を参照
- 合成モデルは単純なアルファ合成:

  ```text
  out = alpha * original + (1 - alpha) * edited
  ```

## 2. 現在の機能一覧（実装済み）

| 機能 | 場所 | 備考 |
| --- | --- | --- |
| 静止画抽出（差分推定→Trimap→Matting→Product RGBA） | `backend/lash_extraction/`, `/extract.html` | 未装着画像なしでも暗部ベースで動く |
| 手動ROIモード（横顔・目のアップ） | `/api/session` の `roi_rect`, `/extract.html` | 顔検出とeye_priorをスキップ。下記 §8 |
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
  - 商品の形状そのものを変える非相似変形（根元を軸にした回転＋縦圧縮）になり、「商品領域には相似変換のみ」という原則を破る
  - まばたきの形状変化に追従できない
- **却下: 毎フレーム独立にAlpha抽出**
  - 閉眼中は差分が「まぶたの動き」に支配され破綻し、フレーム間でAlphaが揺れてちらつく
  - どうしてもやるなら時間方向の一貫性を持つ video matting（SAM2 / MatAnyone 系）が必要で重い → `docs/video-expression-matting.md`
- **採用: 目元領域まるごと差し替え方式** → §4
- **静止画の商品ピクセルは「ビット一致」ではない**（ドキュメントでそう書かない）。実際の変換チェーン:
  1. `crop_roi`（`lash_extraction/roi.py`）— ROI幅 > `MAX_ROI_WIDTH=1100` なら抽出時点で `INTER_AREA` 縮小され、以降全工程はそのROI解像度
  2. `estimate_foreground_ml`（`lash_extraction/matting.py`）— Product RGBA の RGB は元画素のコピーでなく前景色の**推定値**
  3. 再合成 — landmark経路は `warpAffine(INTER_LINEAR)`、手動ROI-B経路は前乗算→`resize(INTER_AREA/INTER_LANCZOS4)`→（角度指定時）`warpAffine(INTER_LANCZOS4)`
  4. 半透明域は `alpha*fg + (1-alpha)*edited` でAI生成画素と混色
  保証できるのは「生成AIに描き直させず、実物画像由来の商品領域を相似変換だけで再利用する」こと
- **スタビライズは「顔ランドマーク基準」で行う**: 全フレームをベストフレームの顔位置へ affine で揃えると、手ブレ除去と貼り付け位置合わせが同じ変換で済む。ffmpeg vidstab 等の汎用スタビライザーは背景基準なので顔が動き残る（前処理としてなら有用）

## 4. 動画モードの方式（現状の基本形）

1. 動画からベストフレーム（目が最も開いたフレーム）を1枚出力し、外部AIで加工する（目元は変えない指示）
2. 各フレームについて、AI加工画像を顔ランドマークで位置合わせし、その上に**元動画のそのフレームの目元領域**（まつ毛・自まつ毛・まぶた・まばたきの動きごと）をフェザー付きマスクで貼り戻す
3. 再エンコードして動画にする

- 商品ピクセルは合成段階では無変形（warpはAI加工画像側にしか掛けない）、まばたきの形状変化は元動画そのものなので追従問題が起きない。
  ただし境界のフェザー帯は混色し、出力は libx264 / yuv420p 再エンコードを経るので、最終MP4はコーデック分だけ元フレームからずれる
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
- レイヤー切替時のズームは**解像度が同じ間は維持**する（`showLayer` の `sizeChanged` 判定）。解像度が変わったとき（ROIレイヤー ↔ `source_*` 等）だけ自動フィット/等倍に戻る
- コントロールは `#controls` 内の `fieldset.group` で手順ごと（① 入力と解析 / ② ブラシ補正とMatting / ③ AI加工画像へ再合成 / ④ 商品登録 / セッション）にグループ化。表示レイヤー・ズーム・保存はステージ直上の `#viewBar`。ボタンラベルの丸数字は廃止（legend側に付与）
- ROI-Bの位置合わせコントロール（`#fitControls`）は**常時表示**し、使えないときは非活性＋ツールチップで理由（未解析／自動モード／加工画像レイヤー以外を表示中）を案内する。`display:none` での出し入れは「ボタンが見つからない」問題を生むため戻さないこと
- ブラシ表示ON/OFF（`#brushShow`、キーボード `B`）は**見た目だけ**を切り替える。制約PNGは `paintCanvas` の中身から作られ続けるので、OFFでもMattingにはストロークが効く。ブラシツールを選ぶと自動でONに戻る。表示制御は `applyBrushVisibility()` に集約（表示専用レイヤーの非表示条件もここ）
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

## 6.5 品質の実測値（Synthetic Benchmark）

`evaluation/` に合成データによる計測環境がある。現行 main の実測（100ケース × 4通り）:

- 自動推定は **Recall 0.98 / Precision 0.36**（＝毛の間の肌まで取っている）、Dice 0.52
- 理想的な3値ブラシを与えると Dice 0.72 / Precision 0.75。**品質の大半はブラシ補正が担っている**
- 未装着画像あり（0.524）と無し（0.515）で差がほぼ無い＝差分抽出の利得が出ていない
- 最も崩れるのは JPEG圧縮とピンぼけ。これらは理想ブラシでも回復しない（入力から情報が消えている）
- `recompose_onto` はアルファ前乗算していないため、毛先のRGB誤差が前乗算版の **30〜40倍**（22.6 vs 0.56）

詳細と改善優先順位は `docs/benchmark-findings.md`、指標定義と合成データの限界は `evaluation/README.md`。
**Benchmarkの数値を良くするために production を変えてはいけない**（計測と改善のPRを分ける）。

## 7. 未検証・今後の課題

- 手動ROIモードの実写検証（横顔画像でのAlpha品質）
- 実際の「まばたきする実動画」での動画モード検証（合成動画では数値確認済み）
- AI加工API（Gemini/FLUX）の Adapter 実装 — APIキー未提供
- 抽出Alphaに虹彩付近が混ざることがある（`−背景`ブラシで除外して再抽出する運用）
- 表情変化が大きい動画向け Phase 2/3（マスク伝播・temporal smoothing・SAM2/MatAnyone）
- 横顔の自動ROI（片目のみのROI/prior＋yaw耐性のあるランドマーク）

## 8. 横顔・目のアップは顔検出が使えない（手動ROIモードの理由）

実写の横顔（ほぼ真横）2枚で計測した結果、MediaPipe Face Landmarker は**1点も検出できない**:

- 3000 / 2000 / 1280 / 800 / 512px、±15/30°回転、顔だけクロップ、左右反転すべて検出0件
- `min_face_detection_confidence` を 0.5 → 0.01 まで下げても0件。閾値や解像度の問題ではなく検出器（BlazeFace系）の適用範囲外

さらにランドマークが取れても下流が両目前提で崩れる:

- `compute_eye_roi` は左右両目のbbox → 片目しか見えない横顔ではROIが顔全体級に膨らみ、`MAX_ROI_WIDTH` で縮小されてまつ毛の解像度が落ちる
- `eye_prior` は両目ポリゴンを塗る → 見えない側がノイズ prior になる
- `recompose_onto` の `ALIGN_POINTS` affine は両目＋鼻が見えている前提

そのため横顔・目のアップは**ユーザーが矩形でROIを与える手動ROIモード**で扱う:

- `POST /api/session` に `roi_rect="x0,y0,x1,y1"`（元画像ピクセル）を渡すと顔検出をスキップし、prior無しで暗部/差分evidenceをそのまま probability にする
- 用語は **ROI-A = 装着画像から抽出する範囲**、**ROI-B = 加工画像へ貼り付ける先の矩形**。UIには解析後も押せる `ROI-A 指定（装着画像）` / `ROI-B 指定（加工画像）` ボタン、`ROIクリア`、`ROI枠を表示` チェックボックス（キーボード `H` でも切替）を置く。ROI-Aはシアン実線、ROI-Bはオレンジ破線で表示する
- 解析後に結果レイヤーへ自動切替されると従来のROIドラッグ対象から外れ、さらにモードセレクタは既に `manual` のため `onchange` が発火せず、ROIを引き直せない不具合があった。ROI指定ボタンは適切な装着画像／加工画像レイヤーへ切り替えてドラッグをarmすることで、この落とし穴を避ける
- `POST /api/recompose` に `dest_rect="x0,y0,x1,y1"`（加工画像ピクセル）を指定すると、顔検出・ランドマーク無しでROI-Bへ貼り付けられる。`product_rgba` は縦横比を維持してROI-Bに内接・中央配置し、アルファ前乗算してから拡縮する（ハロー防止）。回転・遠近変形には対応しない
- 手動貼り付けのサイズ基準は、余白込みのROI-A矩形ではなく `product_bbox`（alpha実体の連結成分bbox）へ変更した。ROI-Aの引き方による余白量でまつ毛の貼付サイズが変わる問題を避け、まつ毛外形をROI-Bへ内接させるためである。`product_bbox` は Matting 完了時に `meta.json` へ保存する
- 手動ROI-Bでは `angle`（時計回り、-180〜180度）と `flip`（左右反転）を指定でき、相似変換（移動・縦横比固定の拡縮・回転・反転）のみを行う。自由変形やパースペクティブは商品ピクセルを歪めるため採用しない。回転時はアルファ前乗算後、縮小なら `INTER_AREA` で先に縮小してから回転し、拡大なら `INTER_LANCZOS4` で回転付きリサンプルする
- `dest_rect` 無しの経路は従来のlandmark affineのままで、landmarkが無ければ422になる既存挙動も残っている。したがって横顔同士の貼り付けはlandmark経路では成立せず、ROI-Bの手動指定が必要
- `meta.json` の `mode` が `manual` / `auto`。`dest_rect` は `meta.json` に保存され、セッション取得時に復元できる
- 実測では、実写の横顔2枚に加えて、いただいたAI加工画像（横顔）もMediaPipeでは検出0件だった。横顔同士はlandmark経路では成立しないため、ROI-A抽出後はROI-Bを手動指定する
- trimap / closed-form matting / 3値ブラシは向きに依存しないので、ROIが決まれば横顔でもそのまま動く
- UIのデフォルトは自動。自動で `no face` エラーが返ったらエラー表示で止めず、手動ROIへ自動フォールバックして案内する（`fallbackToManualRoi`）
- 正面画像に手動ROIを使うのも可。顔が検出できれば `eye_prior` と `landmarks.npy` は自動モードと同じく効くので、`dest_rect` を使わない再合成も使える（手動ROIはROIの決め方だけを上書きする）
