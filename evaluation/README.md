# Synthetic Benchmark — 商品まつ毛の抽出・保持・再合成を数値で測る

このディレクトリは **既存の画像処理を測るためだけ** のコードで、`backend/` の挙動は一切変えない。
新機能ではなく「現状のベースラインを客観的に取る」ことが目的である。

- 生成: `scripts/generate_benchmark.py`
- 実行: `scripts/run_evaluation.py`
- 出力: `evaluation-results/report.md` / `summary.json` / `summary.csv` / `cases/*/comparison_*.png`

---

## 0. この数値の位置づけ（最初に読む）

**結果はすべて同格ではない。** 引用する前に、どの層の話かを必ず区別する。

| 層 | 内容 | 実写への通用度 | 使い方 |
| --- | --- | --- | --- |
| **A. データセットに依存しないコードの事実** | 前乗算なし warp（landmark経路と手動ROI経路の不整合）、FG無しtrimapでの未処理例外、`crop_roi` の INTER_AREA 縮小、oracle行＝matting単体の上限になる性質 | **そのまま有効。実写でも同じ**（合成データではなくコードパスの性質を測っている） | そのまま修正対象にしてよい |
| **B. 相対・頑健性の傾向** | どの条件で崩れるか（JPEG・ピンぼけが最悪／理想ブラシでも回復しない）、ブラシ補正の寄与が大きい、未装着画像の利得が小さい | **方向性は信頼できる**が、実写数枚での確認が望ましい | 改善の優先順位付けに使う |
| **C. 絶対スコア** | Dice 0.52 などの値そのもの | **実写性能を示さない** | 回帰検出の基準値としてのみ使う |

**C を「本システムの精度」として引用してはいけない。** 絶対値が低く出る理由は主にGTの粒度である。

- GTは**商品の毛1本単位のalpha**で、320×240のうち中央値 **2,586px（画像の3.4%）** しかない。
  1〜2px幅の線を1px膨らませただけで precision は半分になる。大きな物体のセグメンテーションの Dice とは別物
- パイプラインの目標は「商品＋商品に隠された自まつ毛」なので、GT（商品のみ）より広い。
  ただし自まつ毛を除外しても Dice 0.524→0.559 なので、これは主因ではない
- 人間は再合成結果を見て判断するが、この指標は毛1本の一致を見ている。両者は一致しない

逆に合成が**甘い**側面も同程度ある（§8）。したがって「合成は実写より厳しい／甘い」と単純化はできず、
**実写とは別の、より厳しい問いを測っている**というのが正確な理解である。

---

## 1. 何を測るのか

「商品はそのまま、人だけAIモデルへ」を検証するには、次の4つを**分けて**測る必要がある。

| 問い | 指標 |
| --- | --- |
| 商品領域を見つけられたか | IoU / Dice / Precision / Recall（`alpha >= 0.5`） |
| 細い毛の形を保てたか | MAD / SAD / MSE / Gradient error / boundary F1 / 連結成分差 |
| 商品のRGBを保てたか | RGB MAE / RMSE（両者の alpha >= 0.9 の画素のみ） |
| 再合成で崩れないか | 商品周辺帯の MAE / RMSE、`reconstruction_error` |
| 移動・回転でピクセルが変質しないか | exact color preservation / mutation rate / RGB MAE（`evaluation/mutation.py`） |

IoU/Dice だけでは足りない。まつ毛は1〜2px幅なので、`alpha >= 0.5` で二値化した時点で毛先の半透明が消え、
実質「根元の太い帯がどれだけ合っているか」しか見ていない。したがって alpha 空間の誤差（MAD / Grad）を
併記し、Grad をまつ毛の"にじみ・つぶれ"の主指標として扱う。

---

## 2. なぜ Synthetic Dataset なのか

実写のGolden Dataset（人が撮影して1本ずつマスクを描く）は、この規模では作れない。かつ

- 手作業マスクの品質が上限になる（特に毛先の半透明はまず描けない）
- 条件（回転・明るさ・圧縮）を揃えた比較ができない

一方 Synthetic なら **合成した瞬間に正解が完全に既知** になる。

```text
worn = alpha * product + (1 - alpha) * bare
```

さらに本実装では商品を**ベクタ（毛束のポリライン）として持ち、幾何変換はジオメトリに対して行う**。
ピクセルを warp しないので、拡大・回転しても **GT alpha に再標本化誤差が入らない**。
毛先の alpha はスーパーサンプリング（既定4×）による面積被覆から求めるので、
「本物の半透明」がGTに含まれる。切り抜きPNGでは得られない性質である。

### 合成をわざと"きれい過ぎなく"している理由（重要）

bare をピクセル完全一致で渡すと、

```text
worn - bare = alpha * (product - bare)
```

となり、差分抽出は**データを作った式そのもの**を解くだけの自明問題になる。closed-form matting の
前提（合成式が厳密に成立）も完全に満たされるため、Dice がほぼ天井に張り付き、条件差も出ない。
実写の2枚組はそうではない（センサノイズが独立、露出が微妙に違う、頭が動く）。そこで

- ノイズは worn / bare で **別シード**
- `bare_misalign_px` / `bare_misalign_deg` で bare 側だけを平行移動・微小回転（＝別ショット）
- 商品が落とす影を worn に入れるが **GTには入れない**
- 本人の自まつ毛を背景に描き、**GTには入れない**（`gt_ignore` に記録）

を条件軸として持つ。Mode A（bare あり）の数値は、それでも**楽観的な上限**として読むべきである。

---

## 3. Dataset の作り方

### 外部データなし（既定）

```bash
python scripts/generate_benchmark.py --cases 100 --backgrounds 12 --clean
```

手続き生成の目元（肌・二重・白目・虹彩・目のライン・自まつ毛）に、手続き生成の商品まつ毛を合成する。
ダウンロードもライセンス確認も不要で、CIでもそのまま動く。

### 任意の画像フォルダを背景にする

```bash
python scripts/generate_benchmark.py \
  --background-dir /data/periorbital/images \
  --product evaluation-data/products/product_lash.png \
  --cases 100 --output evaluation-data/generated
```

- MediaPipe Face Landmarker で上まぶた（`UPPER_LID_RIGHT` / `UPPER_LID_LEFT`）を取り、**両目の上まぶたに**商品を配置する
- 顔が検出できない画像は既定で **skip**。`--on-no-face fallback` を付けると、画像中央の固定矩形を目元とみなして生成し、手動ROIモードで評価する
- **注意: 目元クロップや横顔ではMediaPipeは1点も検出できない**（`docs/handover.md` §8 の実測）。
  自動ROIモードを評価したい場合は**全顔画像**を用意する
- 画像は既定で幅900pxに縮小する。`MAX_ROI_WIDTH`(1100) を超えるROIは `crop_roi` が INTER_AREA で縮小してしまい、
  その再標本化が商品RGB誤差に混ざるため

### ライセンスについて

- 背景画像・商品PNGは**リポジトリにコミットしない**（`.gitignore` 済み）
- 自動ダウンロードは実装しない。ローカルフォルダを `--background-dir` で渡す方式のみ
- 公開データセット（Periorbital Segmentation Dataset 等）を使う場合は各自でライセンスを確認する

---

## 4. 生成物のレイアウト

```text
evaluation-data/generated/case_0001/
  bare.png        未装着ショット（別シードのノイズ・任意の位置ずれ入り）
  worn.png        装着ショット（合成＋劣化）
  gt_alpha.png    正解 alpha（0-255）
  gt_mask.png     gt_alpha >= 128
  gt_ignore.png   自まつ毛など、採点から除外してよい領域
  gt_product.png  合成に使った商品 BGRA
  metadata.json   条件（scale / rotation_deg / offset / flip / brightness / blur / jpeg / misalign / shadow / roi_rect …）
```

`metadata.json` の `condition` は「ベースラインから変えた軸」を1つだけ持つ。
全組み合わせではなく **1軸ずつ振る** 設計なので、スコア低下の原因を単一条件に帰属できる。

---

## 5. Benchmark の実行

```bash
python scripts/run_evaluation.py --dataset evaluation-data/generated --output evaluation-results
```

各ケースを最大4通りで実行する。

| mode | brush | 何を測るか |
| --- | --- | --- |
| `bare` | `auto` | 未装着画像あり（差分ベース）・ユーザー補正なし |
| `worn_only` | `auto` | 未装着画像なし（暗部フォールバック）・ユーザー補正なし |
| `bare` | `oracle` | 理想的な3値ブラシで補正した場合の上限 |
| `worn_only` | `oracle` | 同上（bareなし） |

`oracle` は GT から作ったブラシストローク（`＋商品` / `？中間` / `−背景`）を `build_trimap` に渡す。
製品UIは human-in-the-loop（3値ブラシ前提）なので、`auto` だけでは実際の到達点を過小に報告してしまう。
`auto` が下限、`oracle` が上限である。なお oracle では trimap が完全に固定されるため、
bare / worn_only の結果は一致する（＝ oracle 行は matting 単体の性能測定になる）。

パイプラインは `SessionService` を一時ディレクトリのストアで直接叩く（HTTPを経由しない）。
API のルータは薄いラッパなので、**本番と同じコードパス**を通る。リポジトリの `data/` は汚さない。

実行が例外で落ちたケースも1行として記録する（`failed=true` + 例外メッセージ、指標はNaN）。
たとえば probability が `fg_thresh` に一度も届かないと trimap にFGが無く、pymatting が `ValueError` を出す。
**落ちたことも結果**であり、平均から静かに消えてはいけない。

---

## 6. 結果の読み方

- `report.md` … 全体・条件別・best/worst・Pixel Mutation の表
- `summary.csv` … 1行 = 1（ケース × mode × brush）。条件列が入っているので自由に集計できる
- `cases/<id>/comparison_<mode>_<brush>.png` … `bare | worn | GT mask | 予測alpha | 抽出商品 | 再合成 | 差分×4`

読むときの注意:

0. **§0 の A / B / C を先に確認する。** 表の中で層が混ざっている。
   「全体」「条件別」「best/worst」は C（絶対値は基準値のみ）と B（相対傾向）、
   「Pixel mutation」「ROI downscale」は A（実写にもそのまま通用する）
1. **Precision と Recall を必ずセットで見る。** 自動推定は recall が高く precision が低い（過剰検出）傾向がある。
   Dice はその両方を反映する。`pred_px / gt_px`（面積比）を見ると過剰検出の量が直接分かる
2. `*_ex_own` は自まつ毛を除外した数値。パイプラインの抽出対象は「商品＋商品に隠された自まつ毛」なので、
   自まつ毛を拾うのは仕様通りだが GT には含まれない。素の precision は構造的に不利に出る
3. 平均は NaN を無視して取る（未定義の比を 0 や 1 として混ぜない）
4. `roi_scale` が 1.0 でないケースは、ROI縮小の再標本化が混ざっている
5. **合格ラインは設けていない。** まずベースラインを取ることが目的である
6. `reconstruction_error` と `comparison_*.png` を必ず併せて見る。指標が低くても再合成が自然に見えることは
   よくあり、その差が「毛1本単位の一致」と「人間の見た目」のギャップそのものである

---

## 7. Pixel Mutation（補間の影響）

`evaluation/mutation.py` は、商品ピクセルが変換でどれだけ変質するかを測る。

- 参照は **ジオメトリから再レンダリングした厳密なGT**。だから幾何のゆがみ（`alpha_mad` / `alpha_grad`）と
  色の変質（`exact_color_preservation_rate` / `rgb_mae`）を分離できる
- 比較する4通り: `nearest` / `linear`（＝現行の再合成）/ `premultiplied_linear` / `premultiplied_lanczos4`
- `warp_product(..., interpolation="linear", premultiply=False)` は **本番 `recompose_onto` とビット一致**することを
  テストで担保している（`tests/evaluation/test_mutation.py::TestMatchesProduction`）。測っている対象が
  本番コードからずれたらテストが落ちる
- `crop_roi` の INTER_AREA 縮小も別表で測る（実写の高解像度写真では抽出前に必ず起きる）

注意: `exact_color_preservation_rate` は **本番では原理的にほぼ 0 になる**。`recompose_onto` の affine は
ランドマークから LMEDS で推定するため厳密な単位行列にならず、必ず全画素が補間される。
`nearest` は色を保つ代わりに1〜2px幅の毛束の形を壊すので、**色の指標だけを見て `nearest` に変えてはいけない**。
だから幾何指標を必ず併記している。

---

## 8. Synthetic Benchmark の限界（必読）

**C層（絶対スコア）は実写性能を証明しない。** §0 の表と併せて読むこと。

### 合成が実写より「甘い」点（＝スコアが良く出る方向）

- 自まつ毛との重なり・絡み（本実装の自まつ毛は独立に描いた別レイヤーにすぎない）
- まぶた・二重による遮蔽、まつ毛が肌に埋まる見え方
- 立体的なカール（3D形状・被写界深度・毛の前後関係）
- 実際の影・皮脂の反射・接着剤の艶
- 装着による目元自体の変形（テープ・のりでまぶたの形が変わる）
- 撮影ごとの化粧・アイラインとの混同（実写では最大の誤検出源になりうる）
- 肌のテクスチャ（毛穴・産毛・眉）が単純で、誤検出源が実写より少ない
- 合成式 `worn = a*P + (1-a)*bare` が厳密に成立している（closed-form matting の前提そのもの）
- ROIが幾何から確実に取れており、ランドマーク検出・`eye_prior` が試されていない

### 合成が実写より「厳しい」点（＝スコアが悪く出る方向）

- GTが毛1本単位で、面積が画像の3〜4%しかない。1px の膨張で precision が半減する
- 手続き生成の商品は毛束の隙間が広く、「隙間の肌を拾う」ペナルティが実物より大きく出る
  （実物のストリップまつ毛はより密なので、同じアルゴリズムでも precision は高く出ると予想される）
- 実商品の毛束の質感・透け方（手続き生成は近似）

**両方向のバイアスが同程度あるため、「合成は実写より厳しい／甘い」と単純化はできない。**
正確には **実写とは別の、より厳しい問いを測っている**。

したがって本Benchmarkの使い方は次の3つに限られる。

1. **A層の指摘の根拠**: コードパスの性質（前乗算なし warp 等）は合成に依存しないのでそのまま有効
2. **相対比較**: アルゴリズムAとBのどちらが良いか、どの条件で崩れるか
3. **回帰検出**: 変更で数値が落ちていないか

「実写でDiceいくら出るか」の答えにはならない。実写評価には、同じ `runner` で読める形の
実写ケース（`worn.png` / `bare.png` / `gt_alpha.png` / `metadata.json`）を別途用意する必要がある。
`evaluation/dataset.py` はその形だけを要求しているので、実写セットが用意できれば同じ指標で評価できる。

### GTを描かずに B層を実写で確認する方法

マスクを1本ずつ描かなくても、GTの要らない指標だけなら実写で確認できる。

- 予測面積 / GT相当の面積比（合成では自動 2.8倍、oracle 1.08倍）。実写でも2〜3倍なら過剰検出は本物
- `reconstruction_error` と alpha のヒストグラム（中間値の比率＝毛先が出ているか）
- 同一の実写画像をJPEG品質を落として通し、劣化耐性だけ相対比較する

これでアノテーションなしに「過剰検出とJPEG脆弱性が実写でも起きるか」を確かめられる。

---

## 9. CI

CI では外部データを一切ダウンロードしない。`tests/evaluation/test_synthetic_benchmark.py` が
2背景 × 1商品 × 5ケース・約120px の極小データを生成して runner を完走させ、

- generator が動く / GTが構成上正しい（合成式・しきい値・ignore・影の扱い）
- metric が解析的な正解と一致する
- runner と report が完走し、成果物が揃う
- `warp_product` が本番 `recompose_onto` と一致する

ことを確認する。

## 10. Docker で回す

ホストに `.venv` を作らずに実行できる。

```bash
scripts/dev-docker.sh                                   # pytest 全体
scripts/dev-docker.sh python -m pytest tests/evaluation -q
scripts/dev-docker.sh python scripts/generate_benchmark.py --cases 100 --clean
scripts/dev-docker.sh python scripts/run_evaluation.py --dataset evaluation-data/generated --output evaluation-results
scripts/dev-docker.sh ruff check
```

初回は `Dockerfile.dev` をビルドし、MediaPipe モデルを `models/` へ取り出す。
生成物はホストの実行ユーザー所有になる。
