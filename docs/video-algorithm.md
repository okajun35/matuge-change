# 動画モードの目元保持・再合成アルゴリズム

この文書は、`matuge-change` の動画モードで、元動画の目元を保持しながら AI 加工済み人物画像へ再合成する現行アルゴリズムを説明する。

対象コードは主に `backend/video.py`。

関連する設計判断や却下案は [video-approach.md](video-approach.md) を参照。

## 1. 静止画モードとの違い

動画モードでは、静止画モードのように商品まつ毛を Alpha Matte として抽出しない。

静止画モードでは、

```text
Difference
  ↓
Probability
  ↓
Trimap
  ↓
Alpha Matting
  ↓
Product RGBA
```

という処理を行う。

しかし、この処理を動画の全フレームへ独立に適用すると、

- 同じ毛先でもフレームごとに Alpha 値が揺れる
- まばたき中はまぶたの形状差が Difference を支配する
- フレーム間で抽出形状が変化し、ちらつく
- 毎フレーム Matting すると計算量が大きい

という問題が起きる。

そのため動画モードでは、**商品まつ毛だけを抽出することをやめ、元動画の目元領域そのものを保持する**。

```text
商品まつ毛
+
自まつ毛
+
まぶた
+
眼球
+
まばたきの動き
+
目元周辺の一部の肌
```

を元フレームからそのまま使う。

## 2. 全体フロー

```text
元動画
  │
  ├─ フレーム分解
  │
  ├─ 全フレームで MediaPipe Face Landmarker
  │
  └─ Best Frame Selection
       ├─ Sharpness
       └─ Eye Openness
              ↓
        best_frame.png
              ↓
       外部生成AIで人物加工
              ↓
       AI加工済み静止画
              ↓
        MediaPipe Landmarks
              ↓
┌─────────────────────────────┐
│ 各元動画フレームについて    │
│                             │
│ AI加工画像                   │
│    ↓                        │
│ Landmark Affine             │
│    ↓                        │
│ 元フレーム座標へ warp       │
│                             │
│ 元フレーム                  │
│    ↓                        │
│ Eye Region Soft Mask        │
│                             │
│ original eye region         │
│          +                  │
│ warped edited image         │
│          ↓                  │
│       Alpha Blend           │
└─────────────────────────────┘
              ↓
      全フレームを書き出し
              ↓
       H.264 / yuv420p
              ↓
          output.mp4
```

## 3. 動画フレームの読み込み

実装: `backend/video.py::read_video_frames`

OpenCV の `cv2.VideoCapture` で動画を読み込む。

```python
MAX_FRAMES = 600
```

現行実装では最大 600 フレームまで処理する。

FPS は入力動画から取得し、取得できない場合は 30fps を利用する。

## 4. 全フレームの顔ランドマーク

各フレームについて MediaPipe Face Landmarker で顔ランドマークを取得する。

動画専用のトラッキングモデルを使うのではなく、静止画側と同じ顔ランドマーク検出を各フレームへ適用する。

このランドマークは主に次の用途に使う。

- 目の開き具合
- ベストフレーム選択
- 目元 Soft Mask
- AI加工済み画像の Affine 位置合わせ

## 5. ベストフレーム選択

実装: `backend/video.py::select_best_frame`

元動画から、AI加工へ渡す1枚を選ぶ。

現行実装では次の2指標を利用する。

### 5.1 Sharpness

```python
cv2.Laplacian(gray, cv2.CV_64F).var()
```

Laplacian 分散を使い、ピントやブレの少なさを評価する。

値が高いほどシャープなフレームとみなす。

### 5.2 Eye Openness

左右の目について、

```text
上下のまぶた間距離
────────────────
       目の横幅
```

を求める。

使用ランドマークは MediaPipe Face Mesh の目周辺点。

右目:

```text
横幅: 33 - 133
上下: 159 - 145
      158 - 153
```

左目:

```text
横幅: 362 - 263
上下: 386 - 374
      385 - 380
```

左右の比率の平均を `eye_openness` とする。

### 5.3 スコア

Sharpness と Eye Openness を、それぞれ有効フレーム内の最大値で正規化する。

```text
score =
    0.5 × normalized_sharpness
  + 0.5 × normalized_eye_openness
```

最大スコアのフレームを Best Frame とする。

### 5.4 ドキュメントとの差異

`docs/video-approach.md` には `face_angle` と `exposure` も指標候補として記載されているが、**現行 `backend/video.py` で実際に Best Frame 選択へ利用しているのは Sharpness と Eye Openness の2つだけ**である。

## 6. AI加工

Best Frame を画像として出力し、外部の生成AIで人物側を加工する。

想定フロー:

```text
Best Frame
    ↓
外部画像編集AI
    ↓
人物・髪・肌・背景などを変更
    ↓
AI加工済み静止画
```

目元は後から元動画へ戻すため、運用上は目元周辺を大きく変更しないプロンプトを推奨する。

動画を生成AIでフレームごとに編集する方式ではない。

## 7. 目元 Soft Mask

実装: `backend/video.py::eye_region_mask`

動画では商品まつ毛だけを抽出せず、左右の目を含む領域を Soft Mask として作る。

### 7.1 目ポリゴン

MediaPipe の左右の目ランドマークを `cv2.fillPoly` で塗りつぶす。

### 7.2 拡張

左右の目幅の平均を求める。

デフォルト:

```python
expand = 0.45
```

拡張半径は、

```text
radius = 0.45 × average_eye_width
```

とする。

最小値は 2px。

楕円カーネルを使って目領域を dilate する。

この拡張により、まつ毛が目輪郭より外側へ伸びる領域まで含める。

### 7.3 Feather

拡張後のマスクを Gaussian Blur する。

```text
feather_sigma = radius × 0.35
```

現行実装では blur 後に `× 1.2` し、`[0, 1]` に clip する。

結果は、

```text
mask = 1.0
```

に近い中央領域ほど元動画を強く残し、

```text
0 < mask < 1
```

の境界部では AI加工画像と混色する Soft Mask になる。

## 8. AI加工済み画像の位置合わせ

実装: `backend/video.py::warp_edited_to_frame`

重要な設計原則は、**元動画側を warp しないこと**。

各フレームの顔ランドマークと、AI加工済み画像のランドマークを対応させる。

```python
src = lms_edited[ALIGN_POINTS]
dst = lms_frame[ALIGN_POINTS]
```

`cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)` で AI加工済み画像から元フレームへの変換を推定する。

```text
AI加工済み画像
       ↓
estimateAffinePartial2D
       ↓
frame N の顔位置へ
       ↓
warpAffine
```

変形するのは AI加工済み画像側である。

元動画の目元領域は元フレーム座標のまま使う。

`estimateAffinePartial2D` が失敗した場合は Identity Matrix を利用する。

## 9. フレームごとの合成

実装: `backend/video.py::blend_with_mask`

各フレームについて、

```text
original = 元動画フレーム
edited   = 当該フレーム座標へ warp した AI加工画像
mask     = 元フレームの目元 Soft Mask
```

を使う。

合成式は次の通り。

```text
out =
    mask × original
  + (1 - mask) × edited
```

### mask = 1

元動画のピクセルをそのまま使う。

```text
out = original
```

### mask = 0

AI加工済み画像を使う。

```text
out = edited
```

### 0 < mask < 1

境界で元動画と AI加工画像を混色する。

この方式では、目元コア領域の商品ピクセルには `warpAffine` を掛けない。

## 10. まばたきの扱い

動画版では、まばたき形状を推定・生成・変形しない。

各フレームの元目元領域自体を使うため、

```text
開眼
 ↓
半閉じ
 ↓
閉眼
 ↓
開眼
```

という変化はすべて元動画に含まれている。

したがって、

- まつ毛レイヤーを縦圧縮する
- 根元を軸に回転する
- フレームごとに Alpha Matting する

といった処理は不要。

この点が、「ベストフレームの Product RGBA を全フレームへ warp する案」を採用しなかった最大の理由である。

## 11. ランドマーク検出失敗時の fallback

実装: `backend/video.py::compose_frames`

フレーム N で顔ランドマークを取得できなかった場合、

```python
if lms is None:
    lms = last_lms
```

として、直前に取得できたランドマークを再利用する。

```text
frame 10: landmark OK
frame 11: landmark NG → frame 10 の landmark
frame 12: landmark OK
```

動画冒頭などで、まだ一度もランドマークが取得できていない場合は、そのフレームを加工せず元フレームのまま出力する。

これは簡易的な Temporal fallback であり、本格的なランドマークトラッキングではない。

## 12. 動画出力

実装: `backend/video.py::write_video`

まず OpenCV `VideoWriter` で、

```text
mp4v
```

として一時 MP4 を生成する。

実行環境に `ffmpeg` がある場合は、その後、

```text
codec:      libx264
pixel fmt:  yuv420p
movflags:   +faststart
```

で再エンコードする。

`ffmpeg` が利用できない場合は OpenCV が生成した MP4 をそのまま出力する。

## 13. 商品保持という観点での意味

動画モードでは、静止画よりも「商品保持」の定義が強い。

目元コア領域では、

```text
元動画ピクセル
    ↓
そのまま合成
```

となり、商品側には幾何変換を掛けない。

ただし最終出力が元ピクセルと完全一致するわけではない。

理由:

- Soft Mask の境界では AI画像と混色する
- MP4 出力時に再エンコードする
- `yuv420p` では色差情報がクロマサブサンプリングされる
- H.264 の量子化による変化がある

したがって、

**合成段階では商品コア領域を無変形で再利用するが、最終 MP4 のビット単位一致は保証しない**

という表現が正確である。

## 14. 現行方式の弱点

### 14.1 元人物の目元が残る

商品まつ毛だけでなく、

- 目
- まぶた
- 目元の肌

まで残る。

そのため AI加工で、

- 目の形
- 肌色
- 年齢
- 顔の骨格
- メイク

を大きく変更すると境界が不自然になりやすい。

### 14.2 目元 Mask は意味的セグメンテーションではない

MediaPipe の目ポリゴンを拡張しているだけであり、

「ここまでが商品まつ毛」

を認識しているわけではない。

### 14.3 顔ランドマーク依存

大きな横顔、遮蔽、極端なブレでは MediaPipe の検出が失敗する可能性がある。

現在の fallback は前フレーム landmark の再利用のみ。

### 14.4 AI加工済み画像は1枚

元動画に表情変化が大きい場合、1枚の AI加工済み静止画を Affine だけで全フレームへ合わせると、口や頬など目元以外の領域で不自然さが出る。

現行方式は主に、

**顔をほぼ固定し、まばたき程度の動きがある動画**

を対象としている。

## 15. 不採用方式

### 15.1 毎フレーム Alpha Matting

不採用理由:

- Alpha の時間方向のちらつき
- まばたきが Difference を支配
- 高コスト
- 商品形状のフレーム間一貫性が弱い

### 15.2 ベストフレームの商品 RGBA をまばたきに追従させる

不採用理由:

- 根元回転
- 縦圧縮
- 非相似 warp

などが必要になり、商品形状そのものを変更してしまう。

Generative Product Preserve の「商品領域には非相似変形を掛けない」という方針に反する。

## 16. コード対応表

| 処理 | 実装 |
| --- | --- |
| Eye Openness | `backend/video.py::eye_openness` |
| Sharpness | `backend/video.py::laplacian_sharpness` |
| Best Frame | `backend/video.py::select_best_frame` |
| Eye Region Mask | `backend/video.py::eye_region_mask` |
| AI画像の位置合わせ | `backend/video.py::warp_edited_to_frame` |
| Alpha Blend | `backend/video.py::blend_with_mask` |
| 全フレーム合成 | `backend/video.py::compose_frames` |
| 動画読込 | `backend/video.py::read_video_frames` |
| MP4出力 | `backend/video.py::write_video` |

## 17. 関連ドキュメント

- [../README.md](../README.md) — プロジェクト全体の目的と静止画 / 動画比較
- [static-image-algorithm.md](static-image-algorithm.md) — 静止画の商品まつ毛抽出・再合成
- [video-approach.md](video-approach.md) — 動画方式を採用した背景・却下案
- [video-expression-matting.md](video-expression-matting.md) — 表情変化が大きい動画への発展案
- [design-philosophy.md](design-philosophy.md) — なぜ商品を生成AIに任せないのか
- [handover.md](handover.md) — 設計判断と既知の落とし穴
