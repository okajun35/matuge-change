# AI モデル加工（Model Editing）の実現方法

設計書 v2 §18 / §28 に対応する検討メモ。PoC で扱う「まつ毛抽出 → 再合成」のうち、
中間工程である **モデル側の AI 加工** をどう実現するかを整理する。

## 結論

- 加工そのものは **外部の画像編集 API** に任せる。ローカルで生成モデルを動かす必要はない。
- 特定モデルに依存しないよう **Adapter 構造**（`EditingBackend` インターフェース）にする。
- どのバックエンドでも共通で、**目元は保護マスクで編集対象から除外**する。
  これが商品保持（Generative Product Preserve＝生成AIに商品を描き直させない）の前提になる。

## バックエンド候補

| 候補 | 特徴 | この用途での評価 |
|---|---|---|
| Gemini 画像編集（`gemini-2.5-flash-image` 系、通称 nano-banana） | 自然言語の指示ベース編集。人物の同一性保持が比較的強い。マスク指定は指示文＋参照画像に依存 | 第一候補。肌・背景・髪・色調といった Safe Editing 用途に向く |
| FLUX Kontext | 参照画像を保ったままの指示編集。inpainting 系のワークフローが組みやすい | 第二候補。マスクを厳密に効かせたい場合に有利 |
| Stable Diffusion 系 inpainting（SDXL inpaint など） | マスクを厳密に適用できる。自前ホスティングも可能 | マスク厳守が最重要なら有力。品質はプロンプト・モデル依存 |

いずれも API 経由で置き換え可能な形にしておき、実写での比較で決める。

## Adapter インターフェース案

```python
class EditingBackend(Protocol):
    def edit(
        self,
        image_bgr: np.ndarray,
        instruction: str,
        protect_mask: np.ndarray | None,  # uint8 255=保護（編集禁止）
    ) -> np.ndarray: ...
```

- マスク対応バックエンド（inpainting 系）: `protect_mask` の反転を編集領域として渡す。
- マスク非対応バックエンド（指示ベース）: 編集後に `protect_mask` の領域だけ元画像へ戻す
  （後段の再合成でまつ毛は上書きされるため、境界のみ注意すればよい）。

```python
edited = backend.edit(worn_bgr, instruction, protect_mask)
m = protect_mask[..., None] / 255.0
edited = (m * worn_bgr + (1 - m) * edited).astype(np.uint8)  # 保護領域を復元
```

## 保護マスクの作り方

既存パイプラインの Alpha をそのまま流用できる。

```python
alpha  # 抽出済み Alpha（ROI 座標）
mask = (alpha > 0.02).astype(np.uint8) * 255
mask = cv2.dilate(mask, ellipse(k=alpha.shape[1] * 0.03))  # 根元・まぶた・目尻を含める
protect_mask = paste_roi_into_full_image(mask, roi)
```

膨張量が小さいと根元が AI に描き替えられ、大きすぎると目元の加工が一切効かなくなる。
実写で 2〜5% 程度から調整する。

## 現行パイプラインへの接続位置

```text
装着画像 A
  ├─(既存) まつ毛抽出 → alpha / product_rgba
  │                        └→ 膨張 → protect_mask
  └→ EditingBackend.edit(A, instruction, protect_mask) → edited
                                                          ↓
                       (既存) pipeline.recompose_onto(product_rgba, roi, lms_worn, edited)
                                                          ↓
                                                    final composite
```

現在の UI では「編集済み画像」を手動アップロードしているが、この Adapter を追加すれば
UI 内で加工 → 再合成まで一気通貫にできる。

## 未確定事項

- API キーの取り扱い（環境変数 + サーバー側保持。フロントには出さない）。
- 課金・レート制限とリトライ方針。
- どのバックエンドが目元付近の改変を最も抑えられるか（実写比較が必要）。
- 構造編集（目の形状変更など、設計書 §33 Structural Editing）は本メモの対象外。
