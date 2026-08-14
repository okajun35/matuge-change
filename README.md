# matuge-change

商品まつ毛保持型 AIモデル画像加工システムの PoC。

まつ毛（商品＋商品に隠された/動かされた自まつ毛を含む「装着後の目元まつ毛外観」）を
Alpha Matte として抽出し、後段の AI モデル加工画像へ再合成することを目標とする。

## PoC の範囲 (Phase 1–2)

- 装着画像（＋任意で未装着画像）から目元 ROI を自動検出
- 未装着画像がある場合: 位置合わせ（Landmark Affine + ECC）→ Difference Map
- 未装着画像がない場合: 局所暗部検出によるフォールバック
- 初期 Probability 推定 → ユーザーがブラシ（＋商品 / −背景）で補正
- Soft Trimap → Closed-form Alpha Matting + Foreground 推定 (pymatting)
- Product RGBA 出力・未装着画像への再合成・再合成誤差の表示

## セットアップ

```bash
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install -r requirements.txt
curl -sL -o models/face_landmarker.task --create-dirs \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## 起動

```bash
. .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

ブラウザで http://localhost:8000 を開く。

## 使い方

1. 装着画像（必要なら未装着画像も）をアップロードして「解析開始」
2. 表示レイヤーを Probability に切り替えて初期推定を確認
3. ＋商品 / −背景 ブラシで必要な箇所だけ補正
4. 「Matting実行」→ Alpha / Product RGBA / 未装着画像への再合成を確認
5. FG/BG 閾値スライダーで Trimap を調整して再実行可能
