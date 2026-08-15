---
name: testing-lash-poc
description: How to run and UI-test the matuge-change lash alpha extraction PoC (FastAPI + single-page canvas app)
---

# Testing the lash alpha extraction PoC

## Start the server
- `. .venv/bin/activate && uvicorn backend.app:app --host 0.0.0.0 --port 8000` (venv created via `uv venv --python 3.11 .venv`, deps via `uv pip install -r requirements.txt`).
- MediaPipe model must exist at `models/face_landmarker.task` (downloaded in blueprint initialize step).
- UI at http://localhost:8000 — one page, Japanese labels, no auth.

## UI testing tips
- File inputs (装着画像 / 未装着画像) open the GTK file dialog; use Ctrl+L then type the absolute path.
- Flow: upload → 解析開始 (~3s) → layer dropdown gets Original/Aligned/Difference/Probability → optional brush strokes (＋商品 = red / −背景 = blue) → Matting実行 (~1-2s) → status shows 再合成誤差 (typically ~6-8 for the sample photo pair) and Trimap/Alpha/Product RGBA/未装着画像へ再合成 layers appear.
- Brush strokes are only sent to /api/matte if any non-transparent pixel exists on the paint canvas; ブラシ消去 clears them (verify by 再合成誤差 returning to the unconstrained value on re-run).
- To verify brush constraints: view Alpha as baseline, paint −背景 over a white lash area, re-run matting — that area should turn black and 再合成誤差 changes.
- Threshold sliders (FG閾値/BG閾値) only take effect on the next Matting実行; compare Trimap before/after.
- Edge cases: worn image only (no bare) → darkness-based Probability, no Aligned/再合成 layers; non-face image → status shows `エラー: {"detail":"no face detected in the worn image"}` (HTTP 422).

## Recompose feature (commits 5340c84+)
- Extra controls: 編集済み画像 file input + 再合成 button + 表示中レイヤーを保存 download link. Flow: new session → Matting実行 → select edited image → 再合成 → layer 「編集済み画像へ再合成」 auto-added and auto-shown; download link href tracks the visible layer (`/api/image/<session>/<layer>`).
- Requires a NEW session: /api/session now saves landmarks.npy; old sessions without it break /api/recompose.
- Error cases: 再合成 before matting → status shows `エラー: {"detail":"run matting first"}` (409); no face in edited image → `no face detected in the edited image` (422).
- Pitfall (fixed in 0a723a4): re-running Matting実行 used to drop 「編集済み画像へ再合成」 from the dropdown; now it is preserved.

## Catalog page (`/` = frontend/index.html)
- `/` is the catalog (静止画モードは /extract.html、動画は /video.html). Cards come from `GET /api/assets?limit=12&offset=..`; pytest runs leave plenty of dummy 48x48 assets (Japanese names) so the page is testable without extracting anything.
- Each card has `.asset-actions` with two `<a download>` links: 「RGBA」→ `/api/assets/{id}/image` (inline, alpha-preserving 4ch PNG) and 「マスク」→ `/api/assets/{id}/mask` (attachment, 1ch grayscale = the alpha channel). Chrome's `download` attribute wins over Content-Disposition, so files land in `~/Downloads` as `<商品名>.png` / `<商品名>-mask.png`.
- Verify downloads on disk, e.g. `cv2.imread(path, cv2.IMREAD_UNCHANGED)` → RGBA must be 4ch with alpha not all 255; mask must be 2-dim and `(mask == rgba[...,3]).all()`.
- Card click runs 類似検索 (status line 「… に近い形状: …」); clicking a link must NOT change the status (guard `e.target.tagName !== 'A'`).
- Unknown id: `GET /api/assets/<bogus>/mask` → 404 `{"detail":"asset <id> not found"}`.
- IMPORTANT: uvicorn does not auto-reload — after pulling new routes (e.g. `/mask`) restart the server or the endpoint 404s and you will misdiagnose a bug. Start it in a persistent shell (`uvicorn backend.app:app --host 0.0.0.0 --port 8000`); backgrounding with `nohup ... &` from a one-shot shell has died on this box.
- Cosmetic (pre-existing, not a PR regression): cards get `class="asset checker"` but `.asset { background: #2a2a33 }` is declared after `.checker`, so the checkerboard is never visible.

## Known pitfall (fixed in commit 2f1cd32)
- Older revisions had a bug: after viewing the Product RGBA layer, switching to any other layer rendered nothing (checker background stayed), because `showLayer` renamed the stage div id to `checker` and later `document.getElementById('stage')` returned null. Fixed by keeping a `stage` variable reference and toggling a `.checker` class. If testing an old revision, reload the page (F5) to recover and view Product RGBA last.
