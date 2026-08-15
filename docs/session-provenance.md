# セッションの来歴保存（ローカル）

「元画像 / 合成元（AI 加工済み画像）/ 合成結果」と Matting の実行履歴をセッションディレクトリに
残す。認証（Auth/RLS）と Supabase Storage への移行は後回しにし、まずローカルで完結させる方針。

## 保存されるもの

```text
data/<session_id>/
  source_with.png        # アップロードされた装着画像そのもの
  source_without.png     # 未装着画像を渡したときのみ
  source_edited.png      # /api/recompose に渡した AI 加工済み画像
  roi_a.png / roi_b.png / difference.png / probability.png(.npy)
  trimap.png / alpha.png / product_rgba.png
  composite_on_bare.png / composite_on_edited.png
  landmarks.npy / meta.json
  runs.json              # Matting 実行履歴（追記）
```

これまでは ROI 切り出し以降しか残らず、アップロード画像と加工済み画像は破棄していたため、
同じ入力での再実行・パラメータ比較・監査ができなかった。

`source_edited.png` は再合成の成否に関わらず保存する（顔検出に失敗した入力も残す）。

## 実行履歴 `runs.json`

`SessionService.run_matte()` が成功するたびに 1 件追記する。同期 API と非同期ジョブの
どちらから呼んでも同じ経路を通るので履歴は共通。

```json
[
  {
    "created_at": "2026-08-14T22:10:31.482913+00:00",
    "params": { "fg_thresh": 0.7, "bg_thresh": 0.18, "unknown_band_px": 6 },
    "reconstruction_error": 0.0412,
    "layers": ["trimap", "alpha", "product_rgba"]
  }
]
```

## API

- `GET /api/sessions/{session_id}` … 保存済みレイヤーに `source_with` / `source_without` /
  `source_edited` が加わる（存在するものだけ）。UI のレイヤー選択からも参照・ダウンロードできる。
- `GET /api/sessions/{session_id}/runs` … `{"runs": [...]}`。未実行なら空配列、
  未知のセッションは 404。

## 実装メモ

- 追記は `SessionStore.append_run()` / `load_runs()` に閉じてあり、保存先を Supabase
  （`matte_runs` テーブル + Storage）へ差し替えるときはこの 2 メソッドと `save_image()` の
  実装を置き換えるだけで済む。
- TDD: `tests/sessions/test_artifacts.py`（保存・履歴のドメインテスト）と
  `tests/api/test_sessions_api.py::TestRunHistoryApi`（API）を Red → Green の順で追加した。

## カタログのページング

商品が増えると一覧が縦に伸び、抽出結果（`#stage`）が画面下に押し出されていた。カタログを
`<details>` で折りたたみ、サーバー側ページングと名前 / ブランド検索を入れた。

- `GET /api/assets?limit=12&offset=0&q=` → `{"assets": [...], "total": n, "limit": ..., "offset": ...}`
- ローカルは JSON インデックスを新しい順に並べて窓を切り出し、Supabase は
  `select(count="exact").or_(name.ilike/brand.ilike).range(offset, offset+limit-1)`
- UI は 12 件/ページ、前へ / 次へ、検索は 250ms デバウンス。末尾ページが空になったら 1 ページ戻る

## 今後（認証と Storage 移行）

- 顔画像・AI 加工画像は機微データなので、Auth 導入時に所有者単位で `data/` 相当を分離し、
  画像配信 API に所有権チェックを入れる。
- Storage へ移行する際は private bucket + ownership policy、`runs.json` は `matte_runs`
  テーブルへ。保持期間・削除 API もそのタイミングで用意する。
