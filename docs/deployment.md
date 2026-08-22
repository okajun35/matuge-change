# デプロイと運用設定

## 公開サーバー

公開デモ: <https://matuge-change.onrender.com/>

画面はページごとに分かれている。

| URL | 画面 |
| --- | --- |
| `/` | 商品カタログ（登録済み商品の一覧・類似検索） |
| `/extract.html` | 静止画モード（まつ毛Alpha抽出・再合成） |
| `/video.html` | 動画モード（ベストフレーム選択・目元領域差し替え合成） |

## Supabase連携（任意）

環境変数が揃っているときだけ、カタログ、類似検索、ジョブ進捗がSupabase
（Postgres + pgvector + Storage + Realtime）に切り替わる。未設定なら `data/` 配下の
ローカル実装で同じ機能が動く。

```bash
export SUPABASE_URL=https://<project-ref>.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=...   # サーバー専用。ブラウザへ渡さない
export SUPABASE_PUBLISHABLE_KEY=...    # Realtime購読用
```

詳細は [supabase-phase-b.md](supabase-phase-b.md) を参照。

## Mattingのメモリ設定

Matting（closed-form solve）のピークメモリは解く画素数に比例し、実測で約3MB / 1000px。
依存ライブラリの常駐が約287MBあるため、Render Starterのような512MBホストではROI全体を
一度に解くとOOM（502）になる。既定は品質優先の `full` で、低メモリ環境だけ `tiled` に
切り替える。

| 環境変数 | 既定値 | 意味 |
| --- | --- | --- |
| `MATTE_SOLVE_MODE` | `full` | `full`はROI全体を一度にsolveする。`tiled`はsolve windowを分割する低メモリ近似 |
| `MATTE_MAX_SOLVE_PIXELS` | `60000` | tiledモードの1 solveあたりの画素上限。contextを含むsolver入力の絶対上限。`0`は無制限 |
| `MATTE_MAX_WORKERS` | `1` | 同期・非同期Mattingが共有するプロセス共通ゲートの同時実行数 |
| `MATTE_LOG_LEVEL` | `INFO` | Mattingログのレベル。`WARNING`以上では起動時・実行時の情報行を出さない |
| `MATTE_DETECT_MAX_SIDE` | `1600` | MediaPipeへ渡す検出用コピーの長辺上限。`0`なら縮小しない。抽出ROI自体は元解像度から切り出す |

負数、数値でない文字列、未対応のsolve modeは設定エラーとしてログへ出す。

512MB環境では次を設定する。

```bash
MATTE_SOLVE_MODE=tiled
MATTE_MAX_SOLVE_PIXELS=60000
MATTE_MAX_WORKERS=1
```

`tiled` はfull solveの近似であり完全一致しない。メモリに余裕がある環境では既定の `full` を
使う。生成時の設定は実行履歴（`GET /api/sessions/{id}/runs`）の `solve_mode` と
`max_solve_pixels` に残る。fullとtiledの比較値は [handover.md](handover.md) §6.6を参照。

### tiledのラベル探索

tiledの各solveにはFG/BG両ラベルが必要である。タイル内にラベルがなければ、画素上限内で
contextを広げ、次に縦長・横長の帯へ変形し、それでも届かなければタイルを分割して探索する。

上限が小さすぎる場合は `MATTE_MAX_SOLVE_PIXELS` を上げるか `full` を使うよう設定エラーを返す。
ROI全体を黙って解いてOOMへ戻ることも、Unknownを一律0/1にして境界を壊すこともしない。

## 512MBホストとnumbaキャッシュ

`pymatting` のnumba関数は初回importでJITコンパイルされ、一時的にRSSを約490MB使用する。
glibcがそのヒープをOSへ返さないため、キャッシュなしで起動したコンテナはリクエスト前から
常駐約575MBになり、512MBホストではOOM killされる。

`Dockerfile` はビルド時に `python -c "import pymatting"` を実行し、キャッシュをイメージへ
焼いている。本番ではこのイメージをそのまま使う。独自イメージや起動方法では同じwarm-upが必要。

numbaのキャッシュキーはCPU名を含む。ビルドホストと実行ホストのCPU差でキャッシュが無視されない
よう、`Dockerfile` は次の環境変数をビルド時・実行時の両方へ設定している。

```text
NUMBA_CPU_NAME=generic
NUMBA_CPU_FEATURES=""
```

この2つを削除したり、実行時に上書きしたりしない。

## ログによる設定確認

起動時、Matting実行時、共通ゲート待機時にstdoutへ次のようなログを出す。

```text
matte settings: solve_mode=tiled max_solve_pixels=60000 max_workers=1
matte run: solve_mode=tiled roi=1100x600 max_solve_pixels=60000 solves=24 max_solve_px=59400 elapsed_ms=8123
matte waiting for a matting slot: active=1 max_workers=1
```

- `matte settings`: 起動時の実効設定。設定値が不正なら `ERROR` を出す
- `matte run`: mode、ROI、solve回数、最大solver入力面積、所要時間
- `matte waiting for a matting slot`: 同時実行ゲートが空くまで待機している状態。処理失敗ではない

`max_solve_px` が設定した `max_solve_pixels` を超えていれば、tiledの上限が破れている。
