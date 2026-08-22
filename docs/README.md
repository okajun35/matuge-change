# 技術ドキュメント

READMEにはプロジェクトの概要と最短の利用手順だけを置き、詳細は目的別にこのページから参照する。

## プロジェクトと設計思想

- [project-background.md](project-background.md) — EC販売の課題、対象商品、生成AIだけでは解決できなかった理由
- [design-philosophy.md](design-philosophy.md) — Generative Product Preserveの定義、商品領域に許す操作、保証の限界
- [generative-ai-cutout-failure.md](generative-ai-cutout-failure.md) — 汎用の画像生成AIへ切り抜きを任せた定性的な失敗例
- [ai-editing-api.md](ai-editing-api.md) — 生成AIを人物側のModel Editingに限定するAdapter設計

## 静止画

- [static-image-algorithm.md](static-image-algorithm.md) — ROI、Evidence、Trimap、Alpha Matting、再合成の全処理
- [static-image-simple-mode.md](static-image-simple-mode.md) — 2画像から一括処理する簡易モードのUI仕様

## 動画

- [video-algorithm.md](video-algorithm.md) — ベストフレーム選択、顔追従、目元保持、動画合成の全処理
- [video-approach.md](video-approach.md) — 顔固定・まばたき動画への方針、スタビライズ、却下案
- [video-expression-matting.md](video-expression-matting.md) — 表情変化が大きい動画への発展案

## 品質評価

- [../evaluation/README.md](../evaluation/README.md) — Synthetic Benchmarkの目的、生成方法、指標、限界、数値の読み方
- [benchmark-findings.md](benchmark-findings.md) — 現行アルゴリズムの実測課題と改善優先順位

## 実装・運用

- [development.md](development.md) — Docker、ローカルPython環境、テスト、Lint
- [deployment.md](deployment.md) — Supabase、低メモリMatting、numbaキャッシュ、運用ログ
- [session-provenance.md](session-provenance.md) — 入力、生成物、実行履歴の保存
- [supabase-phase-b.md](supabase-phase-b.md) — カタログ、類似検索、ジョブ進捗のSupabase連携
- [devin-supabase-usage.md](devin-supabase-usage.md) — Devinによる開発履歴とSupabaseの用途

## 開発エージェント向け

- [../AGENTS.md](../AGENTS.md) — このリポジトリで守る開発ルール
- [handover.md](handover.md) — 採用・却下した方式、既知の落とし穴、未検証事項
