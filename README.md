# keiba-axis-model

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.en.md)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-4B5563)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> レース単位の入力ディレクトリを起点に、Top 3 確率、軸馬候補、レース後の振り返りレポートを生成する競馬予測ワークフローです。

## 概要

このリポジトリは `races/<slug>/` ごとにレース入力を管理し、`main.py` から週次運用を回します。

- `list-races`: 利用可能なレースバンドルを列挙
- `init-race`: 新しいレースディレクトリを雛形付きで生成
- `predict`: 累積学習履歴とレース個別履歴を結合して予測成果物を出力
- `settle`: `result.csv` を取り込み、結果レポートと累積学習履歴を更新

## 現状

この README は、2026-03-29 時点で次を実地確認した内容に合わせています。

- `uv run pytest`: 68 passed
- `uv run python main.py --help`
- `uv run python main.py list-races`
- `uv run python main.py init-race ...`
- `uv run python main.py settle ...`

`predict` コマンド自体は CLI に存在しますが、現行の同梱サンプルデータでは CatBoost ranker が `Groupwise loss/metrics require nontrivial groups` で失敗しました。`outputs/` 以下の成果物は期待される出力形式の参考として扱ってください。

## セットアップ

### 前提条件

- Python 3.10 以上
- `uv`

### インストール

```bash
uv sync
```

### 開発用コマンド

```bash
uv run python main.py --help
uv run python main.py list-races
uv run pytest
```

## クイックスタート

### 1. レース一覧を確認する

```bash
uv run python main.py list-races
```

### 2. 新しいレースバンドルを作る

```bash
uv run python main.py init-race \
  --race spring-tenno-sho-2026-05-03 \
  --race-name "Tenno Sho Spring" \
  --race-date 2026-05-03
```

生成されるファイル:

- `races/spring-tenno-sho-2026-05-03/race.yaml`
- `races/spring-tenno-sho-2026-05-03/entry.csv`
- `races/spring-tenno-sho-2026-05-03/history.csv`
- `races/spring-tenno-sho-2026-05-03/result.csv`
- `races/spring-tenno-sho-2026-05-03/README.md`

### 3. 予測コマンドの形式

```bash
uv run python main.py predict --race hanshin-daishoten-2026-03-22
```

省略形も使えます。

```bash
uv run python main.py hanshin-daishoten-2026-03-22
```

補足:

- `predict` は `data/training/race_results_master.csv` と `races/<slug>/history.csv` を結合して学習データを作ります
- 対象レース自身の `race_id` と対象日以降の履歴は自動で除外されます
- 現行サンプルデータではこのコマンドは失敗するため、成果物の種類は `outputs/` 以下の既存ファイルを参照してください

### 4. レース結果を反映する

```bash
uv run python main.py settle --race takamatsunomiya-kinen-2026-03-29
```

`settle` は `predictions.csv` がなくても実行できます。

- 常に `settled_entry.csv` と `post_race_report.json` を生成
- 予測結果が存在する場合だけ `post_race_analysis.csv` も生成
- `result.csv` がフル結果なら `data/training/race_results_master.csv` を更新

### 5. よく使うオプション


| コマンド                 | オプション                | 役割                    |
| -------------------- | -------------------- | --------------------- |
| `predict` / `settle` | `--race`             | `races/` 配下の slug を指定 |
| `predict` / `settle` | `--race-dir`         | レースディレクトリを直接指定        |
| `predict` / `settle` | `--training-history` | 累積学習 CSV を差し替え        |
| `predict` / `settle` | `--output-root`      | 出力ルートを変更              |
| `predict`            | `--config`           | ベース設定ファイルを差し替え        |
| `init-race`          | `--races-root`       | レースルートを変更             |
| `init-race`          | `--force`            | 雛形ファイルを上書き            |


## レースバンドル構成

各 `races/<slug>/` には次のファイルを置きます。


| ファイル                    | 必須   | 用途                                                                                                              |
| ----------------------- | ---- | --------------------------------------------------------------------------------------------------------------- |
| `race.yaml`             | 必須   | レース条件と設定上書き。`target_race_profile.surface` / `distance` / `course` は必須。`use_odds` で odds/popularity 系特徴量の利用を切り替え |
| `entry.csv`             | 必須   | 今回の出走馬データ                                                                                                       |
| `history.csv`           | 必須   | 今回レース向けに追加する履歴データ                                                                                               |
| `result.csv`            | 事後必須 | レース後の着順と結果情報                                                                                                    |
| `race_result_meta.json` | 任意   | 勝ち時計、ラップ、馬場状態などの付加情報                                                                                            |
| `README.md`             | 任意   | レース固有のメモ                                                                                                        |
| `refresh_race_data.py`  | 任意   | データ取得・再生成スクリプト                                                                                                  |


`result.csv` のテンプレート列は次の通りです。

```text
horse_id,horse_name,finish_rank,result_time,result_margin,result_last3f,result_final_odds,result_final_popularity,result_body_weight,result_body_weight_diff
```

## 出力ファイル

### `predict` が生成するもの


| ファイル                          | 内容                            |
| ----------------------------- | ----------------------------- |
| `predictions.csv`             | 全馬の予測テーブル                     |
| `recommended_axis_horse.json` | 推奨軸馬 1 頭の詳細                   |
| `evaluation_summary.json`     | CV 指標とデータ診断の要約                |
| `cv_fold_metrics.csv`         | Fold ごとの評価結果                  |
| `calibration_curve.csv`       | Top 3 確率のキャリブレーション集計          |
| `feature_importance.csv`      | 特徴量重要度ランキング                   |
| `simulation_diagnostics.csv`  | シミュレーション収束診断                  |
| `schema_report.json`          | カラム正規化と特徴量候補の検証結果             |
| `effective_config.json`       | ベース設定と `race.yaml` をマージした最終設定 |
| `run_context.json`            | 入力ファイルと出力先の実行コンテキスト           |
| `top3_probability_bar.png`    | 上位候補の Top 3 / 勝利確率ダッシュボード     |
| `calibration_plot.png`        | キャリブレーション可視化                  |
| `feature_importance.png`      | 特徴量重要度の可視化                    |


### `settle` が生成するもの


| ファイル                     | 内容                                     |
| ------------------------ | -------------------------------------- |
| `settled_entry.csv`      | 結果を付与した出走表                             |
| `post_race_report.json`  | 上位予測と実結果の比較レポート                        |
| `post_race_analysis.csv` | 馬ごとの順位誤差分析。`predictions.csv` がある場合のみ生成 |


## モデルと前処理

- `data_loader.py` が CSV のエンコーディング吸収、列名エイリアス正規化、型変換、必須カラム検証を担当
- `features.py` が基本特徴量、距離帯・格・騎手/調教師/馬の集約特徴量、相対ランク特徴量を生成
- `hybrid_model.py` が CatBoost ranker と CatBoost classifier、scikit-learn の回帰/分類アンサンブルを組み合わせる
- `simulation.py` が Plackett-Luce / Gaussian ベースの順位分布推定を行う
- `workflow.py` が `config.yaml` と `race.yaml` をマージし、予測と結果反映を束ねる

## ディレクトリ構成

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── README.en.md
├── data/
│   └── training/
│       └── race_results_master.csv
├── outputs/
│   ├── Hanshin Daishoten/
│   └── Takamatsunomiya Kinen/
├── races/
│   ├── hanshin-daishoten-2026-03-22/
│   └── takamatsunomiya-kinen-2026-03-29/
├── src/
│   └── keiba_predictor/
│       ├── config.yaml
│       ├── data_loader.py
│       ├── features.py
│       ├── hybrid_model.py
│       ├── model.py
│       ├── prediction.py
│       ├── simulation.py
│       └── workflow.py
└── tests/
    ├── test_data_loader.py
    ├── test_features.py
    ├── test_hybrid_model.py
    ├── test_model.py
    └── test_workflow.py
```

## ライセンス

このプロジェクトは MIT ライセンスで公開しています。詳細は [LICENSE](LICENSE) を参照してください。
