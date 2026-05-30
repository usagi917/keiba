# keiba-axis-model

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.en.md)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-4B5563)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> レース単位の入力バンドルから、Top 3 確率、軸馬、相手候補、レース後の振り返りレポートを生成する競馬予測ワークフローです。

## 概要

`keiba-axis-model` は、`races/<slug>/` ごとに出走表・追加履歴・レース条件・結果を管理し、週次の競馬予測と振り返りを同じ CLI で回すための Python プロジェクトです。

主な機能:

- `predict`: 累積学習履歴とレース個別履歴を結合し、Top 3 確率・勝率・順位分布を推定
- `recommended_axis_horse.json`: 軸馬候補 1 頭と選定に使うスコアを出力
- `recommended_partners.json` / `ticket_candidates.csv`: 軸馬に対する相手候補を出力
- `settle`: `result.csv` を取り込み、予測と実結果の差分レポートを生成
- `scripts/backtest.py`: market baseline と single ranker の walk-forward バックテストを実行

同梱サンプルレース:

- `races/hanshin-daishoten-2026-03-22/`
- `races/takamatsunomiya-kinen-2026-03-29/`
- `races/osaka-hai-2026-04-05/`
- `races/oka-sho-2026-04-12/`
- `races/satsuki-sho-2026-04-19/`

## 現状

この README は、2026-04-20 に次のコマンドで確認した内容に基づいています。

```bash
uv run python main.py --help
uv run python main.py list-races
uv run pytest
```

確認結果:

- `main.py` の CLI は `predict`, `settle`, `init-race`, `list-races` を提供
- `list-races` は同梱サンプル 5 レースを検出
- `uv run pytest`: 203 passed, 1 warning

## セットアップ

### 前提条件

- Python 3.10 以上
- `uv`

### インストール

```bash
uv sync
```

### よく使う開発コマンド

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

出力例:

```text
hanshin-daishoten-2026-03-22: Hanshin Daishoten (2026-03-22)
oka-sho-2026-04-12: Oka Sho (2026-04-12)
osaka-hai-2026-04-05: Osaka Hai (2026-04-05)
satsuki-sho-2026-04-19: Satsuki Sho (2026-04-19)
takamatsunomiya-kinen-2026-03-29: Takamatsunomiya Kinen (2026-03-29)
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

### 3. 予測する

```bash
uv run python main.py predict --race satsuki-sho-2026-04-19
```

省略形も使えます。

```bash
uv run python main.py satsuki-sho-2026-04-19
```

`predict` は次のデータを結合して学習データを作ります。

- `data/training/race_results_master.csv`
- `races/<slug>/history.csv`

対象レース自身の `race_id` と、対象日以降の履歴は予測用学習データから除外されます。

### 4. レース結果を反映する

```bash
uv run python main.py settle --race satsuki-sho-2026-04-19
```

`settle` は `races/<slug>/result.csv` を読み込みます。

- 常に `settled_entry.csv` と `post_race_report.json` を生成
- 予測結果が存在する場合のみ `post_race_analysis.csv` を生成
- `result.csv` が全頭分の結果なら `data/training/race_results_master.csv` を更新

### 5. バックテストする

```bash
uv run python scripts/backtest.py --data data/training/race_results_master.csv
```

任意で fold 数と最小学習レース数を指定できます。

```bash
uv run python scripts/backtest.py \
  --data data/training/race_results_master.csv \
  --n-splits 5 \
  --min-train-races 20
```

## CLI リファレンス

| コマンド | 役割 |
| --- | --- |
| `predict` | 指定レースの予測成果物を生成 |
| `settle` | `result.csv` を取り込み、レース後レポートと学習履歴を更新 |
| `init-race` | 新しい `races/<slug>/` の雛形を作成 |
| `list-races` | 利用可能なレースバンドルを一覧表示 |

| コマンド | オプション | 役割 |
| --- | --- | --- |
| `predict` / `settle` | `--race` | `races/` 配下の slug を指定 |
| `predict` / `settle` | `--race-dir` | レースディレクトリを直接指定 |
| `predict` / `settle` | `--races-root` | レースディレクトリの親を変更 |
| `predict` / `settle` | `--training-history` | 累積学習 CSV を差し替え |
| `predict` / `settle` | `--output-root` | 出力ルートを変更 |
| `predict` | `--config` | ベース設定ファイルを差し替え |
| `init-race` | `--race-name` | 表示名を指定 |
| `init-race` | `--race-date` | レース日を `YYYY-MM-DD` で指定 |
| `init-race` | `--force` | 既存の雛形ファイルを上書き |

## レースバンドル構成

各 `races/<slug>/` には次のファイルを置きます。

| ファイル | 必須 | 用途 |
| --- | --- | --- |
| `race.yaml` | 必須 | レース条件と設定上書き。`target_race_profile.surface`, `distance`, `course` は必須 |
| `entry.csv` | 必須 | 今回の出走馬データ |
| `history.csv` | 必須 | 今回レース向けに追加する履歴データ |
| `result.csv` | レース後に必須 | 着順と結果情報 |
| `race_result_meta.json` | 任意 | 勝ち時計、ラップ、馬場状態などの付加情報 |
| `README.md` | 任意 | レース固有のメモ |
| `refresh_race_data.py` | 任意 | レース固有のデータ取得・再生成スクリプト |

`result.csv` のテンプレート列:

```text
horse_id,horse_name,finish_rank,result_time,result_margin,result_last3f,result_final_odds,result_final_popularity,result_body_weight,result_body_weight_diff
```

## 設定

ベース設定は `src/keiba_predictor/config.yaml` で管理します。

主な設定領域:

- `similarity_weights`: 学習履歴の類似度重み
- `model.ranker`: CatBoost ranker の設定
- `model.classifier`: CatBoost classifier と scikit-learn 分類器の設定
- `model.regressor`: scikit-learn 回帰器の設定
- `cv`: 時系列 CV の fold 数と最小学習レース数
- `simulation`: 順位分布シミュレーションとキャリブレーション設定
- `partner`: 相手候補の選定数、人気制約、最適化重み

各レースの `race.yaml` では、`target_race_profile` と `use_odds` を上書きできます。

```yaml
race_name: Satsuki Sho
race_date: 2026-04-19
use_odds: true
target_race_profile:
  surface: turf
  distance: 2000
  course: Nakayama
```

## 出力ファイル

### `predict` が生成するもの

| ファイル | 内容 |
| --- | --- |
| `predictions.csv` | 全馬の予測テーブル。Top 3 確率、勝率、平均順位、軸スコア、相手候補スコアを含む |
| `recommended_axis_horse.json` | 推奨軸馬 1 頭の詳細 |
| `recommended_partners.json` | 軸馬に対する相手候補セット |
| `ticket_candidates.csv` | 馬券候補向けの軸馬・相手馬ペア |
| `evaluation_summary.json` | CV 指標とデータ診断の要約 |
| `cv_fold_metrics.csv` | fold ごとの評価結果 |
| `calibration_curve.csv` | Top 3 確率のキャリブレーション集計 |
| `feature_importance.csv` | 特徴量重要度ランキング |
| `simulation_diagnostics.csv` | シミュレーション収束診断 |
| `schema_report.json` | カラム正規化と特徴量候補の検証結果 |
| `effective_config.json` | ベース設定と `race.yaml` をマージした最終設定 |
| `run_context.json` | 入力ファイルと出力先の実行コンテキスト |
| `top3_probability_bar.png` | 上位候補の Top 3 / 勝利確率チャート |
| `calibration_plot.png` | キャリブレーション可視化 |
| `feature_importance.png` | 特徴量重要度の可視化 |

### `settle` が生成するもの

| ファイル | 内容 |
| --- | --- |
| `settled_entry.csv` | 結果を付与した出走表 |
| `post_race_report.json` | 上位予測、軸馬、相手候補と実結果の比較レポート |
| `post_race_analysis.csv` | 馬ごとの順位誤差分析。`predictions.csv` がある場合のみ生成 |

## モデルと前処理

- `data_loader.py`: CSV のエンコーディング吸収、列名エイリアス正規化、型変換、必須カラム検証
- `features.py`: 基本特徴量、距離帯・格・騎手・調教師・馬の集約特徴量、相対ランク特徴量
- `hybrid_model.py`: CatBoost ranker/classifier と scikit-learn の回帰・分類アンサンブル
- `simulation.py`: Plackett-Luce と Gaussian 系の順位分布推定、条件付き Top 3 確率
- `prediction.py`: 軸馬スコア、相手候補スコア、成果物書き出し
- `workflow.py`: `config.yaml` と `race.yaml` のマージ、予測と結果反映の orchestration

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
├── races/
├── scripts/
│   └── backtest.py
├── src/
│   └── keiba_predictor/
│       ├── config.yaml
│       ├── backtest_baseline.py
│       ├── data_loader.py
│       ├── features.py
│       ├── hybrid_model.py
│       ├── model.py
│       ├── prediction.py
│       ├── simulation.py
│       └── workflow.py
└── tests/
```

## ライセンス

このプロジェクトは MIT ライセンスで公開しています。詳細は [LICENSE](LICENSE) を参照してください。
