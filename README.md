# keiba-axis-model

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.en.md)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-4B5563)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 時系列アンサンブルモデルと Plackett-Luce シミュレーションで、レースごとの 3 着内確率と軸馬候補を出力する競馬予測ワークフローです。

## 概要

このリポジトリは、`races/<slug>/` ごとにレース単位の入力データを管理し、`predict` で予測成果物を生成し、`settle` で結果を確定して学習履歴へ反映します。

- `predict`: `predictions.csv`、`recommended_axis_horse.json`、CV 指標、特徴量重要度、シミュレーション診断を出力
- `settle`: `settled_entry.csv` と `post_race_report.json` を出力し、フル結果なら `data/training/race_results_master.csv` を更新
- `init-race`: 新しいレースディレクトリの雛形を作成

サンプル用のレース定義として `races/hanshin-daishoten-2026-03-22/` が含まれています。予測を実行すると、対応する成果物が `outputs/Hanshin Daishoten/` に生成されます。

## セットアップ

### 前提条件

- Python 3.10 以上
- `uv`

### インストール

```bash
uv sync
```

### 動作確認

```bash
uv run python main.py --help
uv run python main.py list-races
```

## ユーザーフロー

```mermaid
flowchart TD
    A[開始] --> B{対象レースは既にあるか}
    B -->|はい| C[uv run python main.py list-races]
    B -->|いいえ| D[uv run python main.py init-race --race ...]
    C --> E[対象の races/{slug}/ を選ぶ]
    D --> E
    E --> F[entry.csv history.csv race.yaml を準備]
    F --> G[uv run python main.py predict --race {slug}]
    G --> H[outputs/{race_name}/ の成果物を確認]
    H --> I{レース結果が確定したか}
    I -->|まだ| H
    I -->|はい| J[result.csv を更新]
    J --> K[uv run python main.py settle --race {slug}]
    K --> L[data/training/race_results_master.csv に反映]
    L --> M[次のレースで同じ流れを繰り返す]
```

## 使い方

### 1. 既存レースを予測する

```bash
uv run python main.py predict --race hanshin-daishoten-2026-03-22
```

主な出力先:

- `outputs/Hanshin Daishoten/predictions.csv`
- `outputs/Hanshin Daishoten/recommended_axis_horse.json`
- `outputs/Hanshin Daishoten/evaluation_summary.json`
- `outputs/Hanshin Daishoten/feature_importance.csv`
- `outputs/Hanshin Daishoten/simulation_diagnostics.csv`

### 2. 新しいレースを作成する

```bash
uv run python main.py init-race \
  --race spring-tenno-sho-2026-05-03 \
  --race-name "Tenno Sho Spring" \
  --race-date 2026-05-03
```

作成されるファイル:

- `races/spring-tenno-sho-2026-05-03/race.yaml`
- `races/spring-tenno-sho-2026-05-03/entry.csv`
- `races/spring-tenno-sho-2026-05-03/history.csv`
- `races/spring-tenno-sho-2026-05-03/result.csv`
- `races/spring-tenno-sho-2026-05-03/README.md`

`entry.csv` と `history.csv` を手動で用意するか、サンプルレースのようにレースディレクトリ内へ任意のデータ取得スクリプトを追加して更新します。

### 3. レース結果を確定する

```bash
uv run python main.py settle --race hanshin-daishoten-2026-03-22
```

このコマンドは `result.csv` を検証し、以下を更新します。

- `outputs/Hanshin Daishoten/settled_entry.csv`
- `outputs/Hanshin Daishoten/post_race_report.json`
- `data/training/race_results_master.csv`

補足:

- `result.csv` がフル結果のときだけ、累積学習履歴に追記されます
- Top-K の部分結果でも `post_race_report.json` は生成されます

### 4. よく使うオプション

| コマンド | オプション | 役割 |
| --- | --- | --- |
| `predict` / `settle` | `--race` | `races/` 配下の slug を指定 |
| `predict` / `settle` | `--race-dir` | レースディレクトリを絶対/相対パスで直接指定 |
| `predict` / `settle` | `--training-history` | 累積学習 CSV の場所を切り替え |
| `predict` / `settle` | `--output-root` | 出力ディレクトリのルートを変更 |
| `predict` | `--config` | ベース設定ファイルを差し替え |
| `init-race` | `--force` | 雛形ファイルを上書き |

## 活用事例

### 1. 週次の重賞レース予測を定型化する

レースごとに `races/<slug>/` を切って入出力を分離できるため、毎週の予測ジョブを同じ手順で回せます。対象レースの条件は `race.yaml` に閉じるので、レース追加時の差分も管理しやすくなります。

### 2. 軸馬の根拠を成果物で確認する

単に 1 頭を出すだけではなく、`predictions.csv`、`recommended_axis_horse.json`、`feature_importance.csv`、`calibration_curve.csv` をまとめて残せます。予測値、特徴量寄与、確率の歪みを見比べながらレビューしたいときに向いています。

### 3. レース後レビューを次回学習へつなげる

`settle` 後は `post_race_report.json` で予測上位と実着順の差分を確認でき、フル結果なら `data/training/race_results_master.csv` に反映されます。単発予測で終わらず、週次で学習履歴を積み上げる運用に使えます。

## アーキテクチャ

```mermaid
flowchart TD
    A[data/training/race_results_master.csv<br/>累積学習履歴]
    B[races/{slug}/entry.csv<br/>出走表]
    C[races/{slug}/history.csv<br/>近走履歴]
    D[races/{slug}/race.yaml<br/>レース条件]
    E[races/{slug}/result.csv<br/>レース結果]

    F[main.py<br/>CLI]
    G[workflow.py<br/>レース単位のオーケストレーション]
    H[data_loader.py<br/>検証・正規化]
    I[features.py<br/>特徴量生成]
    J[hybrid_model.py<br/>ランカー + 分類器 + 回帰]
    K[simulation.py<br/>Plackett-Luce シミュレーション]
    L[prediction.py<br/>CV・ブレンディング・出力]

    M[outputs/{race_name}/<br/>予測成果物]
    N[data/training/race_results_master.csv<br/>更新済み学習履歴]

    A --> G
    B --> F
    C --> F
    D --> F
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    G --> N
```

### モデル構成

- `workflow.py` が `src/keiba_predictor/config.yaml` と各レースの `race.yaml` を統合し、予測と結果確定を制御します
- `prediction.py` が時系列 CV、特徴量生成、ハイブリッドモデル学習、出力ファイル生成をまとめて実行します
- `hybrid_model.py` はランカー、分類器、回帰器を組み合わせ、`simulation.py` が順位分布をモンテカルロ推定します

## ディレクトリ構成

```text
.
├── main.py
├── pyproject.toml
├── data/
│   └── training/
│       └── race_results_master.csv
├── outputs/
│   └── Hanshin Daishoten/
├── races/
│   └── hanshin-daishoten-2026-03-22/
│       ├── README.md
│       ├── entry.csv
│       ├── history.csv
│       ├── race.yaml
│       ├── refresh_race_data.py
│       └── result.csv
└── src/
    └── keiba_predictor/
        ├── config.yaml
        ├── data_loader.py
        ├── features.py
        ├── hybrid_model.py
        ├── model.py
        ├── prediction.py
        ├── simulation.py
        └── workflow.py
```

## 主な出力ファイル

| ファイル | 内容 |
| --- | --- |
| `predictions.csv` | 全馬の予測テーブル |
| `recommended_axis_horse.json` | 推奨軸馬 1 頭の詳細 |
| `evaluation_summary.json` | CV ベースの集約指標 |
| `cv_fold_metrics.csv` | Fold ごとの評価結果 |
| `feature_importance.csv` | 特徴量重要度ランキング |
| `calibration_curve.csv` | 確率キャリブレーション用の集計 |
| `simulation_diagnostics.csv` | シミュレーション収束診断 |
| `settled_entry.csv` | 着順を付与した出走表 |
| `post_race_report.json` | 予測上位と実結果の比較レポート |

## ライセンス

このプロジェクトは MIT ライセンスで公開しています。詳細は [LICENSE](LICENSE) を参照してください。
