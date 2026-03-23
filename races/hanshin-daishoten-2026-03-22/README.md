# Hanshin Daishoten 2026-03-22

## Files
- `race.yaml`: 対象レースの条件
- `entry.csv`: 出走馬データ
- `history.csv`: 今回の予測に使う近走データ
- `result.csv`: レース後の着順
- `refresh_race_data.py`: サンプル用のデータ更新スクリプト

## Commands
- 予測: `uv run python main.py --race hanshin-daishoten-2026-03-22`
- 結果反映: `uv run python main.py settle --race hanshin-daishoten-2026-03-22`
- データ更新: `uv run python races/hanshin-daishoten-2026-03-22/refresh_race_data.py`

## Outputs
- 予測結果: `outputs/Hanshin Daishoten/`
- 学習用累積データ: `data/training/race_results_master.csv`
