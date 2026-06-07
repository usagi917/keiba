"""実データから baselines.json を再生成する。

合成プレースホルダ(synthetic data)を、実CSVを1回CVして得た正直な数字で置き換える。
出力には人気馬追随ベースラインとモデルの上乗せ(lift)、データ十分性(低信頼フラグ)を含める。

Usage:
    uv run python scripts/generate_baselines.py \
        --data data/training/race_results_master.csv \
        --config src/keiba_predictor/config.yaml \
        --out baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keiba_predictor.backtest_baseline import generate_real_baselines, save_baselines  # noqa: E402
from keiba_predictor.data_loader import coerce_dtypes, read_csv_safely  # noqa: E402
from keiba_predictor.prediction import load_config  # noqa: E402


def _load_history(data_path: Path) -> pd.DataFrame:
    raw = read_csv_safely(data_path)
    return coerce_dtypes(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="実データから baselines.json を再生成")
    parser.add_argument("--data", default="data/training/race_results_master.csv")
    parser.add_argument("--config", default="src/keiba_predictor/config.yaml")
    parser.add_argument("--out", default="baselines.json")
    parser.add_argument("--n-partners", type=int, default=5)
    args = parser.parse_args()

    data_path = Path(args.data)
    history = _load_history(data_path)
    config = load_config(args.config)

    baselines = generate_real_baselines(
        history_df=history,
        config=config,
        n_partners=args.n_partners,
        source_file=str(data_path),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    save_baselines(baselines, args.out)

    suff = baselines["data_sufficiency"]
    edge = baselines["edge_vs_market"]
    fav = baselines["favorite_baseline"]

    print("=" * 72)
    print(f"baselines.json を再生成しました → {args.out}")
    print(f"  source            : {baselines['source_file']}")
    print(f"  n_races(評価)     : {baselines['n_races']}  date_range={baselines['date_range']}")
    print(f"  is_synthetic      : {baselines['is_synthetic']}")
    print("-" * 72)
    print(f"  モデル軸Top3的中率   : {edge['model_axis_top3_hit_rate']}")
    print(f"  シャドウ軸Top3的中率 : {edge['shadow_axis_top3_hit_rate']}  (オッズ抜き)")
    print(f"  人気馬軸Top3的中率   : {fav['axis_top3_hit_rate']}  (ベースライン)")
    print(f"  上乗せ(モデル-人気) : {edge['lift_model_over_favorite']}")
    print(f"  上乗せ(シャドウ-人気): {edge['lift_shadow_over_favorite']}  ← 市場を超える本当のエッジの代理")
    print("-" * 72)
    if suff.get("is_low_confidence"):
        print(f"  [⚠ 低信頼] regime={suff['regime']} / n_races={suff['n_races']}")
        for w in suff.get("warnings", []):
            print(f"    - {w}")
    print("=" * 72)
    print(json.dumps(baselines, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
