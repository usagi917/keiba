from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keiba_predictor.workflow import (  # noqa: E402
    create_race_directory,
    list_race_directories,
    predict_race,
    resolve_race_directory,
    settle_race,
    summarize_race_directory,
)


DEFAULT_COMMAND = "predict"
SUPPORTED_COMMANDS = {"predict", "settle", "init-race", "list-races"}
DEFAULT_CONFIG_PATH = ROOT_DIR / "src" / "keiba_predictor" / "config.yaml"
DEFAULT_RACES_ROOT = ROOT_DIR / "races"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs"
DEFAULT_TRAINING_HISTORY_PATH = ROOT_DIR / "data" / "training" / "race_results_master.csv"


def normalize_cli_args(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    if argv[0] in {"-h", "--help"}:
        return argv
    if argv[0] in SUPPORTED_COMMANDS:
        return argv
    if argv[0].startswith("-"):
        return [DEFAULT_COMMAND, *argv]
    return [DEFAULT_COMMAND, "--race", argv[0], *argv[1:]]


def add_race_selector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--race", help="`races/` 配下のレース slug")
    parser.add_argument("--race-dir", help="レースディレクトリの直接指定")
    parser.add_argument("--races-root", default=str(DEFAULT_RACES_ROOT), help="レースディレクトリの親")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="週次競馬レース予測ワークフロー")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="指定レースを予測する")
    add_race_selector_args(predict_parser)
    predict_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="ベース設定ファイル")
    predict_parser.add_argument(
        "--training-history",
        "--master-history",
        dest="training_history",
        default=str(DEFAULT_TRAINING_HISTORY_PATH),
        help="学習用の累積レース結果 CSV",
    )
    predict_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="予測成果物の出力先ルート",
    )

    settle_parser = subparsers.add_parser("settle", help="result.csv を学習データへ反映する")
    add_race_selector_args(settle_parser)
    settle_parser.add_argument(
        "--training-history",
        "--master-history",
        dest="training_history",
        default=str(DEFAULT_TRAINING_HISTORY_PATH),
        help="学習用の累積レース結果 CSV",
    )
    settle_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="予測成果物の出力先ルート",
    )

    init_parser = subparsers.add_parser("init-race", help="新しい週次レースの雛形を作る")
    init_parser.add_argument("--race", required=True, help="作成するレース slug")
    init_parser.add_argument("--races-root", default=str(DEFAULT_RACES_ROOT), help="レースディレクトリの親")
    init_parser.add_argument("--race-name", default="", help="表示名。未指定なら slug ベースで自動生成")
    init_parser.add_argument("--race-date", default="", help="YYYY-MM-DD")
    init_parser.add_argument("--force", action="store_true", help="既存ファイルを上書きする")

    list_parser = subparsers.add_parser("list-races", help="利用可能なレースを表示する")
    list_parser.add_argument("--races-root", default=str(DEFAULT_RACES_ROOT), help="レースディレクトリの親")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    normalized = normalize_cli_args(list(sys.argv[1:] if argv is None else argv))
    return build_parser().parse_args(normalized)


def main() -> None:
    args = parse_args()

    if args.command == "list-races":
        races_root = Path(args.races_root).resolve()
        race_dirs = list_race_directories(races_root)
        if not race_dirs:
            print(f"[INFO] レースディレクトリがありません: {races_root}")
            return
        for race_dir in race_dirs:
            summary = summarize_race_directory(race_dir)
            suffix = f" ({summary['race_date']})" if summary["race_date"] else ""
            print(f"{summary['slug']}: {summary['race_name']}{suffix}")
        return

    if args.command == "init-race":
        races_root = Path(args.races_root).resolve()
        race_dir = races_root / args.race
        try:
            created = create_race_directory(
                race_dir=race_dir,
                race_name=args.race_name,
                race_date_text=args.race_date,
                force=bool(args.force),
            )
        except ValueError as exc:
            raise SystemExit(f"[ERROR] race-date の解釈に失敗しました: {exc}") from exc
        except FileExistsError as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc
        for label, path in created.items():
            print(f"{label}={path}")
        return

    races_root = Path(args.races_root).resolve()
    race_dir = resolve_race_directory(races_root=races_root, race=args.race, race_dir=args.race_dir)
    output_root = Path(args.output_root).resolve()
    training_history_path = Path(args.training_history).resolve()

    if args.command == "predict":
        predict_race(
            race_dir=race_dir,
            base_config_path=Path(args.config).resolve(),
            training_history_path=training_history_path,
            output_root=output_root,
        )
        return

    if args.command == "settle":
        settle_race(
            race_dir=race_dir,
            training_history_path=training_history_path,
            output_root=output_root,
        )
        return

    raise SystemExit(f"[ERROR] 未対応の command です: {args.command}")


if __name__ == "__main__":
    main()
