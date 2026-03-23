from __future__ import annotations

import csv
import html
import re
import sys
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True


BUNDLE_DIR = Path(__file__).resolve().parent
ENTRY_PATH = BUNDLE_DIR / "entry.csv"
HISTORY_PATH = BUNDLE_DIR / "history.csv"
NOTES_PATH = BUNDLE_DIR / "README.md"

RACE_SOURCE_ID = "2609011011"
RACE_URL = f"https://sports.yahoo.co.jp/keiba/race/denma/{RACE_SOURCE_ID}"
USER_AGENT = "keiba-predictor/1.0 (personal research; non-commercial)"

COURSE_MAP = {
    "札幌": "Sapporo",
    "函館": "Hakodate",
    "福島": "Fukushima",
    "新潟": "Niigata",
    "東京": "Tokyo",
    "中山": "Nakayama",
    "中京": "Chukyo",
    "京都": "Kyoto",
    "阪神": "Hanshin",
    "小倉": "Kokura",
}

TURN_MAP = {
    "Sapporo": "right",
    "Hakodate": "right",
    "Fukushima": "right",
    "Niigata": "left",
    "Tokyo": "left",
    "Nakayama": "right",
    "Chukyo": "left",
    "Kyoto": "right",
    "Hanshin": "right",
    "Kokura": "right",
}

SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}

SEX_MAP = {"牡": "M", "牝": "F", "せん": "G", "セ": "G"}
GRADE_MAP = {"GI": "G1", "GII": "G2", "GIII": "G3", "GIV": "G4", "JGI": "JG1", "JGII": "JG2", "JGIII": "JG3", "L": "L"}
OPEN_RACE_NAMES = {
    "万葉S",
    "大ハC",
    "大阪城S",
    "白富士S",
    "万葉ステークス",
    "大阪城ステークス",
}
THREE_WIN_RACE_NAMES = {
    "比叡S",
    "古都S",
    "烏丸S",
    "ムーンライトH",
    "ムーンライト",
    "サンシャインS",
    "湾岸S",
    "御堂筋S",
}

CSV_COLUMNS = [
    "race_id",
    "race_date",
    "race_month",
    "season",
    "course",
    "surface",
    "distance",
    "turn_direction",
    "grade",
    "class",
    "field_size",
    "horse_id",
    "horse_name",
    "draw",
    "horse_number",
    "sex",
    "age",
    "carried_weight",
    "jockey",
    "trainer",
    "body_weight",
    "body_weight_diff",
    "odds",
    "popularity",
    "last_finish",
    "last_distance",
    "last_class",
    "last_margin",
    "last_3f",
    "days_since_last",
    "finish_rank",
]


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def strip_tags(text: str) -> str:
    return collapse_spaces(re.sub(r"<[^>]+>", "", text))


def compact_name(text: str) -> str:
    return re.sub(r"[ 　]+", "", collapse_spaces(text))


def season_from_date(dt: date) -> str:
    return SEASON_BY_MONTH[dt.month]


def age_from_horse_id(horse_id: str, race_dt: date) -> str:
    birth_year = int(str(horse_id)[:4])
    return str(race_dt.year - birth_year)


def parse_weight_block(text: str) -> Tuple[str, str]:
    text = collapse_spaces(text)
    if not text or text == "-(-)":
        return "", ""
    match = re.match(r"(\d+)\(([-+]?\d+)\)", text)
    if match:
        return match.group(1), match.group(2)
    return text, ""


def normalize_grade(text: str) -> str:
    return GRADE_MAP.get(text.strip(), text.strip())


def infer_class(race_name: str, grade: str, old_class: str) -> str:
    if old_class:
        return old_class
    if grade:
        return "Open"
    if "新馬" in race_name:
        return "Newcomer"
    if "未勝利" in race_name:
        return "Maiden"
    if any(token in race_name for token in ["1勝", "2勝", "1勝クラス", "2勝クラス", "500万", "1000万", "Allow"]):
        return "Allowance"
    if any(token in race_name for token in ["3勝", "1600万"]):
        return "3Win"
    if race_name in OPEN_RACE_NAMES:
        return "Open"
    if race_name in THREE_WIN_RACE_NAMES:
        return "3Win"
    if race_name.endswith("S") or race_name.endswith("ステークス"):
        return "Open"
    return "Allowance"


def compute_draw(field_size: int, horse_number: int) -> str:
    if field_size <= 0 or horse_number <= 0:
        return ""
    if field_size <= 8:
        return str(horse_number)
    if field_size <= 16:
        single_brackets = 16 - field_size
        bracket_sizes = [1] * single_brackets + [2] * (8 - single_brackets)
    else:
        triple_brackets = field_size - 16
        bracket_sizes = [2] * (8 - triple_brackets) + [3] * triple_brackets
    total = 0
    for idx, size in enumerate(bracket_sizes, start=1):
        total += size
        if horse_number <= total:
            return str(idx)
    return str(min(8, horse_number))


def parse_course_surface_distance(detail_text: str) -> Tuple[str, str, str, str]:
    match = re.search(r"([^\s]+)\s+([芝ダ障])(\d+)m\s+(\d+)頭", collapse_spaces(detail_text))
    if not match:
        raise ValueError(f"Cannot parse course detail: {detail_text}")
    course_jp, surface_jp, distance, field_size = match.groups()
    surface = {"芝": "turf", "ダ": "dirt", "障": "jump"}[surface_jp]
    course = COURSE_MAP[course_jp]
    return course, surface, distance, field_size


def build_race_id(race_dt: date, course: str, surface: str, distance: str, race_name: str) -> str:
    safe_race_name = re.sub(r"\s+", "", race_name)
    return f"{race_dt.isoformat()}_{course}_{surface}_{distance}_{safe_race_name}"


def extract_row_blocks(table_html: str) -> List[str]:
    return re.findall(r'<tr class="hr-table__row">(.*?)</tr>', table_html, re.S)


def load_existing_history() -> Tuple[OrderedDict[str, int], Dict[Tuple[str, str], Dict[str, str]]]:
    counts: OrderedDict[str, int] = OrderedDict()
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    with HISTORY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            horse_id = row["horse_id"]
            counts[horse_id] = counts.get(horse_id, 0) + 1
            lookup[(horse_id, row["race_date"])] = row
    return counts, lookup


def parse_entry_page(page_html: str) -> Tuple[Dict[str, str], List[Dict[str, str]], str]:
    date_match = re.search(r'<div class="hr-predictRaceInfo__text">(\d{4})年(\d{1,2})月(\d{1,2})日', page_html)
    title_match = re.search(r'<h2 class="hr-predictRaceInfo__title">\s*([^<\n]+)\s*<span class="hr-label[^"]*">([^<]+)</span>', page_html, re.S)
    status_match = re.search(r'<span class="hr-predictRaceInfo__text">([芝ダ障])・([左右直])\s+(\d+)m</span>', page_html)
    class_match = re.search(r'<span class="hr-predictRaceInfo__text">([^<]*オープン[^<]*)</span>', page_html)
    update_match = re.search(r'<time class="hr-tableNote__update" datetime="[^"]+">([^<]+)</time>', page_html)
    tbody_match = re.search(r"<tbody>(.*?)</tbody>", page_html, re.S)
    if not (date_match and title_match and status_match and tbody_match):
        raise ValueError("Could not parse race entry page.")

    year, month, day = map(int, date_match.groups())
    race_dt = date(year, month, day)
    race_name = collapse_spaces(title_match.group(1))
    grade = normalize_grade(collapse_spaces(title_match.group(2)))
    surface_jp, turn_jp, distance = status_match.groups()
    surface = {"芝": "turf", "ダ": "dirt", "障": "jump"}[surface_jp]
    turn = {"左": "left", "右": "right", "直": "straight"}[turn_jp]
    race_class = "Open" if class_match else "Unknown"

    rows = []
    for block in extract_row_blocks(tbody_match.group(1)):
        draw_match = re.search(r'hr-icon__bracketNum[^"]*">(\d+)</span>', block)
        number_cells = re.findall(r'<td class="hr-table__data hr-table__data--number(?:[^"]*)?">\s*(?:<span[^>]+>)?([^<\s]+)', block)
        horse_match = re.search(
            r'<td class="hr-table__data hr-table__data--name">\s*<a href="/keiba/directory/horse/(\d+)/"[^>]*>([^<]+)</a>\s*<p>([^<]+)</p>',
            block,
            re.S,
        )
        jockey_match = re.search(
            r'<td class="hr-table__data hr-table__data--name"><a href="/keiba/directory/jockey/\d+/"[^>]*>([^<]+)</a><p>([^<]+)</p></td>',
            block,
            re.S,
        )
        trainer_match = re.search(
            r'<td class="hr-table__data hr-table__data--name"><a href="/keiba/directory/trainer/\d+/"[^>]*>([^<]+)</a><p>\(([^<]+)\)</p></td>',
            block,
            re.S,
        )
        weight_cells = re.findall(r'<td class="hr-table__data hr-table__data--name"><p>([^<]+)</p></td>', block)
        odds_match = re.search(r'<td class="hr-table__data hr-table__data--odds">\s*([^<(]+)\(<span[^>]*>([^<]+)</span>\)', block, re.S)
        if not (draw_match and len(number_cells) >= 2 and horse_match and jockey_match and trainer_match and weight_cells and odds_match):
            continue

        sex_age = collapse_spaces(horse_match.group(3)).split("/")[0]
        sex = next((SEX_MAP[key] for key in SEX_MAP if sex_age.startswith(key)), "")
        age = re.search(r"(\d+)", sex_age).group(1)
        body_weight, body_weight_diff = parse_weight_block(weight_cells[-1])
        trainer_region = trainer_match.group(2)
        rows.append(
            {
                "race_id": build_race_id(race_dt, "Hanshin", surface, distance, race_name),
                "race_date": race_dt.isoformat(),
                "race_month": str(race_dt.month),
                "season": season_from_date(race_dt),
                "course": "Hanshin",
                "surface": surface,
                "distance": distance,
                "turn_direction": turn,
                "grade": grade,
                "class": race_class,
                "field_size": "",
                "horse_id": horse_match.group(1),
                "horse_name": collapse_spaces(horse_match.group(2)),
                "draw": draw_match.group(1),
                "horse_number": number_cells[1],
                "sex": sex,
                "age": age,
                "carried_weight": jockey_match.group(2),
                "jockey": compact_name(jockey_match.group(1)),
                "trainer": compact_name(trainer_match.group(1)),
                "body_weight": body_weight,
                "body_weight_diff": body_weight_diff,
                "odds": collapse_spaces(odds_match.group(2)),
                "popularity": collapse_spaces(odds_match.group(1)),
                "last_finish": "",
                "last_distance": "",
                "last_class": "",
                "last_margin": "",
                "last_3f": "",
                "days_since_last": "",
                "finish_rank": "",
                "trainer_region": trainer_region,
            }
        )

    rows.sort(key=lambda row: int(row["horse_number"]))
    field_size = str(len(rows))
    for row in rows:
        row["field_size"] = field_size

    meta = {
        "race_date": race_dt.isoformat(),
        "race_name": race_name,
        "grade": grade,
        "field_size": field_size,
    }
    update_text = collapse_spaces(update_match.group(1)) if update_match else ""
    return meta, rows, update_text


def parse_horse_history(
    horse_info: Dict[str, str],
    existing_lookup: Dict[Tuple[str, str], Dict[str, str]],
    keep_count: int,
) -> List[Dict[str, str]]:
    horse_url = f"https://sports.yahoo.co.jp/keiba/directory/horse/{horse_info['horse_id']}/"
    page_html = fetch_html(horse_url)
    section_match = re.search(r'<h2 class="hr-head01__title">出走レース</h2>(.*?)<footer class="hr-tableNote">', page_html, re.S)
    if not section_match:
        raise ValueError(f"Could not locate history table: {horse_url}")
    blocks = extract_row_blocks(section_match.group(1))
    recent_blocks = blocks[:keep_count]
    rows: List[Dict[str, str]] = []
    for block in recent_blocks:
        date_match = re.search(r'hr-table__data--date">([^<]+)</td>', block)
        race_match = re.search(
            r'hr-table__data--race">\s*<a href="/keiba/race/index/(\d+)"[^>]*>([^<]+)</a>(.*?)<p>([^<]+)</p></td>',
            block,
            re.S,
        )
        number_cells = re.findall(r'hr-table__data hr-table__data--number(?:[^"]*)?">([^<]+)</td>', block)
        jockey_match = re.search(
            r'hr-table__data--name"><a href="/keiba/directory/jockey/\d+/"[^>]*>([^<]+)</a><p>\(([^<]+)\)</p></td>',
            block,
            re.S,
        )
        odds_match = re.search(r'hr-table__data--odds">\s*([^<\s]+)<p>\(([^<]+)\)</p>\s*</td>', block, re.S)
        weight_match = re.search(r'hr-table__data--weight">([^<]*)</td>', block)
        time_match = re.search(r'hr-table__data--time">([^<]*)(?:<p>\(([^<]*)\)</p>)?</td>', block, re.S)
        order_match = re.search(r'hr-table__data--order">([^<]*)(?:<p>([^<]*)</p>)?</td>', block, re.S)
        if not (date_match and race_match and len(number_cells) >= 2 and jockey_match and odds_match and weight_match and time_match and order_match):
            continue

        race_dt = datetime.strptime(collapse_spaces(date_match.group(1)), "%Y/%m/%d").date()
        race_link_id, race_name, grade_block, detail_text = race_match.groups()
        grade_text_match = re.search(r'<span class="hr-label[^"]*">([^<]+)</span>', grade_block)
        grade = normalize_grade(strip_tags(grade_text_match.group(1))) if grade_text_match else ""
        course, surface, distance, field_size = parse_course_surface_distance(detail_text)
        finish_text = collapse_spaces(number_cells[0])
        horse_number = collapse_spaces(number_cells[1])
        finish_rank = finish_text if finish_text.isdigit() else ""
        body_weight, body_weight_diff = parse_weight_block(collapse_spaces(weight_match.group(1)))
        margin = collapse_spaces(time_match.group(2) or "")
        current_3f = collapse_spaces(order_match.group(2) or "")
        if current_3f == "-":
            current_3f = ""
        old_row = existing_lookup.get((horse_info["horse_id"], race_dt.isoformat()), {})
        row = {
            "race_id": build_race_id(race_dt, course, surface, distance, collapse_spaces(race_name)),
            "race_date": race_dt.isoformat(),
            "race_month": str(race_dt.month),
            "season": season_from_date(race_dt),
            "course": course,
            "surface": surface,
            "distance": distance,
            "turn_direction": TURN_MAP[course],
            "grade": grade or old_row.get("grade", ""),
            "class": infer_class(collapse_spaces(race_name), grade, old_row.get("class", "")),
            "field_size": field_size,
            "horse_id": horse_info["horse_id"],
            "horse_name": horse_info["horse_name"],
            "draw": compute_draw(int(field_size), int(horse_number)),
            "horse_number": horse_number,
            "sex": horse_info["sex"],
            "age": age_from_horse_id(horse_info["horse_id"], race_dt),
            "carried_weight": jockey_match.group(2),
            "jockey": compact_name(jockey_match.group(1)),
            "trainer": horse_info["trainer"],
            "body_weight": body_weight,
            "body_weight_diff": body_weight_diff,
            "odds": collapse_spaces(odds_match.group(2)),
            "popularity": collapse_spaces(odds_match.group(1)),
            "last_finish": "",
            "last_distance": "",
            "last_class": "",
            "last_margin": "",
            "last_3f": "",
            "days_since_last": "",
            "finish_rank": finish_rank,
            "_margin": margin,
            "_three_f": current_3f,
            "_race_name": collapse_spaces(race_name),
            "_source_race_id": race_link_id,
        }
        rows.append(row)

    rows.sort(key=lambda row: row["race_date"])
    previous = None
    for row in rows:
        if previous is not None:
            prev_date = datetime.strptime(previous["race_date"], "%Y-%m-%d").date()
            curr_date = datetime.strptime(row["race_date"], "%Y-%m-%d").date()
            row["last_finish"] = previous["finish_rank"]
            row["last_distance"] = previous["distance"]
            row["last_class"] = previous["class"]
            row["last_margin"] = previous["_margin"]
            row["last_3f"] = previous["_three_f"]
            row["days_since_last"] = str((curr_date - prev_date).days)
        previous = row
    return rows


def build_entry_rows(meta: Dict[str, str], entry_rows: List[Dict[str, str]], history_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    latest_history: Dict[str, Dict[str, str]] = {}
    for row in history_rows:
        latest_history[row["horse_id"]] = row

    entry_date = datetime.strptime(meta["race_date"], "%Y-%m-%d").date()
    out: List[Dict[str, str]] = []
    for row in entry_rows:
        prev = latest_history.get(row["horse_id"])
        if prev:
            prev_date = datetime.strptime(prev["race_date"], "%Y-%m-%d").date()
            row["last_finish"] = prev["finish_rank"]
            row["last_distance"] = prev["distance"]
            row["last_class"] = prev["class"]
            row["last_margin"] = prev["_margin"]
            row["last_3f"] = prev["_three_f"]
            row["days_since_last"] = str((entry_date - prev_date).days)
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def write_notes(
    entry_rows: List[Dict[str, str]],
    history_rows: List[Dict[str, str]],
    entry_update_text: str,
) -> None:
    historical_missing = {key: 0 for key in ["grade", "last_3f", "last_margin", "last_finish", "days_since_last", "last_class", "last_distance", "body_weight_diff", "finish_rank"]}
    for row in history_rows:
        for key in historical_missing:
            if not str(row.get(key, "")).strip():
                historical_missing[key] += 1
    body_weight_missing = sum(1 for row in entry_rows if not str(row.get("body_weight", "")).strip())

    note = f"""# 阪神大賞典 2026-03-22 用 CSV ノート

## 作成物
- history.csv
- entry.csv
- refresh_race_data.py

## 行数
- history.csv: {len(history_rows)} 行
- entry.csv: {len(entry_rows)} 行

## スキーマ
両ファイルの列は同一です。

race_id, race_date, race_month, season, course, surface, distance, turn_direction, grade, class, field_size, horse_id, horse_name, draw, horse_number, sex, age, carried_weight, jockey, trainer, body_weight, body_weight_diff, odds, popularity, last_finish, last_distance, last_class, last_margin, last_3f, days_since_last, finish_rank

## データの作り方
- `entry.csv`
  - スポーツナビの阪神大賞典出馬表 `https://sports.yahoo.co.jp/keiba/race/denma/{RACE_SOURCE_ID}` を基に再生成
  - 2026/3/22 時点の枠順、馬番、騎手、斤量、単勝オッズ、人気を反映
  - 出馬表更新時刻: {entry_update_text or "取得時刻未取得"}
- `history.csv`
  - 同出走10頭の `https://sports.yahoo.co.jp/keiba/directory/horse/<horse_id>/` から近走テーブルを取得
  - 既存バンドルと同じ馬ごとの件数を維持しつつ、公開履歴を機械抽出して再構造化
  - `last_*` と `days_since_last` は抽出後に機械再計算
- `draw`
  - `entry.csv` は出馬表の枠順
  - `history.csv` は頭数と馬番から JRA の一般的な枠順配分に従って再計算
- `trainer`
  - 履歴行では当日出走表の所属厩舎名を採用
- `class`
  - グレード競走と Listed は `Open`
  - 非グレード戦は既存バンドルのクラス定義を優先し、不足分だけレース名パターンで補完

## 欠損
- `entry.csv` の `body_weight` / `body_weight_diff`: {body_weight_missing} 件空欄
  - 取得時点の出馬表では `-(-)` 表記で、馬体重発表前のため空欄維持
- `history.csv` の主な空欄件数
  - `grade`: {historical_missing['grade']}
  - `last_3f`: {historical_missing['last_3f']}
  - `last_margin`: {historical_missing['last_margin']}
  - `last_finish`: {historical_missing['last_finish']}
  - `days_since_last`: {historical_missing['days_since_last']}

## 注意点
- `history.csv` は過去レース全頭分ではなく、出走10頭の近走テーブルだけです。
- スポーツナビ側で `中止` / `取消` などの非完走は `finish_rank` を空欄にしています。
- スポーツナビの注記どおり、最終確認は主催者発表値と照合してください。

## 更新方法
```bash
uv run python races/hanshin-daishoten-2026-03-22/refresh_race_data.py
```
"""
    NOTES_PATH.write_text(note, encoding="utf-8")


def main() -> None:
    keep_counts, existing_lookup = load_existing_history()
    _, entry_rows, entry_update_text = parse_entry_page(fetch_html(RACE_URL))

    entry_by_horse_id = {row["horse_id"]: row for row in entry_rows}
    history_rows: List[Dict[str, str]] = []
    for horse_id, keep_count in keep_counts.items():
        if horse_id not in entry_by_horse_id:
            raise KeyError(f"horse_id not found in entry page: {horse_id}")
        history_rows.extend(parse_horse_history(entry_by_horse_id[horse_id], existing_lookup, keep_count))

    final_entry_rows = build_entry_rows(
        {"race_date": entry_rows[0]["race_date"] if entry_rows else ""},
        entry_rows,
        history_rows,
    )
    write_csv(HISTORY_PATH, history_rows)
    write_csv(ENTRY_PATH, final_entry_rows)
    write_notes(final_entry_rows, history_rows, entry_update_text)


if __name__ == "__main__":
    main()
