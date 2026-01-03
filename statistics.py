import argparse
import json
from collections import Counter
from typing import Dict, Any, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error at {path}:{lineno}: {e}") from e
    return rows


def print_counter(title: str, counter: Counter) -> None:
    total = sum(counter.values())
    print(f"\n=== {title} ===")
    print(f"Total: {total}")
    if total == 0:
        return

    # 件数の多い順、同数ならキーの辞書順
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    width = max(len(str(k)) for k, _ in items)
    for k, v in items:
        pct = 100.0 * v / total
        print(f"{str(k):<{width}} : {v:>4}  ({pct:6.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to JSONL (each line is a JSON object).",
    )
    parser.add_argument(
        "--type-key",
        type=str,
        default="type",
        help="Key name for problem type (default: type).",
    )
    parser.add_argument(
        "--level-key",
        type=str,
        default="level",
        help="Key name for problem level (default: level).",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.path)

    type_counter = Counter()
    level_counter = Counter()

    missing_type = 0
    missing_level = 0

    for r in rows:
        if args.type_key in r and r[args.type_key] is not None:
            type_counter[str(r[args.type_key])] += 1
        else:
            missing_type += 1

        if args.level_key in r and r[args.level_key] is not None:
            level_counter[str(r[args.level_key])] += 1
        else:
            missing_level += 1

    print(f"Loaded {len(rows)} rows from: {args.path}")

    if missing_type:
        print(f"[warn] Missing '{args.type_key}' in {missing_type} row(s)")
    if missing_level:
        print(f"[warn] Missing '{args.level_key}' in {missing_level} row(s)")

    print_counter(f"Counts by '{args.type_key}'", type_counter)
    print_counter(f"Counts by '{args.level_key}'", level_counter)


if __name__ == "__main__":
    main()
