#!/usr/bin/env python3
"""Convert validated offline event logs to a supervised Parquet dataset."""

from __future__ import annotations

import argparse

from mahjong_ai.data import build_decision_records, load_event_log, write_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for path in args.logs:
        records.extend(build_decision_records(load_event_log(path)))
    write_parquet(records, args.output)
    print(f"wrote {len(records)} decisions to {args.output}")


if __name__ == "__main__":
    main()
