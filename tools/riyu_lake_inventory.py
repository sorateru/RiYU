#!/usr/bin/env python3
"""Read-only annual inventory and deterministic, private selection plans."""

from __future__ import annotations

import argparse
import bz2
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from time import perf_counter


GO = re.compile(rb'<GO type="(\d+)"')
PLAYERS = re.compile(rb"<UN [^>]*>")
META = re.compile(rb"<(?:UN|BYE)\b[^>]*>")
WINS = re.compile(rb"<AGARI\b[^>]*>")
YAKU = re.compile(rb' yaku="([\d,]+)"')
END = re.compile(rb' owari="([^"]+)"')
REACH_RON = re.compile(rb'<REACH who="\d+" step="1"[^>]*>\s*<[DEFGdefg]\d+[^>]*>\s*<AGARI')
DOUBLE = re.compile(rb"<AGARI\b[^>]*>\s*<AGARI\b[^>]*>")
NAMES = tuple(re.compile(fr' n{i}="([^"]*)"'.encode()) for i in range(4))


def inventory_row(record):
    locator, blob = record
    raw = bz2.decompress(blob)
    go_tag = GO.search(raw)
    if go_tag is None:
        raise ValueError("missing GO in the archive")
    rule = int(go_tag[1])
    start = raw.find(b"<INIT")
    digest = sha256(go_tag[1] + b"\0" + META.sub(b"", raw[start:]).replace(b"/>\n<", b"/><")).digest()
    flags = set()
    if b'type="ron3"' in raw:
        flags.add("triple_ron")
    if b'type="kan4"' in raw:
        flags.add("four_kan")
    if b" paoWho=" in raw:
        flags.add("pao")
    if DOUBLE.search(raw):
        flags.add("double_ron")
    if REACH_RON.search(raw):
        flags.add("riichi_tile_ron")
    if re.search(rb'<INIT seed="(?:8|9|10|11),', raw):
        flags.add("west_extension")
    if (end := END.search(raw)) is not None:
        scores = [float(v) for v in end[1].split(b",")[::2]]
        if len(set(scores)) < len(scores):
            flags.add("tie")
        if min(scores) < 0:
            flags.add("bankruptcy")
    for tag in WINS.findall(raw):
        match = YAKU.search(tag)
        if match and b"3" in match[1].split(b",")[::2]:
            flags.add("chankan")
            break
    player = PLAYERS.search(raw)
    names = tuple(x[1] for pattern in NAMES if player and (x := pattern.search(player[0])) and x[1])
    ranks = re.search(rb' dan="([^"]+)"', player[0]) if player else None
    ratings = re.search(rb' rate="([^"]+)"', player[0]) if player else None
    seed = b'<SHUFFLE seed="mt19937ar-sha512-n288-base64,' in raw
    return (
        locator, rule, digest, flags, names, seed,
        ranks[1].split(b",")[:len(names)] if ranks else [],
        ratings[1].split(b",")[:len(names)] if ratings else [],
    )


def write_private(path, locators):
    with path.open("x") as stream:
        stream.writelines(locator + "\n" for locator in locators)


def run(database, output, size, fixture_limit, workers):
    os.umask(0o077)
    output.mkdir(mode=0o700)
    t = perf_counter()
    months = defaultdict(Counter)
    hanchan = defaultdict(list)
    all_games, rare, players, period_players = {}, defaultdict(list), Counter(), defaultdict(set)
    parents, graph_matches = {}, Counter()
    rank_counts, rating_buckets, flags_count = Counter(), Counter(), Counter()
    duplicates, seed_count = 0, 0

    def component(name):
        while (p := parents.setdefault(name, name)) != name:
            parents[name] = parents[p]
            name = p
        return name

    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db, ThreadPoolExecutor(max_workers=workers) as pool:
        cursor = db.execute("SELECT log_id, log_content FROM logs ORDER BY log_id")
        while batch := cursor.fetchmany(256):
            for locator, rule, digest, flags, names, seed, ranks, ratings in pool.map(inventory_row, batch):
                month = locator[:6]
                months[month][rule] += 1
                seed_count += seed
                duplicates += digest in all_games
                all_games.setdefault(digest, locator)
                if all_games[digest] != locator or rule not in (169, 225):
                    continue
                for flag in flags:
                    flags_count[flag] += 1
                    if len(rare[flag]) < fixture_limit:
                        rare[flag].append(locator)
                if rule != 169:
                    continue
                hanchan[month].append(locator)
                players.update(names)
                period_players["early" if month < "202011" else "late"].update(names)
                rank_counts.update(int(r) for r in ranks)
                rating_buckets.update(int(float(r)) // 100 * 100 for r in ratings)
                if names:
                    for player in names[1:]:
                        parents[component(player)] = component(names[0])
                    graph_matches[names[0]] += 1
    if size > sum(map(len, hanchan.values())):
        raise ValueError("not enough unique four-player hanchan")
    sample = []
    quota, extra = divmod(size, len(hanchan))
    months_sampled = {}
    for index, (month, locators) in enumerate(sorted(hanchan.items())):
        n = quota + (index < extra)
        ranked = sorted(locators, key=lambda x: sha256(("riyu-2020-sample-v1\0" + x).encode()).digest())
        if len(ranked) < n:
            raise ValueError("month cannot satisfy the requested balanced plan")
        sample.extend(ranked[:n])
        months_sampled[month] = n
    # Interleave months to make resume and prefix measurements representative.
    sample.sort(key=lambda x: sha256(("riyu-2020-order-v1\0" + x).encode()).digest())
    write_private(output / "balanced.plan", sample)
    for name, locators in sorted(rare.items()):
        write_private(output / (name + ".plan"), locators)
    write_private(output / "special.plan", sorted({x for locators in rare.values() for x in locators}))
    components = Counter()
    for p, n in graph_matches.items():
        components[component(p)] += n
    report = {
        "matches": sum(sum(m.values()) for m in months.values()), "monthly_by_go": months,
        "hanchan_169": sum(map(len, hanchan.values())), "sample_months": months_sampled,
        "gameplay_byte_duplicates": duplicates, "seed_format_matches": seed_count,
        "special_counts_169_225": flags_count, "local_fixture_matches": {k: len(v) for k, v in rare.items()},
        "hanchan_player_count": len(players), "hanchan_player_top_share": max(players.values()) / players.total(),
        "hanchan_rank_values": rank_counts, "rating_buckets": rating_buckets,
        "temporal_players": {k: len(v) for k, v in period_players.items()},
        "temporal_player_overlap": len(period_players["early"] & period_players["late"]),
        "player_component_sizes": sorted(components.values(), reverse=True)[:5],
        "seconds": perf_counter() - t,
    }
    (output / "aggregate.json").write_text(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("database", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--size", type=int, default=10000)
    p.add_argument("--fixture-limit", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if min(a.size, a.fixture_limit, a.workers) < 1:
        p.error("counts must be positive")
    run(a.database, a.output, a.size, a.fixture_limit, a.workers)


if __name__ == "__main__":
    main()
