#!/usr/bin/env python3
"""Offline Tenhou shuffle verification. Emits aggregate counts only."""

from __future__ import annotations

import argparse
import base64
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import struct
import xml.etree.ElementTree as ET


MASK = (1 << 32) - 1
SOURCE_FORMAT = "mt19937ar-sha512-n288-base64"


def rng_from_log(value):
    format_name, encoded = value.split(",", 1)
    if format_name != SOURCE_FORMAT:
        raise ValueError("unsupported shuffle seed format")
    material = struct.unpack("<624I", base64.b64decode(encoded, validate=True))
    mt = [19650218]
    for j in range(1, 624):
        mt.append((1812433253 * (mt[-1] ^ (mt[-1] >> 30)) + j) & MASK)
    i = 1
    for j, seed in enumerate(material):
        prev = mt[i-1]
        mt[i] = ((mt[i] ^ ((prev ^ (prev >> 30)) * 1664525)) + seed + j) & MASK
        i += 1
        if i == 624:
            mt[0], i = mt[623], 1
    for _ in range(623):
        prev = mt[i-1]
        mt[i] = ((mt[i] ^ ((prev ^ (prev >> 30)) * 1566083941)) - i) & MASK
        i += 1
        if i == 624:
            mt[0], i = mt[623], 1
    mt[0] = 0x80000000
    rng = random.Random()
    rng.setstate((3, (*mt, 624), None))
    return rng


def next_wall(rng):
    source = [rng.getrandbits(32) for _ in range(288)]
    hashed = b"".join(hashlib.sha512(struct.pack("<32I", *source[i:i+32])).digest()
                      for i in range(0, 288, 32))
    words = struct.unpack("<144I", hashed)
    wall = list(range(136))
    for i in range(135):
        j = i + words[i] % (136-i)
        wall[i], wall[j] = wall[j], wall[i]
    return wall, [words[135] % 6, words[136] % 6]


def verify_document(root, counts):
    rng = rng_from_log(root.find("SHUFFLE").attrib["seed"])
    wall = None
    for e in root:
        tag, attrs = e.tag, e.attrib
        if tag == "INIT":
            wall, dice = next_wall(rng)
            hand = [[] for _ in range(4)]
            current = 136
            for n in (4, 4, 4, 1):
                for seat in range(4):
                    hand[(seat + int(attrs["oya"])) % 4].extend(wall[current-n:current])
                    current -= n
            for seat, tiles in enumerate(hand):
                counts["deal_checked"] += 1
                counts["deal_mismatch"] += sorted(map(int, attrs[f"hai{seat}"].split(","))) != sorted(tiles)
            seed = list(map(int, attrs["seed"].split(",")))
            counts["dice_checked"] += 1
            counts["dice_mismatch"] += seed[3:5] != dice
            counts["dora_checked"] += 1
            counts["dora_mismatch"] += seed[5] != wall[5]
            live, indicators, replacement = 83, 1, 0
            pending = False
        elif wall is not None and tag in ("DORA", "AGARI", "N"):
            if tag == "DORA":
                counts["dora_checked"] += 1
                counts["dora_mismatch"] += int(attrs["hai"]) != wall[5+2*indicators]
                indicators += 1
            elif tag == "AGARI":
                for key, offset in (("doraHai", 5), ("doraHaiUra", 4)):
                    if key in attrs:
                        for i, tile in enumerate(map(int, attrs[key].split(","))):
                            counts[key + "_checked"] += 1
                            counts[key + "_mismatch"] += tile != wall[offset + 2*i]
            else:
                code = int(attrs["m"])
                if not code & 4 and (code & 0x10 or not code & 0x28):
                    pending = True
        elif wall is not None and tag[0] in "TUVW" and tag[1:].isdigit():
            if pending:
                counts["rinshan_checked"] += 1
                counts["rinshan_mismatch"] += int(tag[1:]) != wall[replacement ^ 1]
                replacement += 1
                pending = False
            else:
                counts["draw_checked"] += 1
                counts["draw_mismatch"] += int(tag[1:]) != wall[live]
                live -= 1


def run(database, plan, count):
    totals = Counter()
    locators = plan.read_text().splitlines()[:count]
    with sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True) as db:
        for locator in locators:
            (blob,) = db.execute("SELECT log_content FROM logs WHERE log_id=?", (locator,)).fetchone()
            doc = ET.fromstring(bz2.decompress(blob))
            if int(doc.find("GO").attrib["type"]) & 0x10:
                continue
            totals["matches"] += 1
            verify_document(doc, totals)
    return totals


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("database", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("--limit", type=int, default=128)
    a = p.parse_args()
    if not 1 <= a.limit <= 1000:
        p.error("limit must be 1..1000")
    print(json.dumps(run(a.database, a.plan, a.limit), indent=2))
