#!/usr/bin/env python3
"""Independent match/defense/Belief audits for local direct Parquet artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from mahjong.shanten import Shanten
import pyarrow.parquet as pq

from mahjong_ai.data.lake_loader import TILE_INDEX


def tile_code(tile):
    return 2 * TILE_INDEX[tile["kind"]] + bool(tile["red"])


def winning_transfers(wins, dealer):
    """Independent Tenhou payout calculation from the logged base hand values."""
    first = json.loads(wins[0]["attrs"])
    honba, sticks = (int(x) for x in first["ba"].split(","))
    discarder = wins[0]["from"]
    head = min(wins, key=lambda w: (w["seat"] - discarder) % 4)["seat"]
    for w in wins:
        seat, points, liable = w["seat"], w["points"], w["responsibility"]
        bonus = honba * 300 if seat == head else 0
        expected = [0] * 4
        if liable is not None:
            if seat == discarder or liable == discarder:
                expected[liable] -= points + bonus
            else:
                expected[discarder] -= points // 2
                expected[liable] -= points - points // 2 + bonus
        elif seat == discarder:
            if seat == dealer:
                for i in range(4):
                    if i != seat:
                        expected[i] -= points // 3 + 100 * honba
            else:
                child = (points + 399) // 400 * 100
                for i in range(4):
                    if i != seat:
                        expected[i] -= (points - 2 * child if i == dealer else child) + 100 * honba
        else:
            expected[discarder] -= points + bonus
        expected[seat] = -sum(expected) + (1000 * sticks if seat == head else 0)
        yield w, (expected == w["score_change"]["delta"])


def hand_at(events, wanted, own):
    """Reconstruct one seat's hand from its own, already-projected event list."""
    result = {}
    for event in events:
        seq, tag = event["sequence"], event["type"]
        if tag == "round_start":
            hand = Counter(map(tile_code, event["initial_hand"]))
            drawn = None
            pons = {}
        elif tag == "draw" and event["tile"] is not None:
            assert own == event["player"]
            assert drawn is None
            drawn = tile_code(event["tile"])
        elif event.get("player") == own:
            if tag == "discard":
                value = tile_code(event["tile"])
                if event["tsumogiri"]:
                    assert drawn == value
                else:
                    hand[value] -= 1
                    assert hand[value] >= 0
                    if drawn is not None:
                        hand[drawn] += 1
                drawn = None
            elif tag in ("chi", "pon", "kan"):
                meld = event["meld"]
                used = Counter(map(tile_code, meld["tiles"]))
                called = meld["called_tile"]
                kind = meld["kind"]
                if kind == "shouminkan":
                    used.subtract(pons[tile_code(called) // 2])
                elif called is not None:
                    if kind == "pon":
                        pons[tile_code(called) // 2] = used.copy()
                    used[tile_code(called)] -= 1
                if drawn is not None:
                    hand[drawn] += 1
                    drawn = None
                hand.subtract(used)
                assert all(n >= 0 for n in hand.values())
        if seq in wanted:
            counts = [0] * 34
            red = 0
            for code, n in hand.items():
                counts[code // 2] += n
                if n and code % 2:
                    red |= 1 << (code // 18)
            result[seq] = (bytes(counts), red, drawn)
    return result


def audit(root, max_shards=0, belief_stride=101):
    status, coverage, anomalies = Counter(), Counter(), Counter()
    manifest = json.loads((root / "manifest.json").read_text())
    shanten = Shanten()
    for shard in manifest["shards"][:max_shards or None]:
        directory = root / shard["file"]
        status.update(shard["footer"]["statuses"])

        def read(table, columns=None):
            if table not in shard["footer"]["files"]:
                return []
            return pq.ParquetFile(directory / (table + ".parquet")).read(columns=columns).to_pylist()

        canonical = defaultdict(list)
        for e in read("restricted/events"):
            canonical[e["match_id"]].append(e)
        match_headers = {m["match_id"]: m for m in read("restricted/matches")}
        for match in match_headers.values():
            if (error := match["diagnostic"]) is not None:
                anomalies["replay: " + error.rsplit(": ", 1)[-1].split(" {")[0]] += 1
        targets = {r["decision_id"]: r for r in read("targets/policy")}
        defense = {r["decision_id"]: r["candidates"] for r in read("targets/defense")}
        decisions = read("inputs/decisions")
        choices = {}
        for d in decisions:
            p = targets[d["decision_id"]]
            if p["choice_status"] == "censored":
                assert not p["trainable"] and p["choice"] is None
                coverage["censored"] += 1
            index = p["choice"]
            if index is None or d["window"] != "turn" or d["actions"][index]["semantic"][1] > 1:
                continue
            ordinal = p["evidence_ordinal"]
            if d["actions"][index]["semantic"][1] == 1:
                seq = canonical[d["match_id"]]
                ordinal = next(e["source_ordinal"] for e in seq[ordinal+1:] if e["type"] == "discard")
            choices[(d["match_id"], ordinal)] = next(c for c in defense[d["decision_id"]]
                                                      if c["action_index"] == index)["opponents"]
        accepted = {row["match_id"] for row in read("targets/match")}
        for match in accepted:
            events = canonical[match]
            actual_discards = {}
            hands_wins, dealers = defaultdict(list), {}
            last_discard = None
            previous = None
            for e in events:
                kind = e["type"]
                if kind == "start":
                    last_discard = None
                    dealers[e["round"]] = e["dealer"]
                    if e["hand_number"] >= 8:
                        coverage["west_extension_rounds"] += 1
                elif kind == "discard":
                    last_discard = e["source_ordinal"]
                elif kind == "call":
                    last_discard = None
                elif kind == "draw_game":
                    coverage["draw_" + e["reason"]] += 1
                elif kind == "win":
                    hands_wins[e["round"]].append(e)
                    winner, loser = e["seat"], e["from"]
                    if e["responsibility"] is not None:
                        coverage["pao"] += 1
                    if previous and previous["type"] == "win":
                        coverage["double_ron_rounds"] += 1
                    if winner != loser and last_discard is None:
                        coverage["chankan"] += 1
                    elif winner != loser:
                        actual_discards.setdefault(last_discard, []).append(e)
                previous = e
            for round_number, wins in hands_wins.items():
                for win, correct in winning_transfers(wins, dealers[round_number]):
                    if not correct:
                        anomalies["independent settlement mismatch"] += 1
                    else:
                        coverage["independent_settlement"] += 1
                        coverage["independent_pao"] += win["responsibility"] is not None
            m = match_headers[match]
            if m["final_scores"] and len(set(m["final_scores"])) < 4:
                adjustments = m["final_adjustments"]
                if adjustments and len(set(adjustments)) == 4:
                    ranks = sorted(range(4), key=lambda seat: -adjustments[seat])
                    expected = [1 + ranks.index(seat) for seat in range(4)]
                    if m["final_ranks"] != expected:
                        anomalies["final tie-break disagrees with source placement"] += 1
                    coverage["tie_break_verified_from_adjustments"] += 1
            for ordinal, wins in actual_discards.items():
                coverage["riichi_tile_ron"] += events[ordinal-1]["type"] == "riichi" and events[ordinal-1]["step"] == 1
                possible = choices.get((match, ordinal))
                if possible is None:
                    anomalies["factual ron lacks selected discard defense label"] += len(wins)
                    continue
                if len(wins) > 1:
                    coverage["factual_multi_ron_winners"] += len(wins)
                for win in wins:
                    cell = next(c for c in possible if c["seat"] == win["seat"])
                    if cell["ron_legal"] is None:
                        coverage["factual_ron_unknown"] += 1
                    elif not cell["ron_legal"]:
                        anomalies["factual ron labelled illegal"] += 1
                    else:
                        coverage["factual_ron_legal"] += 1
                    if cell["conditional_loss"] is not None and len(wins) == 1:
                        paid = -win["score_change"]["delta"][win["from"]]
                        if paid != cell["conditional_loss"]:
                            anomalies["single factual ron loss mismatch"] += 1
                        else:
                            coverage["factual_single_ron_loss"] += 1
                    elif win["ura"]:
                        coverage["factual_ura_observations"] += 1
                        if (len(wins) == 1 and win["responsibility"] is None
                                and cell["no_ura_no_pao_loss"] is not None):
                            paid = -win["score_change"]["delta"][win["from"]]
                            estimate = cell["no_ura_no_pao_loss"]
                            if paid > estimate:
                                coverage["factual_loss_above_no_ura"] += 1
                            elif paid == estimate:
                                coverage["factual_loss_equal_to_no_ura"] += 1
                            else:
                                anomalies["factual loss below no-ura candidate"] += 1
        if belief_stride:
            picked = {d["decision_id"]: d for d in decisions[::belief_stride]}
            needed = defaultdict(set)
            full = [b for b in read("targets/belief_full") if b["decision_id"] in picked]
            lite = {b["decision_id"]: b["opponents"] for b in read("targets/belief_lite")
                    if b["decision_id"] in picked}
            for b in full:
                d = picked[b["decision_id"]]
                for h in b["hands"]:
                    needed[(d["match_id"], d["round"], h["seat"])].add(d["observation_seq"])
            streams = defaultdict(list)
            for e in read("inputs/events"):
                key = (e["match_id"], e["round"], e["seat"])
                if key in needed and e["seq"] <= max(needed[key]):
                    streams[key].append(json.loads(e["event"]))
            recovered = {key: hand_at(events, needed[key], key[-1]) for key, events in streams.items()}
            for b in full:
                d = picked[b["decision_id"]]
                independent = {}
                for h, auxiliary in zip(b["hands"], lite[b["decision_id"]], strict=True):
                    key = (d["match_id"], d["round"], h["seat"])
                    expect = recovered[key][d["observation_seq"]]
                    got = (h["concealed_counts"], h["red_mask"], h["drawn"])
                    coverage["belief_hands"] += 1
                    if expect != got:
                        anomalies["belief reconstruction mismatch"] += 1
                    counts = list(got[0])
                    if got[2] is not None:
                        counts[got[2] // 2] += 1
                    assert sum(counts) == auxiliary["hand_size"]
                    # The Python reference infers fixed meld count from normal hand length.
                    if sum(counts) + 3 * auxiliary["meld_count"] not in (13, 14):
                        coverage["belief_pending_kan_shape"] += 1
                        continue
                    expected_shanten = shanten.calculate_shanten(counts)
                    if expected_shanten != auxiliary["shanten"]:
                        anomalies["independent shanten mismatch"] += 1
                    if auxiliary["wait_mask"] is not None:
                        waits = 0
                        for i, n in enumerate(counts):
                            if n < 4:
                                counts[i] += 1
                                if shanten.calculate_shanten(counts) == -1:
                                    waits |= 1 << i
                                counts[i] -= 1
                        coverage["belief_waits"] += 1
                        if waits != auxiliary["wait_mask"]:
                            anomalies["independent wait mismatch"] += 1
                        independent[h["seat"]] = (waits, auxiliary)
                    for kind in ("temporary_furiten", "riichi_furiten", "discard_furiten"):
                        if auxiliary[kind] is None:
                            coverage[kind + "_unknown_or_not_applicable"] += 1
                        elif auxiliary[kind]:
                            coverage[kind + "_true"] += 1
                for candidate in defense[b["decision_id"]]:
                    code = d["actions"][candidate["action_index"]]["semantic"]
                    kind = code[2] // 2
                    for cell in candidate["opponents"]:
                        coverage["candidate_defense_cells"] += 1
                        if cell["conditional_loss"] is not None and (
                            cell["ron_legal"] is not True or cell["loss_status"] != "valid_single_ron"
                            or cell["conditional_loss"] != cell["no_ura_no_pao_loss"]
                        ):
                            anomalies["conditional loss status mismatch"] += 1
                        if cell["ron_legal"] is not True:
                            continue
                        coverage["candidate_positive_ron"] += 1
                        observed = independent.get(cell["seat"])
                        if observed is None or not observed[0] & (1 << kind) or any(
                            observed[1][f] is True
                            for f in ("temporary_furiten", "riichi_furiten", "discard_furiten")
                        ):
                            anomalies["positive ron contradicts reference wait or known furiten"] += 1
    return {"status": status, "coverage": coverage, "anomalies": anomalies}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("release", type=Path)
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--belief-stride", type=int, default=101)
    a = p.parse_args()
    print(json.dumps(audit(a.release, a.max_shards, a.belief_stride), indent=2))
