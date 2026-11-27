#!/usr/bin/env python3
"""Audit a direct Parquet release, including an independent local-action sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.shanten import Shanten
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mahjong_ai.data.lake_training import _snapshot_map, _streams
from validate_riyu_lake import committed_shards, structural_standard, tile_code


ACTIONS = (
    "discard", "riichi", "chi", "pon", "daiminkan", "ankan", "shouminkan",
    "ron", "tsumo", "abort", "pass",
)
CALLS = (2, 3, 4)
KANS = (4, 5, 6)
RIICHI = ("accepted", "double_accepted")


def action_codes(row):
    return [a["semantic"] for a in row["actions"]]


def digest_axis(actions):
    canonical = [{"id": a["id"].hex(), "semantic": list(a["semantic"])} for a in actions]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).digest()


def number_counts(tiles):
    counts = [0] * 34
    for tile in tiles:
        counts[tile // 2] += 1
    return counts


def kuikae(kind, called, consumed):
    if kind == called:
        return True
    low, high = sorted(consumed)
    if high != low + 1 or called >= 27:
        return False
    flank = high + 1 if called == low - 1 else low - 1 if called == high + 1 else -1
    return flank // 9 == called // 9 and flank == kind


def inspect_cells(cells, prefix, counts):
    assert len(cells) == 3 and len({c["seat"] for c in cells}) == 3
    for cell in cells:
        legal, loss = cell["ron_legal"], cell["conditional_loss"]
        counts[prefix + "_cells"] += 1
        if legal is None:
            assert cell["legality_status"] == "unknown_furiten"
            counts[prefix + "_unknown"] += 1
        if legal is True:
            counts[prefix + "_legal"] += 1
        if loss is not None:
            assert legal is True and cell["loss_status"] == "valid_single_ron"
            assert loss == cell["no_ura_no_pao_loss"] and loss > 0
            counts[prefix + "_priced"] += 1
        if cell["loss_status"] == "pending_kan_dora":
            assert loss is None
            counts[prefix + "_pending_dora"] += 1


def check_profile(profile, shanten, known, counts):
    tiles = list(profile["concealed"])
    if profile["drawn"] is not None:
        tiles.append(profile["drawn"])
    hand = number_counts(tiles)
    assert max(hand) <= 4
    fixed = profile["meld_count"]
    if len(tiles) + 3 * fixed not in (13, 14):
        raise AssertionError("unexpected number of tiles in local profile")
    regular = structural_standard(shanten, hand, fixed)
    if regular != shanten.calculate_shanten_for_regular_hand(hand):
        counts["local_quad_convention_difference"] += 1
    pairs = shanten.calculate_shanten_for_chiitoitsu_hand(hand) if fixed == 0 else None
    orphans = shanten.calculate_shanten_for_kokushi_hand(hand) if fixed == 0 else None
    assert (regular, pairs, orphans) == (profile["standard"], profile["seven_pairs"], profile["thirteen_orphans"])
    best = min(n for n in (regular, pairs, orphans) if n is not None)
    assert best == profile["minimum"]
    if len(tiles) + 3 * fixed == 14:
        assert profile["ukeire_mask"] is None and profile["wait_mask"] is None
    else:
        improved = []
        for i, count in enumerate(hand):
            if count < 4:
                hand[i] += 1
                improved_value = min(structural_standard(shanten, hand, fixed),
                                     shanten.calculate_shanten(hand, not fixed, not fixed))
                if improved_value < best:
                    improved.append(i)
                hand[i] -= 1
        mask = sum(1 << i for i in improved)
        assert mask == profile["ukeire_mask"]
        assert profile["wait_mask"] == (mask if best == 0 else None)
        assert profile["theoretical_copies"] == sum(4 - hand[i] for i in improved)
        if known is not None:
            assert profile["visible_adjusted_copies"] == sum(4 - known[i] for i in improved)
    counts["independent_attack_profiles"] += 1


def stable_known(snapshot):
    tiles = [tile_code(t) for t in snapshot["hand"] + snapshot["dora"]]
    if snapshot["drawn"] is not None:
        tiles.append(tile_code(snapshot["drawn"]))
    for view in snapshot["players"]:
        for discard in view["discards"]:
            if discard["called_by"] is None:
                tiles.append(tile_code(discard["tile"]))
        for meld in view["melds"]:
            tiles.extend(map(tile_code, meld["tiles"]))
    return number_counts(tiles)


def check_owned_dora(profile, meld_codes, indicators):
    owned = [*profile["concealed"], *meld_codes]
    if profile["drawn"] is not None:
        owned.append(profile["drawn"])
    dora_kinds = []
    for indicator in indicators:
        kind = tile_code(indicator) // 2
        if kind < 27:
            dora_kinds.append(kind // 9 * 9 + (kind + 1) % 9)
        elif kind < 31:
            dora_kinds.append(27 + (kind - 26) % 4)
        else:
            dora_kinds.append(31 + (kind - 30) % 3)
    assert profile["red_count"] == sum(t % 2 for t in owned)
    assert profile["visible_dora_count"] == sum(dora_kinds.count(t // 2) for t in owned)


def score_ron(snapshot, seat, hand, code, *, call=False, chankan=False):
    """Independent scoring from semantic copies, public context and the sampled oracle hand."""
    player = snapshot["players"][seat]
    taken = set()

    def take(code):
        kind, red = divmod(code, 2)
        slots = (0,) if red else range(1, 4) if kind in (4, 13, 22) else range(4)
        ident = next(4 * kind + slot for slot in slots if 4 * kind + slot not in taken)
        taken.add(ident)
        return ident

    won = take(code)
    concealed = [take(tile) for tile in hand]
    melds = []
    for meld in player["melds"]:
        kind = meld["kind"]
        melds.append(Meld(kind if kind in ("chi", "pon", "shouminkan") else "kan",
                          [take(tile_code(t)) for t in meld["tiles"]], opened=kind != "ankan"))
    round_info = snapshot["round"]
    riichi = player["riichi"]
    # The declaration discard was not won; a completed call is after acceptance.
    if call:
        riichi = {"declared": "accepted", "double_declared": "double_accepted"}.get(riichi, riichi)
    config = HandConfig(
        is_riichi=riichi in RIICHI, is_daburu_riichi=riichi == "double_accepted",
        is_ippatsu=snapshot["ippatsu"][seat] and not call, is_chankan=chankan,
        is_houtei=snapshot["remaining"] == 0,
        player_wind=27 + (seat - round_info["dealer"]) % 4,
        round_wind=27 + ("east", "south", "west", "north").index(round_info["wind"]),
        tsumi_number=round_info["honba"],
        options=OptionalRules(has_open_tanyao=True, has_aka_dora=True, has_double_yakuman=False),
    )
    result = HandCalculator().estimate_hand_value(
        [won, *concealed, *(tile for meld in melds for tile in meld.tiles)], won, melds=melds,
        dora_indicators=[tile_code(t) // 2 * 4 for t in snapshot["dora"]], config=config,
    )
    return None if result.error else result.cost["main"] + result.cost["main_bonus"]


def compare_ron(cells, code, snapshot, belief, counts, *, call=False, chankan=False, ankan=False):
    full, lite = belief
    for cell in cells:
        seat = cell["seat"]
        hand, info = full[seat], lite[seat]
        counts["independent_ron_cells"] += 1
        if cell["ron_legal"] is None:
            counts["independent_ron_unknown"] += 1
            continue
        waits = info["wait_mask"] or 0
        known_furiten = any(info[f] is True for f in ("discard_furiten", "temporary_furiten", "riichi_furiten"))
        expected = None
        if not ankan and not known_furiten and waits & (1 << (code // 2)):
            tiles = list(hand["concealed_counts"])
            # Copy assignment is only an input format for the independent scorer.
            concealed = []
            for kind, number in enumerate(tiles):
                red = kind in (4, 13, 22) and bool(hand["red_mask"] & (1 << (kind // 9)))
                concealed.extend([2 * kind] * (number - int(red)))
                if red:
                    concealed.append(2 * kind + 1)
            if hand["drawn"] is not None:
                concealed.append(hand["drawn"])
            if tiles[code // 2] < 4:
                expected = score_ron(snapshot, seat, concealed, code, call=call, chankan=chankan)
        assert cell["ron_legal"] == (expected is not None), ("reference ron legality", cells, code)
        if expected is not None:
            counts["independent_ron_positive"] += 1
            assert cell["no_ura_no_pao_loss"] == expected, ("reference visible ron price", expected, cell, code)


def public_context(events):
    context = {}
    windows = 0
    for key, stream in events.items():
        turn, riichi = 0, 0
        for seq, raw in stream.items():
            event = json.loads(raw)
            match event["type"]:
                case "draw":
                    turn += 1
                case "riichi" if event["step"] == "accepted":
                    riichi |= 1 << event["player"]
                case "discard":
                    windows += 1
                case "kan" if event["meld"]["kind"] in ("ankan", "shouminkan"):
                    windows += 1
            context[(*key[:2], seq)] = (turn, riichi)
    return context, windows


def read_table(directory, table, columns=None):
    return pq.ParquetFile(directory / (table + ".parquet")).read(columns=columns)


def selections_from_log(directory):
    rows = read_table(directory, "restricted/events", [
        "match_id", "round", "source_ordinal", "type", "seat", "from", "tile", "meld", "step", "reason",
    ]).to_pylist()
    logs = defaultdict(dict)
    active_draw = {}
    for row in rows:
        mid = row["match_id"]
        logs[mid][row["source_ordinal"]] = row
        if row["type"] == "draw":
            active_draw[mid] = row["tile"]["id"]
        elif row["type"] in ("call", "start", "discard"):
            drawn = active_draw.pop(mid, None)
            if row["type"] == "discard":
                row["tsumogiri"] = drawn == row["tile"]["id"]
    return logs


def check_policy_evidence(d, p, code, events):
    e = events[p["evidence_ordinal"]]
    actor, operation = d["actor"], code[1]
    assert e["round"] == d["round"]
    if operation == 10:
        assert e["type"] == "draw"
    elif operation in (0, 1):
        if operation == 1:
            assert (e["type"], e["step"], e["seat"]) == ("riichi", 1, actor)
            if (ordinal := p["riichi_acceptance_ordinal"]) is not None:
                accepted = events[ordinal]
                assert (accepted["type"], accepted["step"], accepted["seat"]) == ("riichi", 2, actor)
            e = events[e["source_ordinal"] + 1]
            while e["type"] in ("players", "disconnect"):
                e = events[e["source_ordinal"] + 1]
        assert (e["type"], e["seat"]) == ("discard", actor)
        assert (e["tile"]["kind"] * 2 + e["tile"]["red"], e["tsumogiri"]) == (code[2], code[3])
    elif operation in (2, 3, *KANS):
        assert (e["type"], e["seat"]) == ("call", actor)
        meld = e["meld"]
        assert meld["kind"] == ACTIONS[operation]
        def semantic(tile):
            return tile["kind"] * 2 + tile["red"]

        if operation in CALLS:
            called, consumed = code[2], list(code[4:])
            assert called == semantic(meld["called"]) and code[3] == (meld["from"] - actor) % 4
            used = sorted(semantic(t) for t in meld["tiles"] if t["id"] != meld["called"]["id"])
            assert consumed == used
        elif operation == 5:
            assert all(tile["kind"] == code[2] for tile in meld["tiles"])
            assert sum(tile["red"] for tile in meld["tiles"]) == code[3]
        else:
            assert code[2] == semantic(meld["added"])
            assert code[3] == semantic(meld["called"]) and code[4] == (meld["from"] - actor) % 4
            assert list(code[5:]) == sorted(semantic(t) for t in meld["tiles"] if t["id"] != meld["added"]["id"])
    elif operation == 7:
        cursor = e["source_ordinal"]
        while events[cursor]["type"] == "win" and events[cursor]["seat"] != actor:
            cursor += 1
        win = events[cursor]
        assert win["type"] == "win" and win["seat"] == actor and code[2] == (win["from"] - actor) % 4
    elif operation == 8:
        assert (e["type"], e["from"], e["seat"]) == ("win", actor, actor)
    else:
        assert operation == 9 and (e["type"], e["reason"]) == ("draw_game", "yao9")


def scan_dense_axes(attack, defense, expected, counts):
    for name, table in (("attack", attack), ("defense", defense)):
        flat = pc.list_flatten(table["candidates"])
        assert pc.struct_field(flat, "action_index").to_pylist() == expected
        counts[name + "_candidates"] += len(flat)
        if name == "defense":
            opponents = pc.struct_field(flat, "opponents")
            assert pc.all(pc.equal(pc.list_value_length(opponents), 3)).as_py() is not False
            cells = pc.list_flatten(opponents)
            legal = pc.struct_field(cells, "ron_legal")
            loss = pc.struct_field(cells, "conditional_loss")
            assumption = pc.struct_field(cells, "no_ura_no_pao_loss")
            status = pc.struct_field(cells, "loss_status")
            counts["root_ron_cells"] += len(cells)
            counts["root_ron_unknown"] += legal.null_count
            counts["root_ron_legal"] += pc.sum(pc.cast(legal, "uint64")).as_py() or 0
            counts["root_ron_priced"] += len(cells) - loss.null_count
            invalid = pc.and_(pc.is_valid(loss), pc.invert(pc.fill_null(
                pc.and_(legal, pc.and_(pc.equal(status, "valid_single_ron"), pc.equal(loss, assumption))), False)))
            assert not pc.any(invalid).as_py()


def audit(release, max_shards, samples, check_hash, follow=False):
    result, _ = audit_partition(release, max_shards, samples, check_hash, follow)
    return result


def audit_partition(release, max_shards, samples, check_hash, follow, rank=0, workers=1):
    expected_build, shards = committed_shards(release, max_shards, follow)
    assert expected_build["teacher_spec"] == "local-actions-v1"
    counts, seen, by_kind, response = Counter(), set(), defaultdict(Counter), defaultdict(Counter)
    files, example = Counter(), None
    shanten = Shanten()
    for shard_index, shard in enumerate(shards):
        if shard_index % workers != rank:
            continue
        directory = release / shard["file"]
        footer = shard["footer"]
        build, advertised = footer["header"]["build"], footer["files"]
        assert build == expected_build
        counts.update({"status_" + key: value for key, value in footer["statuses"].items()})
        for name, spec in advertised.items():
            path = directory / (name + ".parquet")
            assert path.stat().st_size == spec["bytes"]
            if check_hash:
                with path.open("rb") as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"]
            assert pq.ParquetFile(path).metadata.num_rows == spec["rows"]
            files[name] += spec["bytes"]
        split = read_table(directory, "governance/split").to_pylist()
        for row in split:
            mid = row["match_id"]
            assert mid not in seen
            seen.add(mid)
            counts["matches_" + row["split"]] += 1
        counts["rounds"] += advertised["targets/round"]["rows"]
        decisions = read_table(directory, "inputs/decisions")
        policy = read_table(directory, "targets/policy")
        attack = read_table(directory, "targets/attack")
        defense = read_table(directory, "targets/defense")
        assert decisions.num_rows == policy.num_rows == attack.num_rows == defense.num_rows
        for table in (policy, attack, defense):
            assert decisions["decision_id"].equals(table["decision_id"])
            assert decisions["action_set_hash"].equals(table["action_set_hash"])
        events = read_table(directory, "inputs/events")
        public, windows = public_context(_streams(events.filter(pc.equal(events["seat"], 0)).to_pylist(), "event"))
        counts["possible_response_seats"] += 3 * windows
        shantens = attack["shanten"].to_pylist()
        source = selections_from_log(directory)
        root, policies = decisions.to_pylist(), policy.to_pylist()
        index = {d["decision_id"]: i for i, d in enumerate(root)}
        assert len(index) == len(root)
        counts["decisions"] += len(root)
        expected, local_expected = [], {}
        for i, (d, p) in enumerate(zip(root, policies, strict=True)):
            codes = action_codes(d)
            assert len(codes) == len(set(codes)) == len({a["id"] for a in d["actions"]})
            choice = p["choice"]
            chosen = codes[choice][1] if choice is not None else None
            status = p["resolution"]
            assert p["trainable"] == (p["legal_status"] == "complete" and choice is not None)
            assert (chosen == 10) == (status == "confirmed_pass")
            assert status != "priority_censored" or choice is None
            assert p["riichi_accepted"] is None if chosen != 1 else p["riichi_accepted"] == (p["riichi_acceptance_ordinal"] is not None)
            if chosen == 1:
                counts["riichi_accepted" if p["riichi_accepted"] else "riichi_not_accepted"] += 1
            if chosen == 0 and bytes((1, 1, *codes[choice][2:])) in codes:
                counts["dama_paired_selected"] += 1
            if chosen is not None:
                check_policy_evidence(d, p, codes[choice], source[d["match_id"]])
                counts["policy_evidence_verified"] += 1
            expected.extend(j for j, code in enumerate(codes) if code[1] <= 1)
            additional = [j for j, code in enumerate(codes) if code[1] not in (0, 10)]
            if additional:
                local_expected[d["decision_id"]] = additional
            counts["policy_" + status] += 1
            is_response = d["window"] != "turn"
            counts["stored_response_seats"] += is_response
            kinds = Counter(code[1] for code in codes)
            if is_response and any(kind in CALLS for kind in kinds):
                result = "call" if chosen in CALLS else "ron" if chosen == 7 else status
                counts["call_window_" + result] += 1
            for kind, n in kinds.items():
                stats = by_kind[ACTIONS[kind]]
                stats["opportunities"] += 1
                stats["candidates"] += n
                stats["selected"] += chosen == kind
                if is_response and kind in (*CALLS, 7):
                    turn, riichi = public[(d["match_id"], d["round"], d["observation_seq"])]
                    group = response[f"{ACTIONS[kind]}/{turn//24}/{shantens[i]}/{bool(riichi)}"]
                    group["opportunities"] += 1
                    group["selected"] += chosen == kind
                    group["confirmed_pass"] += status == "confirmed_pass"
                    group["unknown"] += status in ("priority_censored", "incomplete_legal_set")
        scan_dense_axes(attack, defense, expected, counts)
        local = read_table(directory, "targets/local_actions").to_pylist() if local_expected else []
        assert len(local) == len(local_expected)
        maybe_sample = {}
        for row in local:
            did = row["decision_id"]
            i = index[did]
            d, p = root[i], policies[i]
            codes = action_codes(d)
            assert row["action_set_hash"] == d["action_set_hash"]
            assert [c["action_index"] for c in row["candidates"]] == local_expected.pop(did)
            counts["local_decisions"] += 1
            before = row["before"]
            for parent in row["candidates"]:
                j = parent["action_index"]
                code = codes[j]
                op = code[1]
                children = parent["successors"]
                counts["local_" + ACTIONS[op]] += 1
                if op in (2, 3):
                    called, source_relative, a, b = code[2:]
                    del source_relative
                    expected_hand = Counter(before["concealed"])
                    expected_hand.subtract((a, b))
                    assert min(expected_hand.values()) >= 0
                    assert parent["after"]["concealed"] == bytes(sorted(expected_hand.elements()))
                    assert parent["successor_set_hash"] == digest_axis(children)
                    allowed = {t for t, n in expected_hand.items() if n and not kuikae(t//2, called//2, (a//2, b//2))}
                    expected_child_codes = {bytes((1, 0, t, 0)) for t in allowed}
                    assert {c["semantic"] for c in children} == expected_child_codes
                    assert len(children) == len(allowed) == len({c["id"] for c in children})
                    for child in children:
                        expect = expected_hand.copy()
                        expect[child["semantic"][2]] -= 1
                        assert child["after_discard"]["concealed"] == bytes(sorted(expect.elements()))
                        assert child["id"] == hashlib.sha256(child["semantic"]).digest()[:8]
                        inspect_cells(child["ron"], "call_ron", counts)
                        counts["successors"] += 1
                    if chosen_child := next((child for child in children if any(c["ron_legal"] for c in child["ron"])), None):
                        maybe_sample.setdefault("call_positive", (row, parent, chosen_child))
                    if p["choice"] != j:
                        maybe_sample.setdefault("call_unchosen", (row, parent, children[-1]))
                        if example is None:
                            example = {
                                "parent": list(code), "before_shanten": before["minimum"],
                                "after_call_shanten": parent["after"]["minimum"],
                                "successors": [child["semantic"][2] for child in children],
                                "example_child": {
                                    "tile_code": children[-1]["semantic"][2],
                                    "shanten": children[-1]["after_discard"]["minimum"],
                                    "visible_ukeire": children[-1]["after_discard"]["visible_adjusted_copies"],
                                    "genbutsu_mask": children[-1]["genbutsu_mask"],
                                },
                            }
                else:
                    assert not children and parent["successor_set_hash"] is None
                if op == 1:
                    assert codes[parent["dama_index"]] == bytes((1, 0, *code[2:]))
                    assert parent["deposit_on_acceptance"] == 1000
                if op in KANS:
                    assert parent["after"] is not None and parent["rinshan_if_continues"] is True
                    if op != 4:
                        inspect_cells(parent["kan_robbery"], "chankan", counts)
                        maybe_sample.setdefault(ACTIONS[op], (row, parent, None))
                        if any(cell["ron_legal"] for cell in parent["kan_robbery"]):
                            maybe_sample.setdefault("kan_positive", (row, parent, None))
                    else:
                        assert not parent["kan_robbery"]
        if samples and root:
            events = _streams(events.to_pylist(), "event")
            sample = {i for i in range((shard_index * 37) % len(root), len(root), max(1, len(root)//samples))}
            sample.update(index[s[0]["decision_id"]] for s in maybe_sample.values())
            snapshots = _snapshot_map(root=directory.parent, shard=(directory.name, True),
                decisions=[root[i] for i in sample], checkpoint=True, events=events)
            full = read_table(directory, "targets/belief_full")["hands"]
            lite = read_table(directory, "targets/belief_lite", ["opponents"])["opponents"]

            def sample_view(i):
                belief = ({h["seat"]: h for h in full[i].as_py()}, {h["seat"]: h for h in lite[i].as_py()})
                return json.loads(snapshots[root[i]["decision_id"]]), belief

            for i in sample:
                cells = defense["candidates"][i].as_py()
                if not cells:
                    continue
                selected = policies[i]["choice"]
                different = next((c for c in reversed(cells) if c["action_index"] != selected), None)
                snapshot, belief = sample_view(i)
                for c in cells:
                    if c is not different and c["action_index"] != selected:
                        continue
                    code = root[i]["actions"][c["action_index"]]["semantic"][2]
                    compare_ron(c["opponents"], code, snapshot, belief, counts)
                    own = list(map(tile_code, snapshot["hand"]))
                    if snapshot["drawn"] is not None:
                        own.append(tile_code(snapshot["drawn"]))
                    own.remove(code)
                    hand = number_counts(own)
                    fixed = len(snapshot["players"][root[i]["actor"]]["melds"])
                    reference = shanten.calculate_shanten(hand, not fixed, not fixed)
                    minimum = min(structural_standard(shanten, hand, fixed), reference)
                    if minimum != reference:
                        counts["root_quad_convention_difference"] += 1
                    attack_cell = next(a for a in attack["candidates"][i].as_py()
                                       if a["action_index"] == c["action_index"])
                    assert minimum == attack_cell["shanten_after"], (
                        "root attack shanten reference", shard_index, i, fixed, hand,
                        minimum, attack_cell["shanten_after"], code,
                    )
                    counts["independent_root_attack"] += 1
                    counts["sample_root_factual" if c["action_index"] == selected else "sample_root_unchosen"] += 1
            for name, (row, parent, child) in maybe_sample.items():
                i = index[row["decision_id"]]
                snapshot, belief = sample_view(i)
                code = root[i]["actions"][parent["action_index"]]["semantic"]
                if child:
                    known = stable_known(snapshot)
                    check_profile(row["before"], shanten, known, counts)
                    check_profile(parent["after"], shanten, None, counts)
                    check_profile(child["after_discard"], shanten, known, counts)
                    tile = child["semantic"][2]
                    actor = root[i]["actor"]
                    owned_melds = [tile_code(t) for meld in snapshot["players"][actor]["melds"] for t in meld["tiles"]]
                    check_owned_dora(row["before"], owned_melds, snapshot["dora"])
                    owned_melds.extend((code[2], code[4], code[5]))
                    check_owned_dora(parent["after"], owned_melds, snapshot["dora"])
                    check_owned_dora(child["after_discard"], owned_melds, snapshot["dora"])
                    genbutsu = sum(1 << seat for seat, p in enumerate(snapshot["players"]) if seat != actor
                                   and any(tile_code(d["tile"])//2 == tile//2 for d in p["discards"]))
                    assert genbutsu == child["genbutsu_mask"]
                    compare_ron(child["ron"], tile, snapshot, belief, counts, call=True)
                else:
                    compare_ron(parent["kan_robbery"], code[2], snapshot, belief, counts,
                                chankan=True, ankan=code[1] == 5)
                counts["sample_" + name] += 1
        if (shard_index + 1) % 10 == 0:
            print(f"checked {shard_index + 1} release shards", file=sys.stderr)
    counts["matches"] = len(seen)
    counts["inferred_no_non_pass_response"] = counts["possible_response_seats"] - counts["stored_response_seats"]
    return {"counts": counts, "actions": by_kind, "response_groups": response,
            "parquet_bytes": files, "example": example}, seen


def parallel_audit(release, max_shards, samples, check_hash, follow, workers):
    if workers == 1:
        return audit(release, max_shards, samples, check_hash, follow)
    result = {
        "counts": Counter(), "actions": defaultdict(Counter), "response_groups": defaultdict(Counter),
        "parquet_bytes": Counter(), "example": None,
    }
    verified = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(audit_partition, release, max_shards, samples, check_hash, follow, rank, workers)
                   for rank in range(workers)]
        for job in pending:
            part, seen = job.result()
            if verified & seen:
                raise ValueError("duplicate match assignment across audit partitions")
            verified |= seen
            for key in ("counts", "parquet_bytes"):
                result[key].update(part[key])
            for key in ("actions", "response_groups"):
                for name, stats in part[key].items():
                    result[key][name].update(stats)
            if result["example"] is None:
                result["example"] = part["example"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4, help="independent sampled decisions per shard")
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args()
    if not 1 <= a.workers <= 8:
        parser.error("--workers must be in 1..8")
    print(json.dumps(parallel_audit(
        a.release, a.max_shards, a.samples, not a.skip_hash, a.follow, a.workers,
    ), indent=2, sort_keys=True))
