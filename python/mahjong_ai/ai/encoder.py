"""Versioned observable-state encoder and dynamic action features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mahjong_ai.schema import ACTION_TYPES, tile_index

ENCODER_VERSION = "observable-planes-v1"
ENCODER_PLANES = 49
ACTION_FEATURES = 83
DANGER_FEATURES = 24


@dataclass(frozen=True)
class EncodedState:
    planes: np.ndarray
    action_features: np.ndarray
    legal_actions: tuple[dict[str, Any], ...]
    danger_features: np.ndarray


class FeatureEncoder:
    version = ENCODER_VERSION

    def encode(
        self, state: dict[str, Any], legal_actions: list[dict[str, Any]]
    ) -> EncodedState:
        information = state.get("information", {})
        if information.get("opponent_hands_visible", False) or information.get("wall_order_visible", False):
            raise ValueError("oracle/private information is forbidden in the observable encoder")
        if "private_hands" in state or "wall" in state:
            raise ValueError("complete-state fields are forbidden in the observable encoder")
        planes = np.zeros((ENCODER_PLANES, 34), dtype=np.float32)
        hand_counts = np.zeros(34, dtype=np.int8)
        for tile in state["hand"]:
            index = tile_index(tile)
            hand_counts[index] += 1
            if tile.get("red", False):
                planes[4, index] += 1.0
        for copy in range(4):
            planes[copy] = hand_counts > copy
        if state.get("drawn_tile") is not None:
            index = tile_index(state["drawn_tile"])
            planes[5, index] = 1.0
            planes[6, index] = float(state["drawn_tile"].get("red", False))

        visible = hand_counts.astype(np.float32)
        if state.get("drawn_tile") is not None:
            visible[tile_index(state["drawn_tile"])] += 1
        for seat, player in enumerate(state["players"]):
            for discard in player["discards"]:
                if discard.get("called_by") is None:
                    index = tile_index(discard["tile"])
                    planes[7 + seat, index] += 0.25
                    visible[index] += 1
                if discard["tile"].get("red", False):
                    planes[44 + seat, tile_index(discard["tile"])] += 1.0
            for meld in player["melds"]:
                for tile in meld["tiles"]:
                    index = tile_index(tile)
                    planes[11 + seat, index] += 0.25
                    visible[index] += 1
                    if tile.get("red", False):
                        planes[44 + seat, index] += 1.0
            planes[15 + seat, :] = float(player["riichi"] in {"accepted", "double_accepted"})
            planes[19 + seat, :] = float(state["ippatsu_eligible"][seat])

        for indicator in state["dora_indicators"]:
            index = tile_index(indicator)
            planes[23, index] += 0.25
            visible[index] += 1
            if indicator.get("red", False):
                planes[48, index] += 1.0
        planes[24] = visible / 4.0
        planes[25] = np.clip(4.0 - visible, 0.0, 4.0) / 4.0

        round_state = state.get("round")
        observer = state["information"]["observer"]
        if round_state is not None:
            planes[26, :] = float(round_state["dealer"] == observer)
            planes[27, :] = observer / 3.0
            wind_index = {"east": 0, "south": 1, "west": 2, "north": 3}[round_state["wind"]]
            planes[28 + wind_index, :] = 1.0
            planes[38, :] = min(round_state["honba"], 10) / 10.0
            planes[39, :] = min(round_state["riichi_sticks"], 10) / 10.0
        for seat, score in enumerate(state["scores"]):
            planes[32 + seat, :] = np.clip(score / 50_000.0, -1.0, 2.0)
        planes[36, :] = min(state["turn"], 70) / 70.0
        planes[37, :] = state["remaining_tiles"] / 70.0
        phase = state["phase"]["phase"]
        planes[40, :] = float(phase == "awaiting_discard" and state["phase"].get("player") == observer)
        planes[41, :] = float(phase == "awaiting_response")
        planes[42, :] = float(phase == "awaiting_kan_response")
        planes[43, :] = float(state["players"][observer]["riichi"] in {"accepted", "double_accepted"})

        actions = tuple(legal_actions)
        action_features = np.stack([self.encode_action(action) for action in actions]) if actions else np.zeros((0, ACTION_FEATURES), dtype=np.float32)
        danger_features = self.encode_danger_features(state, list(actions))
        return EncodedState(
            planes=planes,
            action_features=action_features,
            legal_actions=actions,
            danger_features=danger_features,
        )

    def encode_danger_features(
        self, state: dict[str, Any], legal_actions: list[dict[str, Any]]
    ) -> np.ndarray:
        """Candidate x relative-opponent evidence, without hard safety rules."""

        output = np.zeros((len(legal_actions), 4, DANGER_FEATURES), dtype=np.float32)
        visible = np.zeros(34, dtype=np.float32)

        def mark(tile: dict[str, Any]) -> None:
            visible[tile_index(tile)] += 1.0

        for tile in state.get("hand") or []:
            mark(tile)
        if state.get("drawn_tile") is not None:
            mark(state["drawn_tile"])
        for tile in state.get("dora_indicators") or []:
            mark(tile)
        for player in state["players"]:
            for discard in player["discards"]:
                if discard.get("called_by") is None:
                    mark(discard["tile"])
            for meld in player["melds"]:
                for tile in meld["tiles"]:
                    mark(tile)

        observer = int(state["information"]["observer"])
        turn = min(float(state.get("turn", 0)) / 70.0, 1.0)
        dora_tiles = {
            _dora_from_indicator(tile_index(indicator))
            for indicator in state.get("dora_indicators") or []
        }
        for action_index, action in enumerate(legal_actions):
            nested = action.get("action") or {}
            tile = (
                action.get("tile")
                or action.get("discard")
                or action.get("called_tile")
                or nested.get("tile")
                or nested.get("called_tile")
            )
            if tile is None:
                continue
            target = tile_index(tile)
            for relative in range(1, 4):
                player = state["players"][(observer + relative) % 4]
                features = output[action_index, relative]
                features[0] = visible[target] / 4.0
                for offset, feature_index in zip((-2, -1, 1, 2), range(1, 5), strict=True):
                    neighbor = target + offset
                    same_suit = target < 27 and neighbor // 9 == target // 9
                    if 0 <= neighbor < 34 and same_suit:
                        features[feature_index] = visible[neighbor] / 4.0
                discards = player["discards"]
                discarded_indexes = [tile_index(item["tile"]) for item in discards]
                features[5] = float(target in discarded_indexes)
                features[6] = _latest_discard_recency(discarded_indexes, target)
                if target < 27:
                    rank = target % 9
                    lower = target - 3 if rank >= 3 else None
                    upper = target + 3 if rank <= 5 else None
                    features[7] = float(lower is not None and lower in discarded_indexes)
                    features[8] = float(upper is not None and upper in discarded_indexes)
                    features[9] = _latest_discard_recency(discarded_indexes, lower)
                    features[10] = _latest_discard_recency(discarded_indexes, upper)
                riichi = player["riichi"] in {
                    "declared", "double_declared", "accepted", "double_accepted"
                }
                features[11] = float(riichi)
                declaration = next(
                    (
                        index + 1
                        for index, discard in enumerate(discards)
                        if discard.get("riichi_declaration", False)
                    ),
                    0,
                )
                features[12] = declaration / max(len(discards), 1)
                features[13] = turn
                features[14] = min(len(player["melds"]), 4) / 4.0
                features[15] = float(target >= 27)
                features[16] = float(target < 27 and target % 9 in {0, 8})
                adjacent = [
                    target + offset
                    for offset in (-2, -1, 1, 2)
                    if target < 27
                    and 0 <= target + offset < 27
                    and (target + offset) // 9 == target // 9
                ]
                features[17] = float(any(visible[value] >= 4 for value in adjacent))
                features[18] = float(any(visible[value] == 3 for value in adjacent))
                features[19] = (
                    sum(bool(item.get("tsumogiri", False)) for item in discards)
                    / max(len(discards), 1)
                )
                recent = discards[-6:]
                features[20] = (
                    sum(not bool(item.get("tsumogiri", False)) for item in recent)
                    / max(len(recent), 1)
                )
                features[21] = float(target in dora_tiles)
                features[22] = float(tile.get("red", False))
                features[23] = float(action["type"] in {"discard", "riichi"})
        return output

    def encode_action(self, action: dict[str, Any]) -> np.ndarray:
        output = np.zeros(ACTION_FEATURES, dtype=np.float32)
        action_type = action["type"]
        output[ACTION_TYPES.index(action_type)] = 1.0
        tile = action.get("tile") or action.get("discard") or action.get("called_tile")
        nested = action.get("action", {})
        tile = tile or nested.get("tile") or nested.get("called_tile")
        if tile is not None:
            output[9 + tile_index(tile)] = 1.0
            output[81] = float(tile.get("red", False))
        source = action.get("from", nested.get("from"))
        if source is not None:
            output[43 + int(source)] = 1.0
        consumed = action.get("consume", nested.get("consume", nested.get("tiles", [])))
        for consumed_tile in consumed:
            output[47 + tile_index(consumed_tile)] += 0.25
        output[82] = float(action.get("tsumogiri", False))
        return output


def _latest_discard_recency(indexes: list[int], target: int | None) -> float:
    if target is None:
        return 0.0
    latest = next(
        (index for index in range(len(indexes) - 1, -1, -1) if indexes[index] == target),
        None,
    )
    return (latest + 1) / max(len(indexes), 1) if latest is not None else 0.0


def _dora_from_indicator(indicator: int) -> int:
    if indicator < 27:
        base = indicator - indicator % 9
        return base + (indicator % 9 + 1) % 9
    if indicator < 31:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3
