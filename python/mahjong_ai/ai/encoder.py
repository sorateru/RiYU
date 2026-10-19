"""Versioned observable-state encoder and dynamic action features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mahjong_ai.schema import ACTION_TYPES, tile_index

ENCODER_VERSION = "observable-planes-v1"
ENCODER_PLANES = 49
ACTION_FEATURES = 83


@dataclass(frozen=True)
class EncodedState:
    planes: np.ndarray
    action_features: np.ndarray
    legal_actions: tuple[dict[str, Any], ...]


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
        return EncodedState(planes=planes, action_features=action_features, legal_actions=actions)

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
