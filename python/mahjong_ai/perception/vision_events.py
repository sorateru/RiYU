"""Convert stable visual deltas into service-neutral observed event candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mahjong_ai.schema import TILE_TO_INDEX, canonical_json

from .stability import (
    StableObservation,
    observed_marker,
    observed_metadata,
    observed_tiles,
)

_MELD_REGION = re.compile(r"^meld_([0-3])_([0-3])$")


class VisionDeltaError(ValueError):
    pass


@dataclass(frozen=True)
class DetectedEvent:
    name: str
    data: dict[str, Any]
    confidence: str
    reason: str


class VisionEventDetector:
    """Infer only deltas that can be checked against the current GameState."""

    def __init__(self, observer: int) -> None:
        if not 0 <= observer <= 3:
            raise ValueError("observer must be between zero and three")
        self.observer = observer
        self._previous: StableObservation | None = None

    def prime(self, observation: StableObservation) -> None:
        self._previous = observation

    def detect(
        self, observation: StableObservation, state: dict[str, Any]
    ) -> list[DetectedEvent]:
        if self._previous is None:
            self._previous = observation
            return []
        previous = self._previous
        events: list[DetectedEvent] = []

        terminal_visible = any(
            observed_marker(observation, f"result_{kind}") is True
            for kind in ("ron", "tsumo", "draw", "abortive_draw")
        )
        if terminal_visible and (state.get("phase") or {}).get("phase") == "ended":
            self._previous = observation
            return []
        result = self._detect_result(observation, state)
        if result:
            self._previous = observation
            return result

        declared = [
            player
            for player, value in enumerate(state.get("players") or [])
            if str(value.get("riichi", "")).endswith("declared")
        ]
        if len(declared) > 1:
            raise VisionDeltaError("multiple pending riichi declarations are invalid")
        if declared:
            events.append(
                DetectedEvent(
                    "riichi",
                    {"player": declared[0], "step": "accepted"},
                    "visual_progression_inferred",
                    "a later stable frame confirms the declaration discard was not won",
                )
            )

        call = self._detect_call(previous, observation, state)
        if call is not None:
            phase = state.get("phase") or {}
            if (
                call.name == "kan"
                and call.data.get("kind") in {"ankan", "shouminkan"}
                and phase.get("phase") == "awaiting_draw"
                and phase.get("player") == call.data.get("player")
                and call.data.get("player") != self.observer
            ):
                events.append(
                    DetectedEvent(
                        "draw",
                        {
                            "player": call.data["player"],
                            "tile": None,
                            "rinshan": False,
                        },
                        "phase_inferred",
                        "opponent draw inferred before a self-kan",
                    )
                )
            events.append(call)

        dora = self._detect_dora(observation, state)
        if dora is not None:
            events.append(dora)

        river_events = self._detect_river(previous, observation, state, call)
        events.extend(river_events)
        if not river_events and call is None:
            own = self._detect_observer_hand(observation, state)
            if own is not None:
                events.extend(own)

        self._previous = observation
        return events

    def _detect_result(
        self, observation: StableObservation, state: dict[str, Any]
    ) -> list[DetectedEvent]:
        kinds = [
            kind
            for kind in ("ron", "tsumo", "draw", "abortive_draw")
            if observed_marker(observation, f"result_{kind}") is True
        ]
        if not kinds:
            return []
        if len(kinds) != 1:
            raise VisionDeltaError("result screen has multiple active result markers")
        if (state.get("phase") or {}).get("phase") == "ended":
            return []
        deltas = [observed_metadata(observation, f"delta_{player}") for player in range(4)]
        scores = state.get("scores") or []
        if len(scores) != 4:
            raise VisionDeltaError("authoritative state has no four-player scores")
        if any(not isinstance(value, int) for value in deltas):
            observed_scores = [
                observed_metadata(observation, f"score_{player}")
                for player in range(4)
            ]
            if any(not isinstance(value, int) for value in observed_scores):
                raise VisionDeltaError(
                    "result screen requires four score deltas or four resulting scores"
                )
            deltas = [
                int(value) - int(scores[player])
                for player, value in enumerate(observed_scores)
            ]
        score_deltas = [int(value) for value in deltas]
        final_scores = [int(score) + delta for score, delta in zip(scores, score_deltas)]
        kind = kinds[0]
        events: list[DetectedEvent] = []
        if kind == "ron":
            winners = [
                player
                for player in range(4)
                if observed_marker(observation, f"winner_{player}") is True
            ]
            losers = [
                player
                for player in range(4)
                if observed_marker(observation, f"loser_{player}") is True
            ]
            phase = state.get("phase") or {}
            if not losers and phase.get("phase") == "awaiting_response":
                losers = [int(phase["discarder"])]
            if not winners or len(losers) != 1 or losers[0] in winners:
                raise VisionDeltaError("ron result requires winners and exactly one loser")
            events.append(
                DetectedEvent(
                    "ron",
                    {
                        "winners": winners,
                        "loser": losers[0],
                        "score_deltas": score_deltas,
                    },
                    "result_overlay_confirmed",
                    "ron result and score deltas recognized from stable overlay",
                )
            )
            reason = "ron"
        elif kind == "tsumo":
            winners = [
                player
                for player in range(4)
                if observed_marker(observation, f"winner_{player}") is True
            ]
            if not winners:
                actor = (state.get("phase") or {}).get("player")
                if isinstance(actor, int):
                    winners = [actor]
            if len(winners) != 1:
                raise VisionDeltaError("tsumo result requires exactly one winner")
            events.append(
                DetectedEvent(
                    "tsumo",
                    {"winner": winners[0], "score_deltas": score_deltas},
                    "result_overlay_confirmed",
                    "tsumo result and score deltas recognized from stable overlay",
                )
            )
            reason = "tsumo"
        else:
            if kind == "draw":
                draw_reason = "exhaustive"
                reason = "draw"
            else:
                abortive = [
                    value
                    for value in (
                        "nine_terminals",
                        "four_winds",
                        "four_riichi",
                        "four_kan",
                        "triple_ron",
                    )
                    if observed_marker(observation, f"draw_reason_{value}") is True
                ]
                if len(abortive) != 1:
                    raise VisionDeltaError(
                        "abortive draw requires exactly one draw_reason marker"
                    )
                draw_reason = abortive[0]
                reason = "abortive_draw"
            events.append(
                DetectedEvent(
                    "draw_game",
                    {"reason": draw_reason, "score_deltas": score_deltas},
                    "result_overlay_confirmed",
                    "draw result and score deltas recognized from stable overlay",
                )
            )
        events.append(
            DetectedEvent(
                "round_end",
                {"reason": reason, "scores": final_scores},
                "result_overlay_confirmed",
                "round end follows the recognized result event",
            )
        )
        return events

    def _detect_observer_hand(
        self, observation: StableObservation, state: dict[str, Any]
    ) -> list[DetectedEvent] | None:
        phase = (state.get("phase") or {}).get("phase")
        player = (state.get("phase") or {}).get("player")
        observed = list(observed_tiles(observation, "hand")) + list(
            observed_tiles(observation, "drawn_tile")
        )
        expected = list(state.get("hand") or [])
        if state.get("drawn_tile") is not None:
            expected.append(state["drawn_tile"])
        if phase == "awaiting_draw" and player == self.observer:
            added, removed = _multiset_delta(expected, observed)
            if len(added) == 1 and not removed:
                return [
                    DetectedEvent(
                        "draw",
                        {"player": self.observer, "tile": added[0], "rinshan": False},
                        "exact_visual_delta",
                        "observer hand gained one tile",
                    )
                ]
        if phase == "awaiting_rinshan_draw" and player == self.observer:
            added, removed = _multiset_delta(expected, observed)
            if len(added) == 1 and not removed:
                return [
                    DetectedEvent(
                        "draw",
                        {"player": self.observer, "tile": added[0], "rinshan": True},
                        "exact_visual_delta",
                        "observer hand gained one rinshan tile",
                    )
                ]
        if phase == "awaiting_discard" and player == self.observer:
            added, removed = _multiset_delta(expected, observed)
            if len(removed) == 1 and not added:
                return self._discard_events(
                    self.observer,
                    removed[0],
                    state,
                    observation,
                )
        return None

    def _detect_river(
        self,
        previous: StableObservation,
        current: StableObservation,
        state: dict[str, Any],
        call: DetectedEvent | None,
    ) -> list[DetectedEvent]:
        changes: list[tuple[int, dict[str, object], int]] = []
        removed_called_discard = False
        for player in range(4):
            before = observed_tiles(previous, f"river_{player}")
            after = observed_tiles(current, f"river_{player}")
            if before == after:
                continue
            if (
                call is not None
                and not removed_called_discard
                and call.data.get("from") == player
                and before
                and after == before[:-1]
                and before[-1] == call.data.get("called_tile")
            ):
                removed_called_discard = True
                continue
            if len(after) != len(before) + 1 or tuple(after[: len(before)]) != before:
                raise VisionDeltaError(
                    f"river_{player} changed by more than one append; snapshot resync required"
                )
            changes.append((player, after[-1], len(after) - 1))
        if not changes:
            return []
        if len(changes) > 1:
            raise VisionDeltaError("multiple rivers changed in one stable observation")
        player, tile, river_index = changes[0]
        phase = state.get("phase") or {}
        phase_name = phase.get("phase")
        output: list[DetectedEvent] = []
        if call is not None and call.data.get("player") == player:
            output.extend(
                self._discard_events(player, tile, state, current, river_index)
            )
            return output
        if phase_name == "awaiting_discard" and phase.get("player") == player:
            output.extend(
                self._discard_events(player, tile, state, current, river_index)
            )
            return output
        next_player = None
        if phase_name == "awaiting_response":
            next_player = (int(phase["discarder"]) + 1) % 4
        if phase_name == "awaiting_draw":
            next_player = phase.get("player")
        if next_player == player and player != self.observer:
            output.append(
                DetectedEvent(
                    "draw",
                    {"player": player, "tile": None, "rinshan": False},
                    "phase_inferred",
                    "opponent draw inferred from legal turn progression",
                )
            )
            output.extend(
                self._discard_events(player, tile, state, current, river_index)
            )
            return output
        raise VisionDeltaError(
            f"river_{player} append is inconsistent with phase {phase_name}"
        )

    def _discard_events(
        self,
        player: int,
        tile: dict[str, object],
        state: dict[str, Any],
        observation: StableObservation,
        river_index: int | None = None,
    ) -> list[DetectedEvent]:
        if player == self.observer:
            tsumogiri = state.get("drawn_tile") == tile
            tsumogiri_known = True
            confidence = "exact_visual_delta"
        else:
            marker = (
                observed_marker(observation, f"tsumogiri_{player}_{river_index}")
                if river_index is not None
                else None
            )
            if marker is None:
                marker = observed_marker(observation, f"tsumogiri_{player}")
            tsumogiri = marker is True
            tsumogiri_known = marker is not None
            confidence = (
                "visual_tsumogiri_marker"
                if tsumogiri_known
                else "tsumogiri_unavailable"
            )
        players = state.get("players") or []
        riichi_state = players[player].get("riichi") if player < len(players) else None
        riichi_declaration = (
            observed_marker(observation, f"riichi_{player}") is True
            and riichi_state == "none"
        )
        output: list[DetectedEvent] = []
        if riichi_declaration:
            output.append(
                DetectedEvent(
                    "riichi",
                    {"player": player, "step": "declared"},
                    "visual_riichi_marker",
                    "riichi marker is active on the new discard",
                )
            )
        output.append(
            DetectedEvent(
                "discard",
                {
                    "player": player,
                    "tile": tile,
                    "tsumogiri": tsumogiri,
                    "tsumogiri_known": tsumogiri_known,
                    "riichi_declaration": riichi_declaration,
                },
                confidence,
                (
                    "new river tile with visual tsumogiri classification"
                    if tsumogiri_known
                    else "new river tile; opponent tsumogiri remains explicitly unknown"
                ),
            )
        )
        return output

    def _detect_dora(
        self, observation: StableObservation, state: dict[str, Any]
    ) -> DetectedEvent | None:
        observed = observed_tiles(observation, "dora_indicators")
        expected = tuple(state.get("dora_indicators") or [])
        if observed == expected:
            return None
        if len(observed) == len(expected) + 1 and tuple(observed[:-1]) == expected:
            return DetectedEvent(
                "dora_reveal",
                {"tile": observed[-1]},
                "exact_visual_delta",
                "dora indicator list gained one tile",
            )
        raise VisionDeltaError("dora indicators changed non-monotonically")

    def _detect_call(
        self,
        previous: StableObservation,
        current: StableObservation,
        state: dict[str, Any],
    ) -> DetectedEvent | None:
        changed: list[tuple[int, int, tuple[dict[str, object], ...], tuple[dict[str, object], ...]]] = []
        for name in set(previous.recognized) | set(current.recognized):
            match = _MELD_REGION.match(name)
            if not match:
                continue
            before = observed_tiles(previous, name)
            after = observed_tiles(current, name)
            if before != after:
                changed.append((int(match.group(1)), int(match.group(2)), before, after))
        if not changed:
            return None
        if len(changed) > 1:
            raise VisionDeltaError("multiple meld groups changed in one stable observation")
        player, group_index, before, after = changed[0]
        phase = state.get("phase") or {}
        if not before and len(after) in {3, 4}:
            kind = _meld_kind(after)
            if kind in {"chi", "pon", "kan"} and phase.get("phase") == "awaiting_response":
                event_kind = "daiminkan" if kind == "kan" else kind
                called = phase.get("tile")
                from_player = phase.get("discarder")
                if called not in after:
                    raise VisionDeltaError("new meld does not contain the latest discard")
                return DetectedEvent(
                    "kan" if event_kind == "daiminkan" else event_kind,
                    {
                        "player": player,
                        **({"kind": event_kind} if event_kind == "daiminkan" else {}),
                        "tiles": list(after),
                        "called_tile": called,
                        "from": from_player,
                    },
                    "shape_and_phase_inferred",
                    f"new {event_kind} group detected",
                )
            if (
                kind == "kan"
                and phase.get("player") == player
                and phase.get("phase") in {"awaiting_draw", "awaiting_discard"}
            ):
                return DetectedEvent(
                    "kan",
                    {
                        "player": player,
                        "kind": "ankan",
                        "tiles": list(after),
                        "called_tile": None,
                        "from": None,
                    },
                    "shape_and_phase_inferred",
                    "new concealed kan group detected",
                )
        if len(before) == 3 and len(after) == 4:
            if _meld_kind(before) != "pon" or _meld_kind(after) != "kan":
                raise VisionDeltaError("added meld tile is not a valid shouminkan")
            if (
                phase.get("player") != player
                or phase.get("phase") not in {"awaiting_draw", "awaiting_discard"}
            ):
                raise VisionDeltaError("shouminkan is inconsistent with the current phase")
            players = state.get("players") or []
            if player >= len(players) or group_index >= len(players[player].get("melds") or []):
                raise VisionDeltaError("shouminkan has no matching authoritative pon")
            existing = players[player]["melds"][group_index]
            if existing.get("kind") != "pon":
                raise VisionDeltaError("shouminkan must upgrade an authoritative pon")
            return DetectedEvent(
                "kan",
                {
                    "player": player,
                    "kind": "shouminkan",
                    "tiles": list(after),
                    "called_tile": existing.get("called_tile"),
                    "from": existing.get("from"),
                },
                "shape_and_state_inferred",
                "existing pon gained a fourth tile",
            )
        raise VisionDeltaError("meld group change cannot be mapped safely")


def _multiset_delta(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remaining_before = list(before)
    added: list[dict[str, Any]] = []
    for tile in after:
        if tile in remaining_before:
            remaining_before.remove(tile)
        else:
            added.append(tile)
    return added, remaining_before


def _meld_kind(tiles: tuple[dict[str, object], ...]) -> str:
    kinds = [str(tile["kind"]) for tile in tiles]
    if len(tiles) == 4 and len(set(kinds)) == 1:
        return "kan"
    if len(tiles) == 3 and len(set(kinds)) == 1:
        return "pon"
    if len(tiles) == 3:
        indexes = sorted(TILE_TO_INDEX[kind] for kind in kinds)
        if indexes[2] < 27 and indexes[0] // 9 == indexes[2] // 9 and indexes == list(
            range(indexes[0], indexes[0] + 3)
        ):
            return "chi"
    raise VisionDeltaError(f"invalid meld shape: {canonical_json(tiles)}")
