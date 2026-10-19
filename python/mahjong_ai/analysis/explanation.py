"""Auditable Japanese explanations without inventing unavailable estimates."""

from __future__ import annotations

from typing import Any


def explain_analysis(analysis: dict[str, Any]) -> list[dict[str, str]]:
    explanations: list[dict[str, str]] = []
    efficiency = analysis.get("discard_efficiency", {})
    for rank, candidate in enumerate(efficiency.get("value") or [], start=1):
        tile = candidate["discard"]
        explanations.append(
            {
                "kind": "exact",
                "subject": f"discard:{tile['kind']}:{'red' if tile.get('red') else 'normal'}",
                "text": (
                    f"効率順位{rank}位。打牌後シャンテン数{candidate['shanten']['minimum']}、"
                    f"公開情報を差し引いた受け入れ{candidate['ukeire_count']}枚です。"
                ),
            }
        )
    policy = analysis.get("model_policy", {})
    if policy.get("status") == "model_estimate":
        for rank, candidate in enumerate(policy.get("value") or [], start=1):
            rollout = ""
            if candidate.get("rollout_expected_points") is not None:
                rollout = (
                    f" {candidate['rollout_depth']}巡探索{candidate['rollout_samples']}標本で"
                    f"期待収支{candidate['rollout_expected_points']:.0f}点、"
                    f"和了{candidate['win_probability'] * 100:.1f}%、"
                    f"放銃{candidate['deal_in_probability'] * 100:.1f}%、"
                    f"他家和了{candidate['opponent_win_probability'] * 100:.1f}%の推定です。"
                )
            explanations.append(
                {
                    "kind": "model_estimate",
                    "subject": f"model_rank:{rank}",
                    "text": (
                        f"モデル順位{rank}位、正規化Policyスコア"
                        f"{candidate['policy_score']:.4f}。{rollout}"
                    ),
                }
            )
    else:
        explanations.append(
            {
                "kind": "unavailable",
                "subject": "model",
                "text": policy.get("reason") or "評価済みモデルがないためAI推定はN/Aです。",
            }
        )
    return explanations
