use serde::{Deserialize, Serialize};

use crate::{MahjongError, PlayerId, Result, RoundState, RoundWind, RuleConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchLength {
    EastOnly,
    EastSouth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoundOutcome {
    Win { winner: PlayerId },
    ExhaustiveDraw { dealer_tenpai: bool },
    AbortiveDraw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundTransition {
    pub next_round: Option<RoundState>,
    pub match_ended: bool,
    pub dealer_continues: bool,
}

/// Calculate the normal 3000-point exhaustive-draw exchange.
pub fn noten_penalty(tenpai: [bool; 4]) -> [i32; 4] {
    let count = tenpai.iter().filter(|&&value| value).count() as i32;
    if count == 0 || count == 4 {
        return [0; 4];
    }
    let gain = 3_000 / count;
    let loss = 3_000 / (4 - count);
    std::array::from_fn(|index| if tenpai[index] { gain } else { -loss })
}

/// Determine renchan, honba, dealer movement and scheduled match end.
pub fn next_round(
    current: &RoundState,
    outcome: RoundOutcome,
    scores: [i32; 4],
    rules: &RuleConfig,
) -> Result<RoundTransition> {
    current.validate()?;
    rules.validate()?;
    if rules.bankruptcy_ends_match && scores.iter().any(|&score| score < 0) {
        return Ok(RoundTransition {
            next_round: None,
            match_ended: true,
            dealer_continues: false,
        });
    }

    let dealer_continues = match outcome {
        RoundOutcome::Win { winner } => {
            winner.validate()?;
            winner == current.dealer
        }
        RoundOutcome::ExhaustiveDraw { dealer_tenpai } => dealer_tenpai,
        RoundOutcome::AbortiveDraw => true,
    };
    let final_scheduled = is_final_scheduled_round(current, rules.match_length);
    let dealer_is_top = scores[current.dealer.index()]
        == *scores
            .iter()
            .max()
            .ok_or_else(|| MahjongError::InvalidState("score array unexpectedly empty".into()))?;
    let dealer_reached_target = scores[current.dealer.index()] >= rules.target_score;
    let top_reached_target = scores.iter().copied().max().unwrap_or(i32::MIN) >= rules.target_score;
    let extension_round = is_extension_round(current, rules.match_length);
    if extension_round && top_reached_target {
        return Ok(RoundTransition {
            next_round: None,
            match_ended: true,
            dealer_continues,
        });
    }
    if is_final_extension_round(current, rules.match_length) {
        return Ok(RoundTransition {
            next_round: None,
            match_ended: true,
            dealer_continues,
        });
    }
    let end_by_yame = final_scheduled
        && dealer_continues
        && dealer_is_top
        && dealer_reached_target
        && match outcome {
            RoundOutcome::Win { .. } => rules.agari_yame,
            RoundOutcome::ExhaustiveDraw { .. } => rules.tenpai_yame,
            RoundOutcome::AbortiveDraw => false,
        };
    if end_by_yame {
        return Ok(RoundTransition {
            next_round: None,
            match_ended: true,
            dealer_continues: true,
        });
    }

    let won = matches!(outcome, RoundOutcome::Win { .. });
    if dealer_continues {
        return Ok(RoundTransition {
            next_round: Some(RoundState {
                wind: current.wind,
                hand_number: current.hand_number,
                honba: current.honba + 1,
                riichi_sticks: if won { 0 } else { current.riichi_sticks },
                dealer: current.dealer,
            }),
            match_ended: false,
            dealer_continues: true,
        });
    }

    let (wind, hand_number) = advance_hand(current.wind, current.hand_number)?;
    let crossed_scheduled_end = final_scheduled;
    if crossed_scheduled_end
        && (!rules.allow_west_extension
            || scores.iter().copied().max().unwrap_or(i32::MIN) >= rules.target_score)
    {
        return Ok(RoundTransition {
            next_round: None,
            match_ended: true,
            dealer_continues: false,
        });
    }
    Ok(RoundTransition {
        next_round: Some(RoundState {
            wind,
            hand_number,
            honba: if won { 0 } else { current.honba + 1 },
            riichi_sticks: if won { 0 } else { current.riichi_sticks },
            dealer: current.dealer.next(),
        }),
        match_ended: false,
        dealer_continues: false,
    })
}

fn is_final_scheduled_round(round: &RoundState, length: MatchLength) -> bool {
    round.hand_number == 4
        && match length {
            MatchLength::EastOnly => round.wind == RoundWind::East,
            MatchLength::EastSouth => round.wind == RoundWind::South,
        }
}

fn is_extension_round(round: &RoundState, length: MatchLength) -> bool {
    match length {
        MatchLength::EastOnly => round.wind != RoundWind::East,
        MatchLength::EastSouth => matches!(round.wind, RoundWind::West | RoundWind::North),
    }
}

fn is_final_extension_round(round: &RoundState, length: MatchLength) -> bool {
    round.hand_number == 4
        && match length {
            MatchLength::EastOnly => round.wind == RoundWind::South,
            MatchLength::EastSouth => round.wind == RoundWind::West,
        }
}

fn advance_hand(wind: RoundWind, hand_number: u8) -> Result<(RoundWind, u8)> {
    if hand_number < 4 {
        return Ok((wind, hand_number + 1));
    }
    let next_wind = match wind {
        RoundWind::East => RoundWind::South,
        RoundWind::South => RoundWind::West,
        RoundWind::West => RoundWind::North,
        RoundWind::North => {
            return Err(MahjongError::InvalidState(
                "cannot advance beyond north 4".into(),
            ));
        }
    };
    Ok((next_wind, 1))
}
