use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    Action, DrawGameReason, EventKind, GameState, KanAction, MahjongError, MeldKind, PlayerId,
    Result, RiichiDeclaration, RiichiState, RoundWind, Tile, TileCounts, TileType, TurnPhase,
    WinContext, WinMethod, Yaku, calculate_shanten, is_winning_hand, score_hand,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegalActionReport {
    pub actions: Vec<Action>,
    pub discard_furiten: bool,
    pub temporary_furiten: bool,
    pub riichi_furiten: bool,
}

pub fn legal_actions(state: &GameState) -> Result<LegalActionReport> {
    state.validate()?;
    let observer = state.observer();
    let fixed = &state.players[observer.index()].melds;
    let discard_furiten = is_discard_furiten(state)?;
    let mut actions = match state.phase {
        TurnPhase::AwaitingDiscard { player } if player == observer => {
            legal_turn_actions(state, fixed)?
        }
        TurnPhase::AwaitingResponse { discarder, tile } if discarder != observer => {
            legal_response_actions(state, discarder, tile, discard_furiten)?
        }
        TurnPhase::AwaitingKanResponse {
            declarer,
            tile,
            ankan,
        } if declarer != observer => {
            legal_chankan_actions(state, declarer, tile, ankan, discard_furiten)?
        }
        _ => Vec::new(),
    };
    actions.sort_by_key(action_sort_key);
    actions.dedup();
    Ok(LegalActionReport {
        actions,
        discard_furiten,
        temporary_furiten: state.observer_furiten.temporary,
        riichi_furiten: state.observer_furiten.riichi,
    })
}

/// Score the observer's currently legal ron/tsumo using only their observation.
pub fn score_observer_win(state: &GameState, method: WinMethod) -> Result<crate::ScoreEvaluation> {
    state.validate()?;
    let observer = state.observer();
    let winning_tile = match method {
        WinMethod::Tsumo => state.drawn_tile.ok_or_else(|| {
            MahjongError::InvalidAction("tsumo scoring requires an observer draw".into())
        })?,
        WinMethod::Ron => match state.phase {
            TurnPhase::AwaitingResponse { discarder, tile } if discarder != observer => tile,
            TurnPhase::AwaitingKanResponse { declarer, tile, .. } if declarer != observer => tile,
            _ => {
                return Err(MahjongError::InvalidAction(
                    "ron scoring requires an opponent response window".into(),
                ));
            }
        },
    };
    let mut concealed = state.hand.clone();
    if method == WinMethod::Tsumo {
        concealed.extend(state.drawn_tile);
    } else {
        concealed.push(winning_tile);
    }
    score_hand(
        &concealed,
        &state.players[observer.index()].melds,
        winning_tile,
        &win_context(state, method)?,
        &state.rules,
    )
}

fn legal_chankan_actions(
    state: &GameState,
    declarer: PlayerId,
    tile: Tile,
    ankan: bool,
    discard_furiten: bool,
) -> Result<Vec<Action>> {
    let mut actions = vec![Action::Pass];
    if discard_furiten || state.observer_furiten.temporary || state.observer_furiten.riichi {
        return Ok(actions);
    }
    if ankan && !state.rules.allow_kokushi_ankan_chankan {
        return Ok(actions);
    }
    let observer = state.observer();
    let mut concealed = state.hand.clone();
    concealed.push(tile);
    if let Ok(evaluation) = score_hand(
        &concealed,
        &state.players[observer.index()].melds,
        tile,
        &win_context(state, WinMethod::Ron)?,
        &state.rules,
    ) {
        let kokushi = evaluation.yaku.iter().any(|entry| {
            matches!(
                entry.yaku,
                Yaku::KokushiMusou | Yaku::KokushiMusouThirteenWait
            )
        });
        if !ankan || kokushi {
            actions.push(Action::Ron { target: declarer });
        }
    }
    Ok(actions)
}

fn legal_turn_actions(state: &GameState, fixed: &[crate::Meld]) -> Result<Vec<Action>> {
    let observer = state.observer();
    let riichi = state.players[observer.index()].riichi.is_accepted();
    let mut actions = Vec::new();
    let discard_actions = discard_actions(state, riichi);
    actions.extend(discard_actions.iter().cloned());

    let all_tiles = observer_tiles(state);
    if let Some(drawn_tile) = state.drawn_tile {
        if score_hand(
            &all_tiles,
            fixed,
            drawn_tile,
            &win_context(state, WinMethod::Tsumo)?,
            &state.rules,
        )
        .is_ok()
        {
            actions.push(Action::Tsumo);
        }
        actions.extend(ankan_actions(state, fixed, riichi)?);
        if !riichi {
            actions.extend(shouminkan_actions(state, fixed));
        }
        if can_declare_nine_terminals(state)? {
            actions.push(Action::AbortiveDraw {
                reason: DrawGameReason::NineTerminals,
            });
        }
    }

    if !riichi && can_declare_riichi(state, fixed) {
        for action in &discard_actions {
            let Action::Discard { tile, tsumogiri } = action else {
                continue;
            };
            let mut after = all_tiles.clone();
            remove_physical_tile(&mut after, *tile)?;
            let counts = TileCounts::from_tiles(after)?;
            if calculate_shanten(&counts, fixed.len() as u8)?.minimum == 0 {
                actions.push(Action::Riichi {
                    discard: *tile,
                    tsumogiri: *tsumogiri,
                });
            }
        }
    }

    Ok(actions)
}

fn can_declare_nine_terminals(state: &GameState) -> Result<bool> {
    let observer = state.observer();
    if !state.players[observer.index()].discards.is_empty()
        || state.players.iter().any(|player| !player.melds.is_empty())
    {
        return Ok(false);
    }
    let counts = TileCounts::from_tiles(observer_tiles(state))?;
    let distinct = counts
        .iter()
        .filter(|(kind, count)| *count > 0 && kind.is_terminal_or_honor())
        .count();
    Ok(distinct >= 9)
}

fn legal_response_actions(
    state: &GameState,
    discarder: PlayerId,
    called_tile: Tile,
    discard_furiten: bool,
) -> Result<Vec<Action>> {
    let observer = state.observer();
    let mut actions = vec![Action::Pass];
    let hand = &state.hand;
    let own_riichi = state.players[observer.index()].riichi.is_accepted();

    if !own_riichi {
        if discarder.next() == observer {
            actions.extend(chi_actions(hand, called_tile, discarder));
        }
        let same_kind: Vec<_> = hand
            .iter()
            .copied()
            .filter(|tile| tile.kind == called_tile.kind)
            .collect();
        for consume in choose_distinct(&same_kind, 2) {
            actions.push(Action::Pon {
                called_tile,
                from: discarder,
                consume: [consume[0], consume[1]],
            });
        }
        for consume in choose_distinct(&same_kind, 3) {
            actions.push(Action::Kan {
                action: KanAction::Daiminkan {
                    called_tile,
                    from: discarder,
                    consume: [consume[0], consume[1], consume[2]],
                },
            });
        }
    }

    if !discard_furiten && !state.observer_furiten.temporary && !state.observer_furiten.riichi {
        let mut concealed = state.hand.clone();
        concealed.push(called_tile);
        let fixed = &state.players[observer.index()].melds;
        if score_hand(
            &concealed,
            fixed,
            called_tile,
            &win_context(state, WinMethod::Ron)?,
            &state.rules,
        )
        .is_ok()
        {
            actions.push(Action::Ron { target: discarder });
        }
    }
    Ok(actions)
}

fn discard_actions(state: &GameState, riichi: bool) -> Vec<Action> {
    if riichi {
        return state
            .drawn_tile
            .map(|tile| Action::Discard {
                tile,
                tsumogiri: true,
            })
            .into_iter()
            .collect();
    }
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for &tile in &state.hand {
        if seen.insert(tile) {
            output.push(Action::Discard {
                tile,
                tsumogiri: false,
            });
        }
    }
    if let Some(tile) = state.drawn_tile {
        output.push(Action::Discard {
            tile,
            tsumogiri: true,
        });
    }
    output
}

fn can_declare_riichi(state: &GameState, fixed: &[crate::Meld]) -> bool {
    state.scores[state.observer().index()] >= 1_000
        && state.remaining_tiles >= 4
        && fixed.iter().all(|meld| meld.kind == MeldKind::Ankan)
        && state.players[state.observer().index()].riichi == RiichiState::None
}

fn ankan_actions(state: &GameState, fixed: &[crate::Meld], riichi: bool) -> Result<Vec<Action>> {
    let all_tiles = observer_tiles(state);
    let counts = TileCounts::from_tiles(all_tiles.iter().copied())?;
    let before_waits = if riichi {
        let before = TileCounts::from_tiles(state.hand.iter().copied())?;
        Some(winning_tile_types(&before, fixed.len() as u8)?)
    } else {
        None
    };
    let mut output = Vec::new();
    for (kind, count) in counts.iter() {
        if count != 4 {
            continue;
        }
        let physical: Vec<_> = all_tiles
            .iter()
            .copied()
            .filter(|tile| tile.kind == kind)
            .collect();
        if let Some(before) = &before_waits {
            let mut after_tiles = all_tiles.clone();
            for tile in &physical {
                remove_physical_tile(&mut after_tiles, *tile)?;
            }
            let after = TileCounts::from_tiles(after_tiles)?;
            let after_waits = winning_tile_types(&after, fixed.len() as u8 + 1)?;
            if &after_waits != before {
                continue;
            }
        }
        actions_push_ankan(&mut output, &physical);
    }
    Ok(output)
}

fn actions_push_ankan(output: &mut Vec<Action>, physical: &[Tile]) {
    if physical.len() == 4 {
        output.push(Action::Kan {
            action: KanAction::Ankan {
                tiles: [physical[0], physical[1], physical[2], physical[3]],
            },
        });
    }
}

fn shouminkan_actions(state: &GameState, fixed: &[crate::Meld]) -> Vec<Action> {
    let all_tiles = observer_tiles(state);
    let mut output = Vec::new();
    for (meld_index, meld) in fixed.iter().enumerate() {
        if meld.kind != MeldKind::Pon {
            continue;
        }
        for tile in all_tiles
            .iter()
            .copied()
            .filter(|tile| tile.kind == meld.tiles[0].kind)
        {
            output.push(Action::Kan {
                action: KanAction::Shouminkan { tile, meld_index },
            });
        }
    }
    output
}

fn chi_actions(hand: &[Tile], called: Tile, from: PlayerId) -> Vec<Action> {
    let Some(rank) = called.kind.rank() else {
        return Vec::new();
    };
    let suit_base = called.kind.index() / 9 * 9;
    let mut output = Vec::new();
    for start_rank in rank.saturating_sub(2)..=rank {
        if !(1..=7).contains(&start_rank) || !(start_rank..=start_rank + 2).contains(&rank) {
            continue;
        }
        let kinds = [
            TileType::ALL[suit_base + start_rank as usize - 1],
            TileType::ALL[suit_base + start_rank as usize],
            TileType::ALL[suit_base + start_rank as usize + 1],
        ];
        let needed: Vec<_> = kinds
            .into_iter()
            .filter(|kind| *kind != called.kind)
            .collect();
        if needed.len() != 2 {
            continue;
        }
        let first: Vec<_> = hand
            .iter()
            .copied()
            .filter(|tile| tile.kind == needed[0])
            .collect();
        let second: Vec<_> = hand
            .iter()
            .copied()
            .filter(|tile| tile.kind == needed[1])
            .collect();
        for &a in &first {
            for &b in &second {
                output.push(Action::Chi {
                    called_tile: called,
                    from,
                    consume: [a, b],
                });
            }
        }
    }
    output
}

fn choose_distinct(tiles: &[Tile], amount: usize) -> Vec<Vec<Tile>> {
    fn visit(
        tiles: &[Tile],
        amount: usize,
        start: usize,
        current: &mut Vec<Tile>,
        output: &mut Vec<Vec<Tile>>,
    ) {
        if current.len() == amount {
            if !output.contains(current) {
                output.push(current.clone());
            }
            return;
        }
        for index in start..tiles.len() {
            current.push(tiles[index]);
            visit(tiles, amount, index + 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    visit(tiles, amount, 0, &mut Vec::new(), &mut output);
    output
}

pub fn winning_tile_types(counts: &TileCounts, fixed_melds: u8) -> Result<Vec<TileType>> {
    let mut output = Vec::new();
    for tile in TileType::ALL {
        if counts[tile] >= 4 {
            continue;
        }
        let mut candidate = *counts;
        candidate.add(tile)?;
        if is_winning_hand(&candidate, fixed_melds)? {
            output.push(tile);
        }
    }
    Ok(output)
}

fn is_discard_furiten(state: &GameState) -> Result<bool> {
    let observer = state.observer();
    let fixed_melds = state.players[observer.index()].melds.len() as u8;
    let expected = 13usize.saturating_sub(fixed_melds as usize * 3);
    if state.hand.len() != expected {
        return Ok(false);
    }
    let counts = TileCounts::from_tiles(state.hand.iter().copied())?;
    let waits = winning_tile_types(&counts, fixed_melds)?;
    Ok(state.players[observer.index()]
        .discards
        .iter()
        .any(|discard| waits.contains(&discard.tile.kind)))
}

fn observer_tiles(state: &GameState) -> Vec<Tile> {
    let mut tiles = state.hand.clone();
    tiles.extend(state.drawn_tile);
    tiles
}

fn remove_physical_tile(tiles: &mut Vec<Tile>, target: Tile) -> Result<()> {
    let position = tiles
        .iter()
        .position(|tile| *tile == target)
        .ok_or_else(|| MahjongError::InvalidState(format!("missing physical tile {target:?}")))?;
    tiles.remove(position);
    Ok(())
}

fn win_context(state: &GameState, method: WinMethod) -> Result<WinContext> {
    let round = state
        .round
        .as_ref()
        .ok_or_else(|| MahjongError::InvalidState("active round metadata is missing".into()))?;
    let observer = state.observer();
    let seat_offset = (observer.get() + 4 - round.dealer.get()) % 4;
    let seat_wind = [
        TileType::East,
        TileType::South,
        TileType::West,
        TileType::North,
    ][seat_offset as usize];
    let round_wind = match round.wind {
        RoundWind::East => TileType::East,
        RoundWind::South => TileType::South,
        RoundWind::West => TileType::West,
        RoundWind::North => TileType::North,
    };
    let own_riichi = state.players[observer.index()].riichi.is_accepted();
    let no_calls = state.players.iter().all(|player| player.melds.is_empty());
    let first_own_turn = state.players[observer.index()].discards.is_empty();
    let rinshan = method == WinMethod::Tsumo
        && matches!(
            state.last_event.as_ref().map(|event| &event.kind),
            Some(EventKind::Draw { player, rinshan: true, .. }) if *player == observer
        );
    Ok(WinContext {
        method,
        seat_wind,
        round_wind,
        riichi: if state.players[observer.index()].riichi.is_double() {
            RiichiDeclaration::DoubleRiichi
        } else if own_riichi {
            RiichiDeclaration::Riichi
        } else {
            RiichiDeclaration::None
        },
        ippatsu: state.ippatsu_eligible[observer.index()],
        rinshan,
        chankan: method == WinMethod::Ron
            && matches!(state.phase, TurnPhase::AwaitingKanResponse { .. }),
        haitei: method == WinMethod::Tsumo && state.remaining_tiles == 0,
        houtei: method == WinMethod::Ron && state.remaining_tiles == 0,
        tenhou: method == WinMethod::Tsumo
            && seat_wind == TileType::East
            && first_own_turn
            && no_calls
            && state.turn == 1,
        chiihou: method == WinMethod::Tsumo
            && seat_wind != TileType::East
            && first_own_turn
            && no_calls,
        dora_indicators: state.dora_indicators.iter().map(|tile| tile.kind).collect(),
        ura_dora_indicators: Vec::new(),
        honba: round.honba,
        riichi_sticks: round.riichi_sticks,
    })
}

fn action_sort_key(action: &Action) -> (u8, usize, bool) {
    match action {
        Action::Tsumo => (0, 0, false),
        Action::Ron { .. } => (1, 0, false),
        Action::Riichi { discard, tsumogiri } => (2, discard.kind.index(), *tsumogiri),
        Action::Kan { .. } => (3, 0, false),
        Action::Pon { called_tile, .. } => (4, called_tile.kind.index(), false),
        Action::Chi { called_tile, .. } => (5, called_tile.kind.index(), false),
        Action::Discard { tile, tsumogiri } => (6, tile.kind.index(), *tsumogiri),
        Action::AbortiveDraw { .. } => (7, 0, false),
        Action::Pass => (8, 0, false),
    }
}
