use std::{collections::HashMap, time::Instant};

use anyhow::Result;
use mahjong_core::{
    GameState, MeldKind, TileCounts, TileType, TurnPhase, WinMethod, calculate_shanten,
    legal_actions, score_observer_win,
};
use serde::Serialize;

use crate::shanten::SuitMemo;
use crate::{actions::Candidate, physical::Ledger, source::PhysicalTile};

#[derive(Default)]
pub struct HandAnalysis {
    shanten: HashMap<(TileCounts, u8), i8>,
    use_cache: bool,
    suit_memo: SuitMemo,
    pub evaluations: u64,
    pub hits: u64,
}

impl HandAnalysis {
    pub fn new(use_cache: bool) -> Self {
        Self {
            use_cache,
            ..Self::default()
        }
    }

    pub fn shanten(&mut self, counts: TileCounts, fixed: u8) -> mahjong_core::Result<i8> {
        if self.use_cache {
            if let Some(&result) = self.shanten.get(&(counts, fixed)) {
                self.hits += 1;
                return Ok(result);
            }
        }
        self.evaluations += 1;
        let result = if self.use_cache {
            self.suit_memo.minimum(&counts, fixed)
        } else {
            calculate_shanten(&counts, fixed)?.minimum
        };
        if self.use_cache && self.shanten.len() < 65_536 {
            self.shanten.insert((counts, fixed), result);
        }
        Ok(result)
    }

    pub fn hand(&mut self, state: &GameState) -> Result<(TileCounts, u8, i8, Option<u64>)> {
        let counts = TileCounts::from_tiles(state.hand.iter().copied().chain(state.drawn_tile))?;
        let fixed = state.players[state.observer().index()].melds.len() as u8;
        let shanten = self.shanten(counts, fixed)?;
        let waiting = if shanten == 0 && counts.total() + fixed as usize * 3 == 13 {
            Some(self.improvements(counts, fixed, shanten)?)
        } else {
            None
        };
        Ok((counts, fixed, shanten, waiting))
    }

    fn improvements(&mut self, counts: TileCounts, fixed: u8, shanten: i8) -> Result<u64> {
        let mut mask = 0;
        for tile in TileType::ALL {
            if counts[tile] < 4 {
                let mut next = counts;
                next.add(tile)?;
                if self.shanten(next, fixed)? < shanten {
                    mask |= 1 << tile.index();
                }
            }
        }
        Ok(mask)
    }
}

#[derive(Clone, Copy, Default, Serialize)]
pub struct FuritenUncertain {
    pub temporary: bool,
    pub riichi: bool,
}

impl FuritenUncertain {
    pub fn any(self) -> bool {
        self.temporary || self.riichi
    }
}

#[derive(Serialize)]
pub struct HiddenHand {
    pub seat: u8,
    pub concealed: Vec<PhysicalTile>,
    pub drawn: Option<PhysicalTile>,
    pub hand_size: u8,
    pub meld_count: u8,
    pub shanten: i8,
    pub wait_mask: Option<u64>,
    pub discard_furiten: Option<bool>,
    pub temporary_furiten: Option<bool>,
    pub riichi_furiten: Option<bool>,
}

pub fn hidden_hand(
    state: &GameState,
    ledger: &Ledger,
    uncertain: FuritenUncertain,
    cache: &mut HandAnalysis,
) -> Result<HiddenHand> {
    let seat = state.observer().get();
    let i = seat as usize;
    let (counts, meld_count, shanten, wait_mask) = cache.hand(state)?;
    let mut concealed = ledger.hand[i].clone();
    concealed.sort_by_key(|t| t.id);
    Ok(HiddenHand {
        seat,
        concealed,
        drawn: ledger.drawn[i],
        hand_size: counts.total() as u8,
        meld_count,
        shanten,
        wait_mask,
        discard_furiten: wait_mask.map(|mask| {
            state.players[i]
                .discards
                .iter()
                .any(|d| mask & (1 << d.tile.kind.index()) != 0)
        }),
        temporary_furiten: (!uncertain.temporary || state.observer_furiten.temporary)
            .then_some(state.observer_furiten.temporary),
        riichi_furiten: (!uncertain.riichi || state.observer_furiten.riichi)
            .then_some(state.observer_furiten.riichi),
    })
}

#[derive(Serialize, Clone)]
pub struct Attack {
    pub shanten: i8,
    pub wait_mask: Option<u64>,
    pub candidates: Vec<AttackCandidate>,
}

#[derive(Serialize, Clone)]
pub struct AttackCandidate {
    pub action_index: u8,
    pub shanten_after: i8,
    pub ukeire_mask: u64,
    pub theoretical_copies: u16,
    pub visible_adjusted_copies: u16,
    pub tenpai_after: bool,
}

pub fn attack(
    state: &GameState,
    candidates: &[Candidate],
    cache: &mut HandAnalysis,
) -> Result<Attack> {
    let (hand, fixed, shanten, wait_mask) = cache.hand(state)?;
    let known = TileCounts::from_tiles(
        state
            .hand
            .iter()
            .copied()
            .chain(state.drawn_tile)
            .chain(state.dora_indicators.iter().copied())
            .chain(state.players.iter().flat_map(|s| {
                s.discards
                    .iter()
                    .filter(|d| d.called_by.is_none())
                    .map(|d| d.tile)
            }))
            .chain(
                state
                    .players
                    .iter()
                    .flat_map(|s| s.melds.iter().flat_map(|m| m.tiles.iter().copied())),
            ),
    )?;
    let mut per_kind = HashMap::<TileType, AttackCandidate>::new();
    let mut output = Attack {
        shanten,
        wait_mask,
        candidates: Vec::new(),
    };
    if !candidates.iter().any(|c| c.semantic[1] <= 1) {
        return Ok(output);
    }
    for (index, c) in candidates.iter().enumerate() {
        let tile = match c.action {
            mahjong_core::Action::Discard { tile, .. } => tile,
            mahjong_core::Action::Riichi { discard, .. } => discard,
            _ => continue,
        };
        let mut entry = if let Some(entry) = per_kind.get(&tile.kind) {
            entry.clone()
        } else {
            let mut after = hand;
            after.remove(tile.kind)?;
            let shanten_after = cache.shanten(after, fixed)?;
            let ukeire_mask = cache.improvements(after, fixed, shanten_after)?;
            let copies = |counts: &TileCounts| -> u16 {
                TileType::ALL
                    .into_iter()
                    .filter(|t| ukeire_mask & (1 << t.index()) != 0)
                    .map(|t| u16::from(4 - counts[t]))
                    .sum()
            };
            let entry = AttackCandidate {
                action_index: 0,
                shanten_after,
                ukeire_mask,
                theoretical_copies: copies(&after),
                visible_adjusted_copies: copies(&known),
                tenpai_after: shanten_after == 0,
            };
            per_kind.insert(tile.kind, entry.clone());
            entry
        };
        entry.action_index = index.try_into()?;
        output.candidates.push(entry);
    }
    Ok(output)
}

#[derive(Serialize, Clone)]
pub struct RonCell {
    pub seat: u8,
    pub ron_legal: Option<bool>,
    pub legality_status: &'static str,
    pub conditional_loss: Option<u32>,
    pub no_ura_no_pao_loss: Option<u32>,
    pub loss_status: &'static str,
}

#[derive(Serialize)]
pub struct DefenseCandidate {
    pub action_index: u8,
    pub opponents: Vec<RonCell>,
}

fn pending_kan_dora(state: &GameState) -> bool {
    let completed_kans = state
        .players
        .iter()
        .flat_map(|p| &p.melds)
        .filter(|meld| {
            matches!(
                meld.kind,
                MeldKind::Ankan | MeldKind::Daiminkan | MeldKind::Shouminkan
            )
        })
        .count();
    state.dora_indicators.len() <= completed_kans
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_core::{Meld, PlayerId, RuleConfig, Tile};

    #[test]
    fn completed_kan_remains_unpriced_until_its_indicator_is_visible() {
        let actor = PlayerId::new(0).unwrap();
        let mut state = GameState::new(actor, RuleConfig::default()).unwrap();
        state.dora_indicators.push(Tile::normal(TileType::North));
        assert!(!pending_kan_dora(&state));
        state.players[0].melds.push(Meld {
            kind: MeldKind::Ankan,
            tiles: vec![Tile::normal(TileType::M1); 4],
            from: None,
            called_tile: None,
        });
        assert!(pending_kan_dora(&state));
        state.dora_indicators.push(Tile::normal(TileType::West));
        assert!(!pending_kan_dora(&state));
    }
}

pub fn defense(
    states: &[GameState; 4],
    seat: u8,
    candidates: &[Candidate],
    opponents: &[HiddenHand],
    uncertain: &[FuritenUncertain; 4],
) -> Result<Vec<DefenseCandidate>> {
    let mut output = Vec::new();
    let mut results = HashMap::new();
    let discarder = states[seat as usize].observer();
    let pending_dora = pending_kan_dora(&states[seat as usize]);
    for (index, c) in candidates.iter().enumerate() {
        let tile = match c.action {
            mahjong_core::Action::Discard { tile, .. }
            | mahjong_core::Action::Riichi { discard: tile, .. } => tile,
            _ => continue,
        };
        let cells = if let Some(cells) = results.get(&tile) {
            cells
        } else {
            let mut cells = Vec::new();
            for hand in opponents {
                let i = hand.seat as usize;
                let mut cell = RonCell {
                    seat: hand.seat,
                    ron_legal: Some(false),
                    legality_status: "valid",
                    conditional_loss: None,
                    no_ura_no_pao_loss: None,
                    loss_status: "not_applicable",
                };
                if hand
                    .wait_mask
                    .is_some_and(|mask| mask & (1 << tile.kind.index()) != 0)
                {
                    let mut state = states[i].clone();
                    state.phase = TurnPhase::AwaitingResponse { discarder, tile };
                    state.current_actor = None;
                    // The candidate does not change any opponent hand or the existing visible dora.
                    let can_ron = legal_actions(&state)?
                        .actions
                        .iter()
                        .any(|a| matches!(a, mahjong_core::Action::Ron { .. }));
                    if can_ron {
                        let score = score_observer_win(&state, WinMethod::Ron)?;
                        cell.no_ura_no_pao_loss = score.payment.ron_payment;
                        if uncertain[i].any() {
                            cell.ron_legal = None;
                            cell.legality_status = "unknown_furiten";
                        } else {
                            cell.ron_legal = Some(true);
                        }
                        let pao_possible = score.yaku.iter().any(|e| {
                            matches!(
                                e.yaku,
                                mahjong_core::Yaku::Daisangen | mahjong_core::Yaku::Daisuushii
                            )
                        });
                        if pending_dora {
                            // A delayed indicator can affect the score for
                            // this discard. Retain the score under visible dora
                            // as a separate assumption even when ron is legal.
                            cell.loss_status = "pending_kan_dora";
                        } else if state.players[i].riichi.is_accepted()
                            || pao_possible
                            || uncertain[i].any()
                        {
                            cell.loss_status = "conditional_only";
                        } else {
                            cell.loss_status = "valid_single_ron";
                            cell.conditional_loss = score.payment.ron_payment;
                        }
                    }
                }
                cells.push(cell);
            }
            results.entry(tile).or_insert(cells)
        };
        output.push(DefenseCandidate {
            action_index: index.try_into()?,
            opponents: cells.clone(),
        });
    }
    Ok(output)
}

#[derive(Default, Serialize)]
pub struct Timings {
    pub parse_us: u64,
    pub replay_us: u64,
    pub projection_us: u64,
    pub legal_us: u64,
    pub attack_us: u64,
    pub defense_us: u64,
    pub belief_us: u64,
    pub shanten_evaluations: u64,
    pub shanten_cache_hits: u64,
}

pub fn elapsed(start: Instant) -> u64 {
    start.elapsed().as_micros() as u64
}
