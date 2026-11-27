use anyhow::{Result, ensure};
use mahjong_core::{
    Action, EventKind, GameEvent, GameState, KanAction, Meld, MeldKind, RiichiStep, ShantenResult,
    Tile, TileCounts, TileType, TurnPhase, WinMethod, legal_actions, legal_actions_with_shanten,
    score_observer_win,
};
use serde::Serialize;

use crate::{
    actions::{Candidate, candidates, tenhou_filter},
    physical::Ledger,
    source::digest,
    teachers::{
        Attack, AttackCandidate, FuritenUncertain, HandAnalysis, HiddenHand, RonCell, attack,
        defense, known_tiles, pending_kan_dora, ron_cell,
    },
};

#[derive(Serialize)]
pub struct HandProfile {
    #[serde(flatten)]
    families: ShantenResult,
    wait_mask: Option<u64>,
    ukeire_mask: Option<u64>,
    theoretical_copies: Option<u16>,
    visible_adjusted_copies: Option<u16>,
    closed: bool,
    meld_count: u8,
    concealed: Vec<u8>,
    drawn: Option<u8>,
    red_count: u8,
    visible_dora_count: u8,
}

#[derive(Serialize)]
pub struct Successor {
    #[serde(flatten)]
    action: Candidate,
    after_discard: HandProfile,
    genbutsu_mask: u8,
    ron: Vec<RonCell>,
}

#[derive(Serialize)]
pub struct LocalAction {
    action_index: u8,
    transition_status: &'static str,
    after: Option<HandProfile>,
    successors: Vec<Successor>,
    successor_set_hash: Option<String>,
    dama_index: Option<u8>,
    scores_before: Option<[i32; 4]>,
    sticks_before: Option<u16>,
    deposit_on_acceptance: Option<u16>,
    kan_dora_timing: Option<&'static str>,
    rinshan_if_continues: Option<bool>,
    live_wall_after_rinshan_if_drawn: Option<u16>,
    four_kan_abort_condition: Option<bool>,
    ippatsu_cancelled_on_completion: Option<bool>,
    kan_robbery: Vec<RonCell>,
    visible_win_score: Option<mahjong_core::ScoreEvaluation>,
    win_score_status: Option<&'static str>,
}

#[derive(Serialize)]
pub struct LocalLabels {
    before: HandProfile,
    candidates: Vec<LocalAction>,
}

fn tile_code(tile: Tile) -> u8 {
    2 * tile.kind.index() as u8 + u8::from(tile.red)
}

fn profile(
    state: &GameState,
    discard: Option<Tile>,
    after_kan_completion: Option<&Meld>,
    attack_label: Option<&AttackCandidate>,
    cache: &mut HandAnalysis,
) -> Result<HandProfile> {
    let mut concealed = state.hand.clone();
    let mut drawn = state.drawn_tile;
    if let Some(tile) = discard {
        concealed.extend(drawn.take());
        let index = concealed
            .iter()
            .position(|&t| t == tile)
            .expect("legal discard");
        concealed.swap_remove(index);
    }
    concealed.sort_unstable_by_key(|&tile| tile_code(tile));
    let counts = TileCounts::from_tiles(concealed.iter().copied().chain(drawn))?;
    let mut melds: Vec<_> = state.players[state.observer().index()]
        .melds
        .iter()
        .collect();
    if let Some(kan) = after_kan_completion {
        if let Some(index) = state
            .pending_kan
            .as_ref()
            .and_then(|p| p.replace_meld_index)
        {
            melds[index] = kan;
        } else {
            melds.push(kan);
        }
    }
    let fixed = u8::try_from(melds.len())?;
    let families = cache.shanten_families(counts, fixed)?;
    let ukeire = if concealed.len() + usize::from(drawn.is_some()) + usize::from(fixed) * 3 != 13 {
        None
    } else if let Some(label) = attack_label {
        Some((
            label.ukeire_mask,
            label.theoretical_copies,
            label.visible_adjusted_copies,
        ))
    } else {
        let mask = cache.improvements(counts, fixed, families.minimum)?;
        let known = known_tiles(state)?;
        let copies = |pool: &TileCounts| {
            TileType::ALL
                .into_iter()
                .filter(|t| mask & (1 << t.index()) != 0)
                .map(|t| u16::from(4 - pool[t]))
                .sum()
        };
        Some((mask, copies(&counts), copies(&known)))
    };
    let owned: Vec<_> = concealed
        .iter()
        .copied()
        .chain(drawn)
        .chain(melds.iter().flat_map(|m| m.tiles.iter().copied()))
        .collect();
    Ok(HandProfile {
        families,
        wait_mask: ukeire.filter(|_| families.minimum == 0).map(|x| x.0),
        ukeire_mask: ukeire.map(|x| x.0),
        theoretical_copies: ukeire.map(|x| x.1),
        visible_adjusted_copies: ukeire.map(|x| x.2),
        closed: melds.iter().all(|m| m.kind == MeldKind::Ankan),
        meld_count: fixed,
        concealed: concealed.into_iter().map(tile_code).collect(),
        drawn: drawn.map(tile_code),
        red_count: owned.iter().filter(|tile| tile.red).count().try_into()?,
        visible_dora_count: state
            .dora_indicators
            .iter()
            .map(|indicator| {
                owned
                    .iter()
                    .filter(|tile| tile.kind == indicator.kind.dora())
                    .count()
            })
            .sum::<usize>()
            .try_into()?,
    })
}

fn apply(state: &mut GameState, kind: EventKind) -> Result<()> {
    state.apply_event(GameEvent {
        sequence: state.next_event_sequence,
        kind,
    })?;
    Ok(())
}

fn open_meld(
    kind: MeldKind,
    called: Tile,
    from: mahjong_core::PlayerId,
    consumed: &[Tile],
) -> Meld {
    Meld {
        kind,
        tiles: std::iter::once(called)
            .chain(consumed.iter().copied())
            .collect(),
        called_tile: Some(called),
        from: Some(from),
    }
}
#[allow(clippy::too_many_arguments)]
pub fn labels(
    states: &[GameState; 4],
    seat: u8,
    rows: &[Candidate],
    root_attack: &Attack,
    opponents: &[HiddenHand],
    uncertain: &[FuritenUncertain; 4],
    ledger: &Ledger,
    cache: &mut HandAnalysis,
) -> Result<Option<LocalLabels>> {
    let state = &states[seat as usize];
    let player = state.observer();
    let mut output = Vec::new();
    for (index, candidate) in rows.iter().enumerate() {
        if matches!(candidate.action, Action::Discard { .. } | Action::Pass) {
            continue;
        }
        let mut label = LocalAction {
            action_index: index.try_into()?,
            transition_status: "declared_abort",
            after: None,
            successors: Vec::new(),
            successor_set_hash: None,
            dama_index: None,
            scores_before: None,
            sticks_before: None,
            deposit_on_acceptance: None,
            kan_dora_timing: None,
            rinshan_if_continues: None,
            live_wall_after_rinshan_if_drawn: None,
            four_kan_abort_condition: None,
            ippatsu_cancelled_on_completion: None,
            kan_robbery: Vec::new(),
            visible_win_score: None,
            win_score_status: None,
        };
        let meld = match candidate.action {
            Action::Chi {
                called_tile,
                from,
                consume,
            } => Some(open_meld(MeldKind::Chi, called_tile, from, &consume)),
            Action::Pon {
                called_tile,
                from,
                consume,
            } => Some(open_meld(MeldKind::Pon, called_tile, from, &consume)),
            Action::Kan {
                action:
                    KanAction::Daiminkan {
                        called_tile,
                        from,
                        consume,
                    },
            } => Some(open_meld(MeldKind::Daiminkan, called_tile, from, &consume)),
            Action::Kan {
                action: KanAction::Ankan { tiles },
            } => Some(Meld {
                kind: MeldKind::Ankan,
                tiles: tiles.to_vec(),
                called_tile: None,
                from: None,
            }),
            Action::Kan {
                action: KanAction::Shouminkan { tile, meld_index },
            } => {
                let mut meld = state.players[seat as usize].melds[meld_index].clone();
                meld.kind = MeldKind::Shouminkan;
                meld.tiles.push(tile);
                Some(meld)
            }
            Action::Riichi { discard, tsumogiri } => {
                label.transition_status = "declaration_discard_before_acceptance";
                label.dama_index = rows
                    .iter()
                    .position(|row| {
                        row.action
                            == Action::Discard {
                                tile: discard,
                                tsumogiri,
                            }
                    })
                    .map(u8::try_from)
                    .transpose()?;
                ensure!(
                    label.dama_index.is_some(),
                    "riichi lacks matching dama discard"
                );
                label.scores_before = Some(state.scores);
                label.sticks_before = state.round.as_ref().map(|round| round.riichi_sticks);
                label.deposit_on_acceptance = Some(1000);
                label.after = Some(profile(
                    state,
                    Some(discard),
                    None,
                    root_attack
                        .candidates
                        .iter()
                        .find(|a| a.action_index == label.action_index),
                    cache,
                )?);
                None
            }
            Action::Ron { .. } | Action::Tsumo => {
                label.transition_status = "win_if_resolved";
                let method = if candidate.action == Action::Tsumo {
                    WinMethod::Tsumo
                } else {
                    WinMethod::Ron
                };
                let score = score_observer_win(state, method)?;
                let pao_possible = score.yaku.iter().any(|y| {
                    matches!(
                        y.yaku,
                        mahjong_core::Yaku::Daisangen | mahjong_core::Yaku::Daisuushii
                    )
                });
                label.win_score_status = Some(if pending_kan_dora(state) {
                    "pending_kan_dora"
                } else if state.players[seat as usize].riichi.is_accepted()
                    || pao_possible
                    || (method == WinMethod::Ron && uncertain[seat as usize].any())
                {
                    "conditional_only"
                } else {
                    "valid_for_single_winner"
                });
                label.visible_win_score = Some(score);
                None
            }
            _ => None,
        };
        if let Some(meld) = meld {
            let open_call = matches!(
                meld.kind,
                MeldKind::Chi | MeldKind::Pon | MeldKind::Daiminkan
            );
            let event = match meld.kind {
                MeldKind::Chi => EventKind::Chi {
                    player,
                    meld: meld.clone(),
                },
                MeldKind::Pon => EventKind::Pon {
                    player,
                    meld: meld.clone(),
                },
                _ => EventKind::Kan {
                    player,
                    meld: meld.clone(),
                },
            };
            let mut branch = states.clone();
            for view in &mut branch {
                if open_call {
                    let from = meld.from.expect("open call has source");
                    if view.players[from.index()].riichi.is_declared()
                        && !view.players[from.index()].riichi.is_accepted()
                    {
                        apply(
                            view,
                            EventKind::Riichi {
                                player: from,
                                step: RiichiStep::Accepted,
                            },
                        )?;
                    }
                }
                apply(view, event.clone())?;
            }
            let next = &branch[seat as usize];
            if matches!(meld.kind, MeldKind::Chi | MeldKind::Pon) {
                label.transition_status = "if_call_wins_priority";
                label.after = Some(profile(next, None, None, None, cache)?);
                let mut legal = legal_actions_with_shanten(next, |counts, fixed| {
                    cache.shanten(*counts, fixed)
                })?
                .actions;
                tenhou_filter(next, ledger, &mut legal);
                let successors = candidates(legal, seat, ledger);
                ensure!(!successors.is_empty(), "call branch without legal discard");
                label.successor_set_hash = Some(digest(&serde_json::to_vec(&successors)?));
                let attack = attack(next, &successors, cache)?;
                let mut branch_uncertainty = *uncertain;
                for opponent in opponents {
                    let i = opponent.seat as usize;
                    let can_ron_previous = match state.phase {
                        TurnPhase::AwaitingResponse { discarder, tile } => {
                            discarder.get() != opponent.seat
                                && opponent
                                    .wait_mask
                                    .is_some_and(|mask| mask & (1 << tile.kind.index()) != 0)
                        }
                        _ => false,
                    };
                    if can_ron_previous
                        && legal_actions(&states[i])?
                            .actions
                            .iter()
                            .any(|action| matches!(action, Action::Ron { .. }))
                    {
                        // A successful call still does not record an individual client's response.
                        if branch[i].players[i].riichi.is_accepted() {
                            branch_uncertainty[i].riichi = true;
                        } else {
                            branch_uncertainty[i].temporary = true;
                        }
                    }
                }
                let defense = defense(&branch, seat, &successors, opponents, &branch_uncertainty)?;
                for ((action, attack), defense) in
                    successors.into_iter().zip(&attack.candidates).zip(defense)
                {
                    let Action::Discard { tile, .. } = action.action else {
                        anyhow::bail!("non-discard in call successor axis");
                    };
                    let genbutsu_mask = next
                        .players
                        .iter()
                        .enumerate()
                        .filter(|(i, opponent)| {
                            *i != seat as usize
                                && opponent.discards.iter().any(|d| d.tile.kind == tile.kind)
                        })
                        .fold(0, |mask, (i, _)| mask | (1 << i));
                    label.successors.push(Successor {
                        action,
                        after_discard: profile(next, Some(tile), None, Some(attack), cache)?,
                        genbutsu_mask,
                        ron: defense.opponents,
                    });
                }
            } else {
                let pending = next.pending_kan.is_some();
                label.transition_status = if pending {
                    "if_kan_not_robbed"
                } else {
                    "if_call_wins_priority"
                };
                label.after = Some(profile(next, None, pending.then_some(&meld), None, cache)?);
                label.kan_dora_timing = Some(if meld.kind == MeldKind::Ankan {
                    "before_rinshan"
                } else {
                    "delayed"
                });
                label.rinshan_if_continues = Some(true);
                label.live_wall_after_rinshan_if_drawn = next.remaining_tiles.checked_sub(1);
                let mut kans = 1;
                let mut other_owner = false;
                for (i, opponent) in state.players.iter().enumerate() {
                    for existing in &opponent.melds {
                        if matches!(
                            existing.kind,
                            MeldKind::Ankan | MeldKind::Daiminkan | MeldKind::Shouminkan
                        ) {
                            kans += 1;
                            other_owner |= i != seat as usize;
                        }
                    }
                }
                label.four_kan_abort_condition = Some(kans == 4 && other_owner);
                label.ippatsu_cancelled_on_completion = Some(true);
                if let TurnPhase::AwaitingKanResponse { tile, .. } = next.phase {
                    for hand in opponents {
                        let i = hand.seat as usize;
                        let waits = hand
                            .wait_mask
                            .is_some_and(|mask| mask & (1 << tile.kind.index()) != 0);
                        label.kan_robbery.push(ron_cell(
                            hand.seat,
                            &branch[i],
                            uncertain[i],
                            pending_kan_dora(state),
                            waits,
                        )?);
                    }
                }
            }
        }
        output.push(label);
    }
    if output.is_empty() {
        return Ok(None);
    }
    Ok(Some(LocalLabels {
        before: profile(state, None, None, None, cache)?,
        candidates: output,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        source::{PhysicalTile, Start},
        teachers::hidden_hand,
    };
    use mahjong_core::{PlayerId, RoundState, RoundWind, RuleConfig};

    #[test]
    fn chi_branches_keep_red_choices_and_remove_both_kuikae_flanks() {
        let tile = |id| PhysicalTile {
            id,
            kind: id / 4,
            red: matches!(id, 16 | 52 | 88),
        };
        let own = [8, 12, 16, 17, 20, 24, 28, 36, 37, 44, 80, 108, 116];
        let discarded = tile(13);
        let indicator = tile(124);
        let mut unused = (0..136)
            .filter(|id| !own.contains(id) && *id != discarded.id && *id != indicator.id)
            .map(tile);
        let start = Start {
            hand_number: 0,
            honba: 0,
            sticks: 0,
            dealer: 0,
            dice: [1, 1],
            scores: [25_000; 4],
            hands: std::array::from_fn(|seat| {
                if seat == 1 {
                    own.into_iter().map(tile).collect()
                } else {
                    unused.by_ref().take(13).collect()
                }
            }),
            dora: indicator,
        };
        let source = PlayerId::new(0).unwrap();
        let mut ledger = Ledger::new(&start).unwrap();
        ledger.draw(0, discarded).unwrap();
        ledger.discard(0, discarded).unwrap();
        let states = std::array::from_fn(|seat| {
            let mut view =
                GameState::new(PlayerId::new(seat as u8).unwrap(), RuleConfig::default()).unwrap();
            apply(
                &mut view,
                EventKind::RoundStart {
                    round: RoundState {
                        wind: RoundWind::East,
                        hand_number: 1,
                        honba: 0,
                        riichi_sticks: 0,
                        dealer: source,
                    },
                    scores: start.scores,
                    initial_hand: start.hands[seat].iter().map(|t| t.semantic()).collect(),
                    dora_indicator: indicator.semantic(),
                    remaining_tiles: 70,
                },
            )
            .unwrap();
            apply(
                &mut view,
                EventKind::Draw {
                    player: source,
                    tile: (seat == 0).then(|| discarded.semantic()),
                    rinshan: false,
                },
            )
            .unwrap();
            apply(
                &mut view,
                EventKind::Discard {
                    player: source,
                    tile: discarded.semantic(),
                    tsumogiri: true,
                    tsumogiri_known: true,
                    riichi_declaration: false,
                },
            )
            .unwrap();
            view
        });
        let mut actions = legal_actions(&states[1]).unwrap().actions;
        tenhou_filter(&states[1], &ledger, &mut actions);
        let rows = candidates(actions, 1, &ledger);
        let mut cache = HandAnalysis::new(true);
        let attack = attack(&states[1], &rows, &mut cache).unwrap();
        let uncertain = [FuritenUncertain::default(); 4];
        let others: Vec<_> = [0, 2, 3]
            .into_iter()
            .map(|i| hidden_hand(&states[i], &ledger, uncertain[i], &mut cache).unwrap())
            .collect();
        let local = labels(
            &states, 1, &rows, &attack, &others, &uncertain, &ledger, &mut cache,
        )
        .unwrap()
        .unwrap();
        let mut variants = Vec::new();
        for parent in local.candidates {
            let Action::Chi { consume, .. } = rows[parent.action_index as usize].action else {
                continue;
            };
            assert_eq!(parent.transition_status, "if_call_wins_priority");
            assert!(
                parent
                    .successors
                    .iter()
                    .all(|c| c.action.semantic[2] / 2 != 3)
            );
            let after = parent.after.unwrap();
            assert_eq!(after.meld_count, 1);
            assert!(!after.closed);
            assert_eq!(after.families.seven_pairs, None);
            assert_eq!(after.families.thirteen_orphans, None);
            assert_eq!(after.ukeire_mask, None);
            if consume.iter().any(|t| t.kind == TileType::M6) {
                assert!(
                    parent
                        .successors
                        .iter()
                        .all(|c| c.action.semantic[2] / 2 != 6)
                );
                variants.push(consume.iter().any(|t| t.red));
            }
        }
        variants.sort_unstable();
        assert_eq!(variants, vec![false, true]);
    }
}
