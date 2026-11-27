use anyhow::{Result, ensure};
use mahjong_core::{Action, DrawGameReason, GameState, KanAction, PlayerId, Tile, TileType};
use serde::Serialize;

pub fn tenhou_filter(state: &GameState, ledger: &Ledger, actions: &mut Vec<Action>) {
    let seat = state.observer().index();
    let kans = ledger
        .melds
        .iter()
        .flatten()
        .filter(|m| matches!(m.kind, "ankan" | "daiminkan" | "shouminkan"))
        .count();
    let after_call = state
        .event_history
        .iter()
        .rev()
        .find_map(|e| match &e.kind {
            mahjong_core::EventKind::DoraReveal { .. } => None,
            kind => Some(kind),
        })
        .and_then(|kind| match kind {
            mahjong_core::EventKind::Chi { player, meld }
            | mahjong_core::EventKind::Pon { player, meld }
                if *player == state.observer() =>
            {
                Some(meld)
            }
            _ => None,
        });
    actions.retain(|action| match action {
        Action::Discard { tile, .. } => after_call.is_none_or(|meld| {
            let Some(called) = meld.called_tile else {
                return true;
            };
            let consume = meld.consumed_from_hand().expect("validated meld");
            !kuikae(tile.kind, called.kind, &consume)
        }),
        Action::Chi {
            called_tile,
            consume,
            ..
        }
        | Action::Pon {
            called_tile,
            consume,
            ..
        } => {
            if state.remaining_tiles == 0 {
                return false;
            }
            let mut hand = state.hand.clone();
            for wanted in consume {
                if let Some(index) = hand.iter().position(|t| t == wanted) {
                    hand.swap_remove(index);
                } else {
                    return false;
                }
            }
            hand.iter()
                .any(|t| !kuikae(t.kind, called_tile.kind, consume))
        }
        Action::Kan { action } => {
            state.remaining_tiles > 0
                && kans < 4
                && match action {
                    KanAction::Ankan { tiles } if state.players[seat].riichi.is_accepted() => {
                        state.drawn_tile.is_some_and(|t| t.kind == tiles[0].kind)
                    }
                    _ => true,
                }
        }
        _ => true,
    });
}

fn kuikae(candidate: TileType, called: TileType, consumed: &[Tile]) -> bool {
    if candidate == called {
        return true;
    }
    let [a, b] = consumed else {
        return false;
    };
    if a.kind == b.kind || called.index() >= 27 {
        return false;
    }
    let a = a.kind.index().min(b.kind.index());
    let called = called.index();
    let opposite = if called + 1 == a {
        Some(a + 2)
    } else if called == a + 2 {
        a.checked_sub(1)
    } else {
        None
    };
    opposite.is_some_and(|other| other / 9 == called / 9 && candidate.index() == other)
}

use crate::{
    physical::Ledger,
    source::{PhysicalMeld, PhysicalTile, digest},
};

#[derive(Clone, Serialize)]
pub struct Candidate {
    pub id: String,
    pub semantic: Vec<u8>,
    #[serde(skip)]
    pub action: Action,
}

fn code(tile: Tile) -> u8 {
    2 * tile.kind.index() as u8 + u8::from(tile.red)
}

pub fn semantic(action: &Action, seat: u8, ledger: &Ledger) -> Vec<u8> {
    let relative = |from: PlayerId| (from.get() + 4 - seat) % 4;
    let mut values = match action {
        Action::Discard { tile, tsumogiri } => vec![0, code(*tile), u8::from(*tsumogiri)],
        Action::Riichi { discard, tsumogiri } => vec![1, code(*discard), u8::from(*tsumogiri)],
        Action::Chi {
            called_tile,
            from,
            consume,
        }
        | Action::Pon {
            called_tile,
            from,
            consume,
        } => {
            let mut c = consume.map(code);
            c.sort_unstable();
            vec![
                if matches!(action, Action::Chi { .. }) {
                    2
                } else {
                    3
                },
                code(*called_tile),
                relative(*from),
                c[0],
                c[1],
            ]
        }
        Action::Kan {
            action:
                KanAction::Daiminkan {
                    called_tile,
                    from,
                    consume,
                },
        } => {
            let mut c = consume.map(code);
            c.sort_unstable();
            vec![4, code(*called_tile), relative(*from), c[0], c[1], c[2]]
        }
        Action::Kan {
            action: KanAction::Ankan { tiles },
        } => {
            vec![
                5,
                tiles[0].kind.index() as u8,
                tiles.iter().filter(|t| t.red).count() as u8,
            ]
        }
        Action::Kan {
            action: KanAction::Shouminkan { tile, meld_index },
        } => {
            let meld = &ledger.melds[seat as usize][*meld_index];
            let mut c: Vec<_> = meld.tiles.iter().map(|t| code(t.semantic())).collect();
            c.sort_unstable();
            let mut out = vec![
                6,
                code(*tile),
                code(meld.called.expect("pon has called tile").semantic()),
                (meld.from.expect("pon has source") + 4 - seat) % 4,
            ];
            out.extend(c);
            out
        }
        Action::Ron { target } => vec![7, relative(*target)],
        Action::Tsumo => vec![8],
        Action::AbortiveDraw { reason } => vec![
            9,
            match reason {
                DrawGameReason::NineTerminals => 0,
                DrawGameReason::FourWinds => 1,
                DrawGameReason::FourRiichi => 2,
                DrawGameReason::FourKan => 3,
                DrawGameReason::TripleRon => 4,
                DrawGameReason::Exhaustive => 5,
            },
        ],
        Action::Pass => vec![10],
    };
    values.insert(0, 1); // ActionSchema version
    values
}

pub fn candidates(actions: Vec<Action>, seat: u8, ledger: &Ledger) -> Vec<Candidate> {
    let mut candidates: Vec<_> = actions
        .into_iter()
        .map(|action| {
            let semantic = semantic(&action, seat, ledger);
            Candidate {
                id: digest(&semantic)[..16].to_owned(),
                semantic,
                action,
            }
        })
        .collect();
    candidates.sort_by(|a, b| a.semantic.cmp(&b.semantic));
    candidates.dedup_by(|a, b| a.semantic == b.semantic);
    candidates
}

pub fn called_action(meld: &PhysicalMeld, seat: u8, ledger: &Ledger) -> Result<Action> {
    let semantic = meld.semantic()?;
    if meld.kind == "ankan" {
        return Ok(Action::Kan {
            action: KanAction::Ankan {
                tiles: semantic
                    .tiles
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid ankan size"))?,
            },
        });
    }
    if meld.kind == "shouminkan" {
        let added = meld
            .added
            .ok_or_else(|| anyhow::anyhow!("missing added tile"))?;
        let meld_index = ledger.melds[seat as usize]
            .iter()
            .position(|m| m.kind == "pon" && m.tiles[0].kind == added.kind)
            .ok_or_else(|| anyhow::anyhow!("missing pon for added kan"))?;
        return Ok(Action::Kan {
            action: KanAction::Shouminkan {
                tile: added.semantic(),
                meld_index,
            },
        });
    }
    let called_tile = semantic
        .called_tile
        .ok_or_else(|| anyhow::anyhow!("missing called tile"))?;
    let from = semantic
        .from
        .ok_or_else(|| anyhow::anyhow!("missing call source"))?;
    let consume = semantic.consumed_from_hand()?;
    let invalid_size = |_| anyhow::anyhow!("invalid consumed meld size");
    Ok(match meld.kind {
        "chi" => Action::Chi {
            called_tile,
            from,
            consume: consume.try_into().map_err(invalid_size)?,
        },
        "pon" => Action::Pon {
            called_tile,
            from,
            consume: consume
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid pon size"))?,
        },
        "daiminkan" => Action::Kan {
            action: KanAction::Daiminkan {
                called_tile,
                from,
                consume: consume
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid kan size"))?,
            },
        },
        _ => anyhow::bail!("unsupported call"),
    })
}

pub fn execution(action: &Action, seat: u8, ledger: &Ledger) -> Result<Vec<u8>> {
    let index = seat as usize;
    let drawn = ledger.drawn[index];
    let mut pool = ledger.hand[index].clone();
    pool.sort_by_key(|t| t.id);
    let (tiles, include_draw) = match action {
        Action::Discard { tile, tsumogiri }
        | Action::Riichi {
            discard: tile,
            tsumogiri,
        } => {
            if *tsumogiri {
                let draw = drawn
                    .filter(|t| t.semantic() == *tile)
                    .ok_or_else(|| anyhow::anyhow!("missing physical drawn tile"))?;
                return Ok(vec![draw.id]);
            }
            (vec![*tile], false)
        }
        Action::Chi { consume, .. } | Action::Pon { consume, .. } => (consume.to_vec(), false),
        Action::Kan {
            action: KanAction::Daiminkan { consume, .. },
        } => (consume.to_vec(), false),
        Action::Kan {
            action: KanAction::Ankan { tiles },
        } => (tiles.to_vec(), true),
        Action::Kan {
            action: KanAction::Shouminkan { tile, .. },
        } => (vec![*tile], true),
        _ => return Ok(Vec::new()),
    };
    if include_draw {
        pool.extend(drawn);
        pool.sort_by_key(|t| t.id);
    }
    let mut ids = Vec::new();
    for tile in tiles {
        let pos = pool
            .iter()
            .position(|t: &PhysicalTile| t.semantic() == tile);
        ensure!(pos.is_some(), "no physical execution for semantic action");
        ids.push(pool.remove(pos.unwrap()).id);
    }
    ids.sort_unstable();
    Ok(ids)
}
