use mahjong_core::{
    Action, DiscardRecord, EventKind, GameEvent, PlayerId, RiichiState, RoundState, RoundWind,
    RuleConfig, Tile, TileType, WinMethod, Yaku, legal_actions, replay_events, score_observer_win,
};

fn player(id: u8) -> PlayerId {
    PlayerId::new(id).unwrap()
}

fn tile(kind: TileType) -> Tile {
    Tile::normal(kind)
}

fn tiles(entries: &[(TileType, u8)]) -> Vec<Tile> {
    entries
        .iter()
        .flat_map(|&(kind, count)| (0..count).map(move |_| tile(kind)))
        .collect()
}

fn round_start(observer_hand: Vec<Tile>, dealer: u8) -> GameEvent {
    GameEvent {
        sequence: 0,
        kind: EventKind::RoundStart {
            round: RoundState {
                wind: RoundWind::East,
                hand_number: 1,
                honba: 0,
                riichi_sticks: 0,
                dealer: player(dealer),
            },
            scores: [25_000; 4],
            initial_hand: observer_hand,
            dora_indicator: tile(TileType::North),
            remaining_tiles: 70,
        },
    }
}

fn pinfu_tenpai() -> Vec<Tile> {
    use TileType::*;
    tiles(&[
        (M1, 1),
        (M2, 1),
        (M3, 1),
        (M4, 1),
        (M5, 1),
        (M6, 1),
        (P7, 1),
        (P8, 1),
        (P9, 1),
        (S2, 1),
        (S3, 1),
        (P5, 2),
    ])
}

#[test]
fn turn_actions_include_riichi_discard_and_all_physical_discards() {
    let state = replay_events(
        player(0),
        RuleConfig::default(),
        [
            round_start(pinfu_tenpai(), 0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: Some(tile(TileType::East)),
                    rinshan: false,
                },
            },
        ],
    )
    .unwrap();
    let report = legal_actions(&state).unwrap();

    assert!(report.actions.iter().any(|action| matches!(
        action,
        Action::Discard { tile: discarded, tsumogiri: true } if discarded.kind == TileType::East
    )));
    assert!(report.actions.iter().any(|action| matches!(
        action,
        Action::Riichi { discard, tsumogiri: true } if discard.kind == TileType::East
    )));
}

#[test]
fn response_actions_include_pinfu_ron_and_pass() {
    let state = replay_events(
        player(1),
        RuleConfig::default(),
        [
            round_start(pinfu_tenpai(), 0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: None,
                    rinshan: false,
                },
            },
            GameEvent {
                sequence: 2,
                kind: EventKind::Discard {
                    player: player(0),
                    tile: tile(TileType::S4),
                    tsumogiri: false,
                    tsumogiri_known: true,
                    riichi_declaration: false,
                },
            },
        ],
    )
    .unwrap();
    let report = legal_actions(&state).unwrap();

    assert!(report.actions.contains(&Action::Pass));
    assert!(report.actions.contains(&Action::Ron { target: player(0) }));
}

#[test]
fn own_discard_of_any_wait_causes_discard_furiten() {
    let mut state = replay_events(
        player(1),
        RuleConfig::default(),
        [
            round_start(pinfu_tenpai(), 0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: None,
                    rinshan: false,
                },
            },
            GameEvent {
                sequence: 2,
                kind: EventKind::Discard {
                    player: player(0),
                    tile: tile(TileType::S4),
                    tsumogiri: false,
                    tsumogiri_known: true,
                    riichi_declaration: false,
                },
            },
        ],
    )
    .unwrap();
    state.players[1].discards.push(DiscardRecord {
        tile: tile(TileType::S1),
        tsumogiri: false,
        tsumogiri_known: true,
        riichi_declaration: false,
        called_by: None,
    });

    let report = legal_actions(&state).unwrap();
    assert!(report.discard_furiten);
    assert!(!report.actions.contains(&Action::Ron { target: player(0) }));
}

#[test]
fn passing_ron_sets_temporary_furiten_and_next_draw_clears_it() {
    let mut state = replay_events(
        player(1),
        RuleConfig::default(),
        [
            round_start(pinfu_tenpai(), 0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: None,
                    rinshan: false,
                },
            },
            GameEvent {
                sequence: 2,
                kind: EventKind::Discard {
                    player: player(0),
                    tile: tile(TileType::S4),
                    tsumogiri: false,
                    tsumogiri_known: true,
                    riichi_declaration: false,
                },
            },
        ],
    )
    .unwrap();
    state
        .apply_event(GameEvent {
            sequence: 3,
            kind: EventKind::Pass {
                player: player(1),
                ron_available: true,
            },
        })
        .unwrap();
    assert!(state.observer_furiten.temporary);

    state
        .apply_event(GameEvent {
            sequence: 4,
            kind: EventKind::Draw {
                player: player(1),
                tile: Some(tile(TileType::East)),
                rinshan: false,
            },
        })
        .unwrap();
    assert!(!state.observer_furiten.temporary);
}

#[test]
fn riichi_pass_persists_after_the_observers_draw() {
    let mut state = replay_events(
        player(1),
        RuleConfig::default(),
        [
            round_start(pinfu_tenpai(), 0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: None,
                    rinshan: false,
                },
            },
            GameEvent {
                sequence: 2,
                kind: EventKind::Discard {
                    player: player(0),
                    tile: tile(TileType::S4),
                    tsumogiri: false,
                    tsumogiri_known: true,
                    riichi_declaration: false,
                },
            },
        ],
    )
    .unwrap();
    state.players[1].riichi = RiichiState::Accepted;
    state
        .apply_event(GameEvent {
            sequence: 3,
            kind: EventKind::Pass {
                player: player(1),
                ron_available: true,
            },
        })
        .unwrap();
    assert!(state.observer_furiten.riichi);
    assert!(
        !legal_actions(&state)
            .unwrap()
            .actions
            .contains(&Action::Ron { target: player(0) })
    );
    state
        .apply_event(GameEvent {
            sequence: 4,
            kind: EventKind::Draw {
                player: player(1),
                tile: Some(tile(TileType::East)),
                rinshan: false,
            },
        })
        .unwrap();
    assert!(state.observer_furiten.riichi);
}

#[test]
fn rinshan_on_the_last_available_tile_does_not_add_haitei() {
    let mut start = round_start(pinfu_tenpai(), 0);
    if let EventKind::RoundStart {
        remaining_tiles, ..
    } = &mut start.kind
    {
        *remaining_tiles = 1;
    }
    let mut state = replay_events(
        player(0),
        RuleConfig::default(),
        [
            start,
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: Some(tile(TileType::S4)),
                    rinshan: false,
                },
            },
        ],
    )
    .unwrap();
    state.turn = 8;
    let normal = score_observer_win(&state, WinMethod::Tsumo).unwrap();
    assert!(normal.yaku.iter().any(|y| y.yaku == Yaku::Haitei));
    let last = state.event_history.last_mut().unwrap();
    if let EventKind::Draw { rinshan, .. } = &mut last.kind {
        *rinshan = true;
    }
    state.last_event = Some(last.clone());
    let replacement = score_observer_win(&state, WinMethod::Tsumo).unwrap();
    assert!(
        replacement
            .yaku
            .iter()
            .any(|y| y.yaku == Yaku::RinshanKaihou)
    );
    assert!(!replacement.yaku.iter().any(|y| y.yaku == Yaku::Haitei));
}

#[test]
fn accepted_riichi_restricts_discard_to_tsumogiri() {
    let mut state = replay_events(
        player(0),
        RuleConfig::default(),
        [
            round_start(pinfu_tenpai(), 0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: Some(tile(TileType::East)),
                    rinshan: false,
                },
            },
        ],
    )
    .unwrap();
    state.players[0].riichi = RiichiState::Accepted;
    let report = legal_actions(&state).unwrap();
    let discards: Vec<_> = report
        .actions
        .iter()
        .filter(|action| matches!(action, Action::Discard { .. }))
        .collect();
    assert_eq!(discards.len(), 1);
    assert!(matches!(
        discards[0],
        Action::Discard { tile: discarded, tsumogiri: true } if discarded.kind == TileType::East
    ));
}
