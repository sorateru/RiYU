use mahjong_core::{
    Action, EventKind, GameEvent, MahjongError, Meld, MeldKind, PlayerId, RiichiStep,
    RoundEndReason, RoundState, RoundWind, RuleConfig, Tile, TileType, TurnPhase,
    game_state_from_json, game_state_to_json, replay_events,
};

fn player(value: u8) -> PlayerId {
    PlayerId::new(value).unwrap()
}

fn tile(kind: TileType) -> Tile {
    Tile::normal(kind)
}

fn initial_hand() -> Vec<Tile> {
    use TileType::*;
    [M1, M2, M3, M4, M5, M6, M7, P1, P2, P3, S7, East, White]
        .into_iter()
        .map(tile)
        .collect()
}

fn round_start(dealer: u8) -> GameEvent {
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
            initial_hand: initial_hand(),
            dora_indicator: tile(TileType::North),
            remaining_tiles: 70,
        },
    }
}

#[test]
fn replays_observer_draw_and_discard() {
    let events = vec![
        round_start(0),
        GameEvent {
            sequence: 1,
            kind: EventKind::Draw {
                player: player(0),
                tile: Some(tile(TileType::P9)),
                rinshan: false,
            },
        },
        GameEvent {
            sequence: 2,
            kind: EventKind::Discard {
                player: player(0),
                tile: tile(TileType::P9),
                tsumogiri: true,
                riichi_declaration: false,
            },
        },
    ];

    let state = replay_events(player(0), RuleConfig::default(), events).unwrap();
    assert_eq!(state.hand, initial_hand());
    assert_eq!(state.drawn_tile, None);
    assert_eq!(state.remaining_tiles, 69);
    assert_eq!(state.players[0].discards.len(), 1);
    assert!(matches!(
        state.phase,
        TurnPhase::AwaitingResponse { discarder, .. } if discarder == player(0)
    ));
}

#[test]
fn opponent_draw_must_not_leak_tile() {
    let mut state = replay_events(player(0), RuleConfig::default(), [round_start(1)]).unwrap();
    let before = state.clone();
    let error = state
        .apply_event(GameEvent {
            sequence: 1,
            kind: EventKind::Draw {
                player: player(1),
                tile: Some(tile(TileType::M9)),
                rinshan: false,
            },
        })
        .unwrap_err();

    assert!(error.to_string().contains("must hide"));
    assert_eq!(state, before, "failed event application must be atomic");
}

#[test]
fn accepts_hidden_opponent_draw() {
    let state = replay_events(
        player(0),
        RuleConfig::default(),
        [
            round_start(1),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(1),
                    tile: None,
                    rinshan: false,
                },
            },
        ],
    )
    .unwrap();
    assert_eq!(state.drawn_tile, None);
    assert!(matches!(
        state.phase,
        TurnPhase::AwaitingDiscard { player: current } if current == player(1)
    ));
}

#[test]
fn json_round_trip_is_validated_and_contains_no_oracle_fields() {
    let state = replay_events(player(0), RuleConfig::default(), [round_start(0)]).unwrap();
    let json = game_state_to_json(&state).unwrap();
    assert!(!json.contains("concealed_hands"));
    assert!(!json.contains("live_wall"));
    assert!(!json.contains("dead_wall"));
    assert_eq!(game_state_from_json(&json).unwrap(), state);
}

#[test]
fn rejects_non_contiguous_event_sequence_without_mutation() {
    let mut state = replay_events(player(0), RuleConfig::default(), [round_start(0)]).unwrap();
    let before = state.clone();
    let error = state
        .apply_event(GameEvent {
            sequence: 7,
            kind: EventKind::Draw {
                player: player(0),
                tile: Some(tile(TileType::M9)),
                rinshan: false,
            },
        })
        .unwrap_err();
    assert!(matches!(
        error,
        MahjongError::SequenceMismatch {
            expected: 1,
            actual: 7
        }
    ));
    assert_eq!(state, before);
}

#[test]
fn riichi_declaration_and_acceptance_update_public_state_and_score() {
    let state = replay_events(
        player(0),
        RuleConfig::default(),
        [
            round_start(0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: Some(tile(TileType::P9)),
                    rinshan: false,
                },
            },
            GameEvent {
                sequence: 2,
                kind: EventKind::Riichi {
                    player: player(0),
                    step: RiichiStep::Declared,
                },
            },
            GameEvent {
                sequence: 3,
                kind: EventKind::Discard {
                    player: player(0),
                    tile: tile(TileType::P9),
                    tsumogiri: true,
                    riichi_declaration: true,
                },
            },
            GameEvent {
                sequence: 4,
                kind: EventKind::Riichi {
                    player: player(0),
                    step: RiichiStep::Accepted,
                },
            },
        ],
    )
    .unwrap();
    assert_eq!(state.scores[0], 24_000);
    assert_eq!(state.round.unwrap().riichi_sticks, 1);
}

#[test]
fn action_json_keeps_action_variants_distinct() {
    let discard = Action::Discard {
        tile: tile(TileType::M5),
        tsumogiri: false,
    };
    discard.validate().unwrap();
    let value = serde_json::to_value(discard).unwrap();
    assert_eq!(value["type"], "discard");
    assert_eq!(value["tile"]["kind"], "m5");
}

#[test]
fn invalid_red_honor_is_rejected() {
    let invalid = Tile {
        kind: TileType::East,
        red: true,
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn chi_marks_discard_called_and_moves_observer_tiles_to_meld() {
    let meld = Meld {
        kind: MeldKind::Chi,
        tiles: vec![tile(TileType::M1), tile(TileType::M2), tile(TileType::M3)],
        called_tile: Some(tile(TileType::M3)),
        from: Some(player(0)),
    };
    let state = replay_events(
        player(1),
        RuleConfig::default(),
        [
            round_start(0),
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
                    tile: tile(TileType::M3),
                    tsumogiri: false,
                    riichi_declaration: false,
                },
            },
            GameEvent {
                sequence: 3,
                kind: EventKind::Chi {
                    player: player(1),
                    meld: meld.clone(),
                },
            },
        ],
    )
    .unwrap();

    assert_eq!(state.hand.len(), 11);
    assert_eq!(state.players[1].melds, vec![meld]);
    assert_eq!(state.players[0].discards[0].called_by, Some(player(1)));
    assert!(matches!(
        state.phase,
        TurnPhase::AwaitingDiscard { player: current } if current == player(1)
    ));
}

#[test]
fn result_must_precede_matching_round_end() {
    let state = replay_events(
        player(0),
        RuleConfig::default(),
        [
            round_start(0),
            GameEvent {
                sequence: 1,
                kind: EventKind::Draw {
                    player: player(0),
                    tile: Some(tile(TileType::P9)),
                    rinshan: false,
                },
            },
            GameEvent {
                sequence: 2,
                kind: EventKind::Tsumo {
                    winner: player(0),
                    score_deltas: [3_000, -1_000, -1_000, -1_000],
                },
            },
            GameEvent {
                sequence: 3,
                kind: EventKind::RoundEnd {
                    reason: RoundEndReason::Tsumo,
                    scores: [28_000, 24_000, 24_000, 24_000],
                },
            },
        ],
    )
    .unwrap();

    assert_eq!(state.phase, TurnPhase::Ended);
    assert_eq!(state.scores, [28_000, 24_000, 24_000, 24_000]);
}
