use mahjong_core::{
    Action, FourPlayerSimulator, Meld, MeldKind, Payment, PlayerId, Responsibility, RonClaim,
    RoundState, RoundWind, RuleConfig, ScoreLimit, Tile, TileType, TurnPhase, WinMethod,
    calculate_points, settle_ron,
};

fn player(value: u8) -> PlayerId {
    PlayerId::new(value).unwrap()
}

#[test]
fn double_ron_awards_sticks_to_first_winner_and_honba_to_each() {
    let rules = RuleConfig::default();
    let payment = calculate_points(1, 30, 0, false, WinMethod::Ron, 2, 0, &rules).unwrap();
    let claims = [
        RonClaim {
            winner: player(1),
            payment: payment.clone(),
            responsibility: None,
        },
        RonClaim {
            winner: player(2),
            payment: payment.clone(),
            responsibility: None,
        },
    ];
    let deltas = settle_ron(player(0), &claims, 2, 1, &rules).unwrap();
    let each = payment.ron_payment.unwrap() as i32;
    assert_eq!(deltas, [-each * 2, each + 1_000, each, 0]);
    assert_eq!(deltas.iter().sum::<i32>(), 1_000);
}

#[test]
fn ron_responsibility_splits_yakuman_component() {
    let rules = RuleConfig::default();
    let payment = calculate_points(0, 20, 1, false, WinMethod::Ron, 0, 0, &rules).unwrap();
    let deltas = settle_ron(
        player(0),
        &[RonClaim {
            winner: player(1),
            payment,
            responsibility: Some(Responsibility {
                liable: player(2),
                yakuman_multiplier: 1,
            }),
        }],
        0,
        0,
        &rules,
    )
    .unwrap();
    assert_eq!(deltas, [-16_000, 32_000, -16_000, 0]);
}

#[test]
fn local_simulator_completes_a_no_call_round() {
    let round = RoundState {
        wind: RoundWind::East,
        hand_number: 1,
        honba: 0,
        riichi_sticks: 0,
        dealer: player(0),
    };
    let mut simulator =
        FourPlayerSimulator::new(42, RuleConfig::default(), round, [25_000; 4]).unwrap();
    for _ in 0..300 {
        match simulator.observation(player(0)).phase {
            TurnPhase::AwaitingDraw { .. } | TurnPhase::AwaitingRinshanDraw { .. } => {
                simulator.draw().unwrap();
            }
            TurnPhase::AwaitingDiscard { player: actor } => {
                let legal = simulator.legal_actions(actor).unwrap();
                if let Some(tsumo) = legal.iter().find(|action| matches!(action, Action::Tsumo)) {
                    simulator.act(actor, tsumo.clone()).unwrap();
                } else {
                    let discard = legal
                        .into_iter()
                        .find(|action| matches!(action, Action::Discard { .. }))
                        .unwrap();
                    simulator.act(actor, discard).unwrap();
                }
            }
            TurnPhase::AwaitingResponse { discarder, .. }
            | TurnPhase::AwaitingKanResponse {
                declarer: discarder,
                ..
            } => {
                let mut choices = Vec::new();
                for responder in PlayerId::ALL {
                    if responder == discarder {
                        continue;
                    }
                    let legal = simulator.legal_actions(responder).unwrap();
                    let choice = legal
                        .iter()
                        .find(|action| matches!(action, Action::Ron { .. }))
                        .cloned()
                        .unwrap_or(Action::Pass);
                    choices.push((responder, choice));
                }
                simulator.resolve_responses(&choices).unwrap();
                if matches!(
                    simulator.observation(player(0)).phase,
                    TurnPhase::AwaitingResponse { .. } | TurnPhase::AwaitingKanResponse { .. }
                ) {
                    simulator.draw().unwrap();
                }
            }
            TurnPhase::Ended => break,
            phase => panic!("unexpected phase {phase:?}"),
        }
    }
    assert!(matches!(
        simulator.observation(player(0)).phase,
        TurnPhase::Ended
    ));
    let complete = simulator.complete_state(player(0)).unwrap();
    assert!(!complete.public.information.opponent_hands_visible);
}

#[test]
fn payment_fixture_type_is_serializable() {
    let payment = Payment {
        han: 1,
        fu: 30,
        limit: ScoreLimit::Normal,
        basic_points: 240,
        dealer: false,
        method: WinMethod::Ron,
        ron_payment: Some(1_000),
        tsumo_from_dealer: None,
        tsumo_from_non_dealer: None,
        total_payment: 1_000,
        total_received: 1_000,
    };
    assert!(
        serde_json::to_string(&payment)
            .unwrap()
            .contains("ron_payment")
    );
}

#[test]
fn meld_fixture_is_a_real_kan() {
    let meld = Meld {
        kind: MeldKind::Ankan,
        tiles: vec![Tile::normal(TileType::East); 4],
        called_tile: None,
        from: None,
    };
    meld.validate().unwrap();
}
