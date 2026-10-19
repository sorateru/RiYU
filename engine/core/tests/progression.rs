use mahjong_core::{
    MatchLength, PlayerId, RoundOutcome, RoundState, RoundWind, RuleConfig, next_round,
    noten_penalty,
};

fn round(wind: RoundWind, hand_number: u8, dealer: u8) -> RoundState {
    RoundState {
        wind,
        hand_number,
        honba: 0,
        riichi_sticks: 2,
        dealer: PlayerId::new(dealer).unwrap(),
    }
}

#[test]
fn noten_exchange_is_zero_sum_for_all_tenpai_counts() {
    assert_eq!(noten_penalty([false; 4]), [0; 4]);
    assert_eq!(
        noten_penalty([true, false, false, false]),
        [3000, -1000, -1000, -1000]
    );
    assert_eq!(
        noten_penalty([true, true, false, false]),
        [1500, 1500, -1500, -1500]
    );
    assert_eq!(
        noten_penalty([true, true, true, false]),
        [1000, 1000, 1000, -3000]
    );
    assert_eq!(noten_penalty([true; 4]), [0; 4]);
}

#[test]
fn dealer_win_repeats_and_increments_honba() {
    let current = round(RoundWind::East, 2, 1);
    let transition = next_round(
        &current,
        RoundOutcome::Win {
            winner: PlayerId::new(1).unwrap(),
        },
        [25_000; 4],
        &RuleConfig::default(),
    )
    .unwrap();
    let next = transition.next_round.unwrap();
    assert!(transition.dealer_continues);
    assert_eq!(next.hand_number, 2);
    assert_eq!(next.honba, 1);
    assert_eq!(next.riichi_sticks, 0);
}

#[test]
fn nondealer_win_moves_dealer_and_resets_honba() {
    let current = round(RoundWind::East, 4, 0);
    let transition = next_round(
        &current,
        RoundOutcome::Win {
            winner: PlayerId::new(2).unwrap(),
        },
        [20_000, 25_000, 35_000, 20_000],
        &RuleConfig::default(),
    )
    .unwrap();
    let next = transition.next_round.unwrap();
    assert_eq!(next.wind, RoundWind::South);
    assert_eq!(next.hand_number, 1);
    assert_eq!(next.dealer, PlayerId::new(1).unwrap());
    assert_eq!(next.honba, 0);
}

#[test]
fn south_four_ends_when_target_was_reached() {
    let current = round(RoundWind::South, 4, 3);
    let transition = next_round(
        &current,
        RoundOutcome::Win {
            winner: PlayerId::new(1).unwrap(),
        },
        [20_000, 40_000, 20_000, 20_000],
        &RuleConfig::default(),
    )
    .unwrap();
    assert!(transition.match_ended);
    assert!(transition.next_round.is_none());
}

#[test]
fn east_only_can_extend_to_south_when_no_one_reaches_target() {
    let current = round(RoundWind::East, 4, 3);
    let rules = RuleConfig {
        match_length: MatchLength::EastOnly,
        ..RuleConfig::default()
    };
    let transition = next_round(
        &current,
        RoundOutcome::ExhaustiveDraw {
            dealer_tenpai: false,
        },
        [25_000; 4],
        &rules,
    )
    .unwrap();
    let next = transition.next_round.unwrap();
    assert_eq!(next.wind, RoundWind::South);
    assert_eq!(next.hand_number, 1);
    assert_eq!(next.honba, 1);
}
