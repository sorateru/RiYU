use mahjong_core::{
    HandShape, RuleConfig, ScoreLimit, Tile, TileCounts, TileType, WinMethod, analyze_discards,
    calculate_points, calculate_shanten, calculate_ukeire, decompose_winning_hand, is_winning_hand,
};

fn counts(entries: &[(TileType, u8)]) -> TileCounts {
    let mut values = [0_u8; 34];
    for &(tile, count) in entries {
        values[tile.index()] = count;
    }
    TileCounts::from_array(values).unwrap()
}

fn tiles(entries: &[(TileType, u8)]) -> Vec<Tile> {
    entries
        .iter()
        .flat_map(|&(kind, count)| (0..count).map(move |_| Tile::normal(kind)))
        .collect()
}

#[test]
fn standard_complete_and_tenpai_shanten() {
    use TileType::*;
    let complete = counts(&[
        (M1, 1),
        (M2, 1),
        (M3, 1),
        (P1, 1),
        (P2, 1),
        (P3, 1),
        (S1, 1),
        (S2, 1),
        (S3, 1),
        (East, 3),
        (White, 2),
    ]);
    assert_eq!(calculate_shanten(&complete, 0).unwrap().standard, -1);
    assert!(is_winning_hand(&complete, 0).unwrap());

    let mut tenpai = complete;
    tenpai.remove(White).unwrap();
    let result = calculate_shanten(&tenpai, 0).unwrap();
    assert_eq!(result.minimum, 0);
    assert_eq!(result.standard, 0);
}

#[test]
fn special_hand_shanten_is_exact() {
    use TileType::*;
    let seven_pairs_tenpai = counts(&[
        (M1, 2),
        (M2, 2),
        (M3, 2),
        (M4, 2),
        (M5, 2),
        (M6, 2),
        (M7, 1),
    ]);
    assert_eq!(
        calculate_shanten(&seven_pairs_tenpai, 0)
            .unwrap()
            .seven_pairs,
        Some(0)
    );

    let kokushi = counts(&[
        (M1, 1),
        (M9, 1),
        (P1, 1),
        (P9, 1),
        (S1, 1),
        (S9, 1),
        (East, 1),
        (South, 1),
        (West, 1),
        (North, 1),
        (White, 1),
        (Green, 1),
        (Red, 1),
    ]);
    assert_eq!(
        calculate_shanten(&kokushi, 0).unwrap().thirteen_orphans,
        Some(0)
    );
    let ukeire = calculate_ukeire(&kokushi, &kokushi, 0).unwrap();
    assert_eq!(ukeire.len(), 13);
    assert_eq!(
        ukeire.iter().map(|tile| tile.remaining as u16).sum::<u16>(),
        39
    );
}

#[test]
fn fixed_meld_reduces_required_concealed_structure() {
    use TileType::*;
    let concealed = counts(&[
        (P1, 1),
        (P2, 1),
        (P3, 1),
        (S1, 1),
        (S2, 1),
        (S3, 1),
        (East, 3),
        (White, 2),
    ]);
    assert_eq!(calculate_shanten(&concealed, 1).unwrap().minimum, -1);
    assert!(is_winning_hand(&concealed, 1).unwrap());
}

#[test]
fn decomposition_retains_multiple_valid_interpretations() {
    use TileType::*;
    let hand = counts(&[
        (M1, 2),
        (M2, 2),
        (M3, 2),
        (M4, 2),
        (M5, 2),
        (M6, 2),
        (M7, 2),
    ]);
    let shapes = decompose_winning_hand(&hand, 0, Some(M7)).unwrap();
    assert!(
        shapes
            .iter()
            .any(|shape| matches!(shape, HandShape::SevenPairs { .. }))
    );
    assert!(
        shapes
            .iter()
            .any(|shape| matches!(shape, HandShape::Standard { .. }))
    );
}

#[test]
fn discard_analysis_orders_by_shanten_then_live_ukeire() {
    use TileType::*;
    let hand = tiles(&[
        (M1, 1),
        (M2, 1),
        (M3, 1),
        (P1, 1),
        (P2, 1),
        (P3, 1),
        (S1, 1),
        (S2, 1),
        (S3, 1),
        (East, 3),
        (White, 2),
    ]);
    let known = TileCounts::from_tiles(hand.iter().copied()).unwrap();
    let candidates = analyze_discards(&hand, &known, 0).unwrap();
    assert!(!candidates.is_empty());
    assert_eq!(candidates[0].shanten.minimum, 0);
    assert!(candidates.windows(2).all(|pair| {
        pair[0].shanten.minimum < pair[1].shanten.minimum
            || pair[0].shanten.minimum == pair[1].shanten.minimum
                && pair[0].ukeire_count >= pair[1].ukeire_count
    }));
}

#[test]
fn standard_point_table_examples() {
    let rules = RuleConfig::default();
    let ron = calculate_points(1, 30, 0, false, WinMethod::Ron, 0, 0, &rules).unwrap();
    assert_eq!(ron.ron_payment, Some(1_000));
    assert_eq!(ron.limit, ScoreLimit::Normal);

    let dealer_ron = calculate_points(1, 30, 0, true, WinMethod::Ron, 0, 0, &rules).unwrap();
    assert_eq!(dealer_ron.ron_payment, Some(1_500));

    let tsumo = calculate_points(1, 30, 0, false, WinMethod::Tsumo, 0, 0, &rules).unwrap();
    assert_eq!(tsumo.tsumo_from_dealer, Some(500));
    assert_eq!(tsumo.tsumo_from_non_dealer, Some(300));
    assert_eq!(tsumo.total_payment, 1_100);

    let mangan = calculate_points(5, 30, 0, false, WinMethod::Ron, 0, 0, &rules).unwrap();
    assert_eq!(mangan.limit, ScoreLimit::Mangan);
    assert_eq!(mangan.ron_payment, Some(8_000));
}

#[test]
fn honba_and_riichi_sticks_are_kept_separate() {
    let payment = calculate_points(
        2,
        30,
        0,
        false,
        WinMethod::Ron,
        2,
        3,
        &RuleConfig::default(),
    )
    .unwrap();
    assert_eq!(payment.ron_payment, Some(2_600));
    assert_eq!(payment.total_received, 5_600);
}

#[test]
fn dora_indicator_cycles_are_correct() {
    assert_eq!(TileType::M9.dora(), TileType::M1);
    assert_eq!(TileType::North.dora(), TileType::East);
    assert_eq!(TileType::Red.dora(), TileType::White);
}
