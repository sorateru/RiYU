use mahjong_core::{
    Meld, MeldKind, RiichiDeclaration, RuleConfig, ScoreLimit, Tile, TileType, WaitType,
    WinContext, WinMethod, Yaku, score_hand,
};

fn tile(kind: TileType) -> Tile {
    Tile::normal(kind)
}

fn tiles(entries: &[(TileType, u8)]) -> Vec<Tile> {
    entries
        .iter()
        .flat_map(|&(kind, count)| (0..count).map(move |_| tile(kind)))
        .collect()
}

fn context(method: WinMethod) -> WinContext {
    WinContext {
        method,
        seat_wind: TileType::South,
        round_wind: TileType::East,
        riichi: RiichiDeclaration::None,
        ippatsu: false,
        rinshan: false,
        chankan: false,
        haitei: false,
        houtei: false,
        tenhou: false,
        chiihou: false,
        dora_indicators: Vec::new(),
        ura_dora_indicators: Vec::new(),
        honba: 0,
        riichi_sticks: 0,
    }
}

#[test]
fn closed_pinfu_tsumo_has_twenty_fu() {
    use TileType::*;
    let hand = tiles(&[
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
        (S4, 1),
        (P5, 2),
    ]);
    let result = score_hand(
        &hand,
        &[],
        tile(S4),
        &context(WinMethod::Tsumo),
        &RuleConfig::default(),
    )
    .unwrap();

    assert_eq!(result.wait, WaitType::Ryanmen);
    assert_eq!(result.fu.rounded_total, 20);
    assert!(result.yaku.iter().any(|entry| entry.yaku == Yaku::Pinfu));
    assert!(
        result
            .yaku
            .iter()
            .any(|entry| entry.yaku == Yaku::MenzenTsumo)
    );
    assert_eq!(result.total_han, 2);
    assert_eq!(result.payment.total_payment, 1_500);
}

#[test]
fn open_dragon_pon_is_a_valid_yaku_and_adds_fu() {
    use TileType::*;
    let concealed = tiles(&[
        (M1, 1),
        (M2, 3),
        (M3, 1),
        (P4, 1),
        (P5, 1),
        (P6, 1),
        (S7, 1),
        (S8, 1),
        (S9, 1),
    ]);
    let white_pon = Meld {
        kind: MeldKind::Pon,
        tiles: vec![tile(White), tile(White), tile(White)],
        called_tile: Some(tile(White)),
        from: Some(mahjong_core::PlayerId::new(1).unwrap()),
    };
    let result = score_hand(
        &concealed,
        &[white_pon],
        tile(S9),
        &context(WinMethod::Ron),
        &RuleConfig::default(),
    )
    .unwrap();

    assert!(result.yaku.iter().any(|entry| {
        entry.yaku
            == Yaku::YakuhaiDragon {
                tile: TileType::White,
            }
    }));
    assert_eq!(result.fu.melds, 4);
    assert_eq!(result.fu.rounded_total, 30);
    assert_eq!(result.payment.ron_payment, Some(1_000));
}

#[test]
fn seven_pairs_is_fixed_twenty_five_fu() {
    use TileType::*;
    let hand = tiles(&[
        (M1, 2),
        (M2, 2),
        (P3, 2),
        (P4, 2),
        (S5, 2),
        (S6, 2),
        (East, 2),
    ]);
    let result = score_hand(
        &hand,
        &[],
        tile(East),
        &context(WinMethod::Ron),
        &RuleConfig::default(),
    )
    .unwrap();
    assert!(
        result
            .yaku
            .iter()
            .any(|entry| entry.yaku == Yaku::Chiitoitsu)
    );
    assert_eq!(result.fu.rounded_total, 25);
    assert_eq!(result.payment.ron_payment, Some(1_600));
}

#[test]
fn thirteen_sided_kokushi_uses_configured_double_yakuman() {
    use TileType::*;
    let hand = tiles(&[
        (M1, 2),
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
    let result = score_hand(
        &hand,
        &[],
        tile(M1),
        &context(WinMethod::Ron),
        &RuleConfig::default(),
    )
    .unwrap();
    assert_eq!(result.yakuman_multiplier, 2);
    assert_eq!(result.payment.limit, ScoreLimit::Yakuman { multiplier: 2 });
    assert_eq!(result.payment.ron_payment, Some(64_000));
}

#[test]
fn dora_without_yaku_is_rejected() {
    use TileType::*;
    let concealed = tiles(&[
        (M1, 1),
        (M2, 3),
        (M3, 1),
        (P4, 1),
        (P5, 1),
        (P6, 1),
        (S7, 1),
        (S8, 1),
        (S9, 1),
    ]);
    let open_pon = Meld {
        kind: MeldKind::Pon,
        tiles: vec![tile(P9), tile(P9), tile(P9)],
        called_tile: Some(tile(P9)),
        from: Some(mahjong_core::PlayerId::new(1).unwrap()),
    };
    let mut win_context = context(WinMethod::Ron);
    win_context.dora_indicators = vec![P8];
    let error = score_hand(
        &concealed,
        &[open_pon],
        tile(S9),
        &win_context,
        &RuleConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(error, mahjong_core::MahjongError::NoYaku));
}
