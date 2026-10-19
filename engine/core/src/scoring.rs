use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    HandShape, MahjongError, Meld, MeldKind, Payment, Result, RuleConfig, ShapeMeld, Tile,
    TileCounts, TileType, WinMethod, calculate_points, decompose_winning_hand,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiichiDeclaration {
    None,
    Riichi,
    DoubleRiichi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitType {
    Ryanmen,
    Kanchan,
    Penchan,
    Tanki,
    Shanpon,
    ThirteenOrphansSingle,
    ThirteenOrphansThirteenSided,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Yaku {
    Riichi,
    DoubleRiichi,
    Ippatsu,
    MenzenTsumo,
    Tanyao,
    Pinfu,
    Iipeikou,
    Ryanpeikou,
    YakuhaiDragon { tile: TileType },
    YakuhaiSeatWind,
    YakuhaiRoundWind,
    Haitei,
    Houtei,
    RinshanKaihou,
    Chankan,
    Chiitoitsu,
    Toitoi,
    Sanankou,
    Sankantsu,
    SanshokuDoujun,
    SanshokuDoukou,
    Ittsu,
    Chanta,
    Junchan,
    Honroutou,
    Shousangen,
    Honitsu,
    Chinitsu,
    Tenhou,
    Chiihou,
    KokushiMusou,
    KokushiMusouThirteenWait,
    Suuankou,
    SuuankouTanki,
    Daisangen,
    Shousuushii,
    Daisuushii,
    Tsuuiisou,
    Chinroutou,
    Ryuuiisou,
    ChuurenPoutou,
    JunseiChuurenPoutou,
    Suukantsu,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YakuEntry {
    pub yaku: Yaku,
    pub han: u8,
    pub yakuman_multiplier: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FuBreakdown {
    pub base: u16,
    pub menzen_ron: u16,
    pub tsumo: u16,
    pub melds: u16,
    pub pair: u16,
    pub wait: u16,
    pub raw_total: u16,
    pub rounded_total: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WinContext {
    pub method: WinMethod,
    pub seat_wind: TileType,
    pub round_wind: TileType,
    pub riichi: RiichiDeclaration,
    pub ippatsu: bool,
    pub rinshan: bool,
    pub chankan: bool,
    pub haitei: bool,
    pub houtei: bool,
    pub tenhou: bool,
    pub chiihou: bool,
    pub dora_indicators: Vec<TileType>,
    pub ura_dora_indicators: Vec<TileType>,
    pub honba: u16,
    pub riichi_sticks: u16,
}

impl WinContext {
    pub fn validate(&self) -> Result<()> {
        if !matches!(
            self.seat_wind,
            TileType::East | TileType::South | TileType::West | TileType::North
        ) || !matches!(
            self.round_wind,
            TileType::East | TileType::South | TileType::West | TileType::North
        ) {
            return Err(MahjongError::InvalidAction(
                "seat_wind and round_wind must be wind tiles".into(),
            ));
        }
        if self.ippatsu && self.riichi == RiichiDeclaration::None {
            return Err(MahjongError::InvalidAction(
                "ippatsu requires riichi".into(),
            ));
        }
        if !self.ura_dora_indicators.is_empty() && self.riichi == RiichiDeclaration::None {
            return Err(MahjongError::InvalidAction(
                "ura dora requires riichi".into(),
            ));
        }
        if self.haitei && self.method != WinMethod::Tsumo
            || self.houtei && self.method != WinMethod::Ron
            || self.rinshan && self.method != WinMethod::Tsumo
            || self.chankan && self.method != WinMethod::Ron
        {
            return Err(MahjongError::InvalidAction(
                "win situation conflicts with win method".into(),
            ));
        }
        if self.tenhou && (self.seat_wind != TileType::East || self.method != WinMethod::Tsumo) {
            return Err(MahjongError::InvalidAction(
                "tenhou requires dealer tsumo".into(),
            ));
        }
        if self.chiihou && (self.seat_wind == TileType::East || self.method != WinMethod::Tsumo) {
            return Err(MahjongError::InvalidAction(
                "chiihou requires non-dealer tsumo".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreEvaluation {
    pub shape: HandShape,
    pub wait: WaitType,
    pub yaku: Vec<YakuEntry>,
    pub yaku_han: u8,
    pub dora: u8,
    pub ura_dora: u8,
    pub aka_dora: u8,
    pub total_han: u8,
    pub yakuman_multiplier: u8,
    pub fu: FuBreakdown,
    pub payment: Payment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Completion {
    Pair,
    Meld(usize),
    Special(WaitType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupKind {
    Sequence(TileType),
    Triplet(TileType),
    Kan(TileType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Group {
    kind: GroupKind,
    open: bool,
    concealed_index: Option<usize>,
}

/// Score every valid interpretation and return the highest payment.
#[allow(clippy::too_many_arguments)]
pub fn score_hand(
    concealed_tiles: &[Tile],
    fixed_melds: &[Meld],
    winning_tile: Tile,
    context: &WinContext,
    rules: &RuleConfig,
) -> Result<ScoreEvaluation> {
    context.validate()?;
    rules.validate()?;
    winning_tile.validate()?;
    for meld in fixed_melds {
        meld.validate()?;
    }
    if fixed_melds.len() > 4 {
        return Err(MahjongError::InvalidAction(
            "more than four fixed melds".into(),
        ));
    }
    if !concealed_tiles.contains(&winning_tile) {
        return Err(MahjongError::InvalidAction(
            "winning_tile must be included in concealed_tiles".into(),
        ));
    }
    let counts = TileCounts::from_tiles(concealed_tiles.iter().copied())?;
    let shapes = decompose_winning_hand(&counts, fixed_melds.len() as u8, Some(winning_tile.kind))?;
    if shapes.is_empty() {
        return Err(MahjongError::InvalidAction(
            "tiles do not form a winning hand".into(),
        ));
    }
    validate_physical_tile_counts(concealed_tiles, fixed_melds)?;
    let closed = fixed_melds.iter().all(|meld| meld.kind == MeldKind::Ankan);
    if context.riichi != RiichiDeclaration::None && !closed {
        return Err(MahjongError::InvalidAction(
            "riichi is not valid for an open hand".into(),
        ));
    }

    let mut best: Option<ScoreEvaluation> = None;
    for shape in shapes {
        for completion in completions(&shape, winning_tile.kind) {
            let evaluation = match evaluate_interpretation(
                shape.clone(),
                completion,
                concealed_tiles,
                fixed_melds,
                winning_tile,
                context,
                rules,
                closed,
            ) {
                Ok(evaluation) => evaluation,
                Err(MahjongError::NoYaku) => continue,
                Err(error) => return Err(error),
            };
            if best
                .as_ref()
                .is_none_or(|current| better_score(&evaluation, current))
            {
                best = Some(evaluation);
            }
        }
    }
    best.ok_or(MahjongError::NoYaku)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_interpretation(
    shape: HandShape,
    completion: Completion,
    concealed_tiles: &[Tile],
    fixed_melds: &[Meld],
    winning_tile: Tile,
    context: &WinContext,
    rules: &RuleConfig,
    closed: bool,
) -> Result<ScoreEvaluation> {
    let wait = wait_type(&shape, completion, winning_tile.kind)?;
    let groups = groups(&shape, fixed_melds);
    let all_counts = all_tile_counts(concealed_tiles, fixed_melds)?;
    let mut yaku = detect_yakuman(
        &shape,
        completion,
        &groups,
        &all_counts,
        context,
        rules,
        closed,
        winning_tile.kind,
    );
    let yakuman_multiplier: u8 = yaku.iter().map(|entry| entry.yakuman_multiplier).sum();
    if yakuman_multiplier == 0 {
        yaku = detect_regular_yaku(
            &shape,
            completion,
            &groups,
            &all_counts,
            context,
            closed,
            wait,
            rules,
        );
    }
    let yaku_han: u8 = yaku.iter().map(|entry| entry.han).sum();
    if yakuman_multiplier == 0 && yaku_han == 0 {
        return Err(MahjongError::NoYaku);
    }

    let (dora, ura_dora, aka_dora) = if yakuman_multiplier == 0 {
        (
            count_indicator_dora(&all_counts, &context.dora_indicators),
            count_indicator_dora(&all_counts, &context.ura_dora_indicators),
            count_red_dora(concealed_tiles, fixed_melds),
        )
    } else {
        (0, 0, 0)
    };
    let total_han = yaku_han
        .saturating_add(dora)
        .saturating_add(ura_dora)
        .saturating_add(aka_dora);
    let fu = calculate_fu(
        &shape, completion, &groups, context, closed, wait, &yaku, rules,
    );
    let payment = calculate_points(
        total_han,
        fu.rounded_total,
        yakuman_multiplier,
        context.seat_wind == TileType::East,
        context.method,
        context.honba,
        context.riichi_sticks,
        rules,
    )?;
    Ok(ScoreEvaluation {
        shape,
        wait,
        yaku,
        yaku_han,
        dora,
        ura_dora,
        aka_dora,
        total_han,
        yakuman_multiplier,
        fu,
        payment,
    })
}

fn completions(shape: &HandShape, winning: TileType) -> Vec<Completion> {
    match shape {
        HandShape::SevenPairs { .. } => vec![Completion::Special(WaitType::Tanki)],
        HandShape::ThirteenOrphans {
            thirteen_sided_wait,
            ..
        } => vec![Completion::Special(if *thirteen_sided_wait {
            WaitType::ThirteenOrphansThirteenSided
        } else {
            WaitType::ThirteenOrphansSingle
        })],
        HandShape::Standard {
            pair,
            concealed_melds,
        } => {
            let mut result = Vec::new();
            if *pair == winning {
                result.push(Completion::Pair);
            }
            for (index, meld) in concealed_melds.iter().enumerate() {
                let contains = match meld {
                    ShapeMeld::Triplet { tile } => *tile == winning,
                    ShapeMeld::Sequence { start } => {
                        let start = start.index();
                        (start..=start + 2).contains(&winning.index())
                    }
                };
                if contains {
                    result.push(Completion::Meld(index));
                }
            }
            result
        }
    }
}

fn wait_type(
    shape: &HandShape,
    completion: Completion,
    winning_tile: TileType,
) -> Result<WaitType> {
    match completion {
        Completion::Special(wait) => Ok(wait),
        Completion::Pair => Ok(WaitType::Tanki),
        Completion::Meld(index) => {
            let HandShape::Standard {
                concealed_melds, ..
            } = shape
            else {
                return Err(MahjongError::InvalidState(
                    "meld completion requires standard shape".into(),
                ));
            };
            match concealed_melds[index] {
                ShapeMeld::Triplet { .. } => Ok(WaitType::Shanpon),
                ShapeMeld::Sequence { start } => {
                    let rank = start.rank().expect("sequence is suited");
                    let position =
                        winning_tile
                            .index()
                            .checked_sub(start.index())
                            .ok_or_else(|| {
                                MahjongError::InvalidState(
                                    "winning tile is outside completed sequence".into(),
                                )
                            })?;
                    if position > 2 {
                        return Err(MahjongError::InvalidState(
                            "winning tile is outside completed sequence".into(),
                        ));
                    }
                    if position == 1 {
                        Ok(WaitType::Kanchan)
                    } else if (rank == 1 && position == 2) || (rank == 7 && position == 0) {
                        Ok(WaitType::Penchan)
                    } else {
                        Ok(WaitType::Ryanmen)
                    }
                }
            }
        }
    }
}

fn groups(shape: &HandShape, fixed_melds: &[Meld]) -> Vec<Group> {
    let mut output = Vec::new();
    if let HandShape::Standard {
        concealed_melds, ..
    } = shape
    {
        for (index, meld) in concealed_melds.iter().enumerate() {
            output.push(Group {
                kind: match meld {
                    ShapeMeld::Sequence { start } => GroupKind::Sequence(*start),
                    ShapeMeld::Triplet { tile } => GroupKind::Triplet(*tile),
                },
                open: false,
                concealed_index: Some(index),
            });
        }
    }
    for meld in fixed_melds {
        output.push(Group {
            kind: match meld.kind {
                MeldKind::Chi => GroupKind::Sequence(sequence_start(meld)),
                MeldKind::Pon => GroupKind::Triplet(meld.tiles[0].kind),
                MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Shouminkan => {
                    GroupKind::Kan(meld.tiles[0].kind)
                }
            },
            open: meld.kind != MeldKind::Ankan,
            concealed_index: None,
        });
    }
    output
}

fn sequence_start(meld: &Meld) -> TileType {
    *meld
        .tiles
        .iter()
        .map(|tile| &tile.kind)
        .min_by_key(|tile| tile.index())
        .expect("validated chi")
}

#[allow(clippy::too_many_arguments)]
fn detect_regular_yaku(
    shape: &HandShape,
    completion: Completion,
    groups: &[Group],
    counts: &TileCounts,
    context: &WinContext,
    closed: bool,
    wait: WaitType,
    rules: &RuleConfig,
) -> Vec<YakuEntry> {
    let mut result = Vec::new();
    let mut add = |yaku, han| {
        result.push(YakuEntry {
            yaku,
            han,
            yakuman_multiplier: 0,
        });
    };

    match context.riichi {
        RiichiDeclaration::Riichi if closed => add(Yaku::Riichi, 1),
        RiichiDeclaration::DoubleRiichi if closed => add(Yaku::DoubleRiichi, 2),
        _ => {}
    }
    if context.ippatsu && closed {
        add(Yaku::Ippatsu, 1);
    }
    if context.method == WinMethod::Tsumo && closed {
        add(Yaku::MenzenTsumo, 1);
    }
    if context.haitei {
        add(Yaku::Haitei, 1);
    }
    if context.houtei {
        add(Yaku::Houtei, 1);
    }
    if context.rinshan {
        add(Yaku::RinshanKaihou, 1);
    }
    if context.chankan {
        add(Yaku::Chankan, 1);
    }
    if (closed || rules.open_tanyao) && all_tiles_match(counts, |tile| !tile.is_terminal_or_honor())
    {
        add(Yaku::Tanyao, 1);
    }

    match shape {
        HandShape::SevenPairs { .. } => add(Yaku::Chiitoitsu, 2),
        HandShape::Standard { pair, .. } => {
            let triplet_tiles: Vec<_> = groups
                .iter()
                .filter_map(|group| match group.kind {
                    GroupKind::Triplet(tile) | GroupKind::Kan(tile) => Some(tile),
                    GroupKind::Sequence(_) => None,
                })
                .collect();
            for dragon in [TileType::White, TileType::Green, TileType::Red] {
                if triplet_tiles.contains(&dragon) {
                    add(Yaku::YakuhaiDragon { tile: dragon }, 1);
                }
            }
            if triplet_tiles.contains(&context.seat_wind) {
                add(Yaku::YakuhaiSeatWind, 1);
            }
            if triplet_tiles.contains(&context.round_wind) {
                add(Yaku::YakuhaiRoundWind, 1);
            }

            let sequence_starts: Vec<_> = groups
                .iter()
                .filter_map(|group| match group.kind {
                    GroupKind::Sequence(start) => Some(start),
                    _ => None,
                })
                .collect();
            if closed
                && groups
                    .iter()
                    .all(|group| matches!(group.kind, GroupKind::Sequence(_)))
                && !is_value_pair(*pair, context)
                && wait == WaitType::Ryanmen
            {
                add(Yaku::Pinfu, 1);
            }
            if closed {
                let duplicates = duplicate_sequence_pairs(&sequence_starts);
                if duplicates >= 2 {
                    add(Yaku::Ryanpeikou, 3);
                } else if duplicates == 1 {
                    add(Yaku::Iipeikou, 1);
                }
            }
            if has_sanshoku_doujun(&sequence_starts) {
                add(Yaku::SanshokuDoujun, if closed { 2 } else { 1 });
            }
            if has_ittsu(&sequence_starts) {
                add(Yaku::Ittsu, if closed { 2 } else { 1 });
            }
            if groups
                .iter()
                .all(|group| !matches!(group.kind, GroupKind::Sequence(_)))
            {
                add(Yaku::Toitoi, 2);
            }
            let concealed_triplets = groups
                .iter()
                .filter(|group| group_is_concealed_triplet(group, completion, context.method))
                .count();
            if concealed_triplets >= 3 {
                add(Yaku::Sanankou, 2);
            }
            if groups
                .iter()
                .filter(|group| matches!(group.kind, GroupKind::Kan(_)))
                .count()
                >= 3
            {
                add(Yaku::Sankantsu, 2);
            }
            if has_sanshoku_doukou(&triplet_tiles) {
                add(Yaku::SanshokuDoukou, 2);
            }
            let dragon_triplets = triplet_tiles
                .iter()
                .filter(|tile| matches!(tile, TileType::White | TileType::Green | TileType::Red))
                .count();
            if dragon_triplets == 2
                && matches!(pair, TileType::White | TileType::Green | TileType::Red)
            {
                add(Yaku::Shousangen, 2);
            }
            let (all_terminal_groups, has_sequence, has_honor) =
                terminal_group_properties(*pair, groups);
            if all_terminal_groups && has_sequence {
                if has_honor {
                    add(Yaku::Chanta, if closed { 2 } else { 1 });
                } else {
                    add(Yaku::Junchan, if closed { 3 } else { 2 });
                }
            }
        }
        HandShape::ThirteenOrphans { .. } => {}
    }

    if all_tiles_match(counts, TileType::is_terminal_or_honor) {
        add(Yaku::Honroutou, 2);
    }
    if let Some(has_honors) = one_suit_only(counts) {
        if has_honors {
            add(Yaku::Honitsu, if closed { 3 } else { 2 });
        } else {
            add(Yaku::Chinitsu, if closed { 6 } else { 5 });
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn detect_yakuman(
    shape: &HandShape,
    completion: Completion,
    groups: &[Group],
    counts: &TileCounts,
    context: &WinContext,
    rules: &RuleConfig,
    closed: bool,
    winning_tile: TileType,
) -> Vec<YakuEntry> {
    let mut result = Vec::new();
    let mut add = |yaku, multiplier| {
        result.push(YakuEntry {
            yaku,
            han: 0,
            yakuman_multiplier: multiplier,
        });
    };
    if context.tenhou {
        add(Yaku::Tenhou, 1);
    }
    if context.chiihou {
        add(Yaku::Chiihou, 1);
    }
    match shape {
        HandShape::ThirteenOrphans {
            thirteen_sided_wait,
            ..
        } => add(
            if *thirteen_sided_wait {
                Yaku::KokushiMusouThirteenWait
            } else {
                Yaku::KokushiMusou
            },
            if *thirteen_sided_wait && rules.double_yakuman {
                2
            } else {
                1
            },
        ),
        HandShape::Standard { pair, .. } => {
            let triplets: Vec<_> = groups
                .iter()
                .filter_map(|group| match group.kind {
                    GroupKind::Triplet(tile) | GroupKind::Kan(tile) => Some(tile),
                    _ => None,
                })
                .collect();
            if [TileType::White, TileType::Green, TileType::Red]
                .iter()
                .all(|tile| triplets.contains(tile))
            {
                add(Yaku::Daisangen, 1);
            }
            let wind_triplets = [
                TileType::East,
                TileType::South,
                TileType::West,
                TileType::North,
            ]
            .iter()
            .filter(|tile| triplets.contains(tile))
            .count();
            if wind_triplets == 4 {
                add(Yaku::Daisuushii, if rules.double_yakuman { 2 } else { 1 });
            } else if wind_triplets == 3
                && matches!(
                    pair,
                    TileType::East | TileType::South | TileType::West | TileType::North
                )
            {
                add(Yaku::Shousuushii, 1);
            }
            let concealed_triplets = groups
                .iter()
                .filter(|group| group_is_concealed_triplet(group, completion, context.method))
                .count();
            if concealed_triplets == 4 {
                let tanki = matches!(completion, Completion::Pair);
                add(
                    if tanki {
                        Yaku::SuuankouTanki
                    } else {
                        Yaku::Suuankou
                    },
                    if tanki && rules.double_yakuman { 2 } else { 1 },
                );
            }
            if groups
                .iter()
                .filter(|group| matches!(group.kind, GroupKind::Kan(_)))
                .count()
                == 4
            {
                add(Yaku::Suukantsu, 1);
            }
        }
        HandShape::SevenPairs { .. } => {}
    }
    if all_tiles_match(counts, TileType::is_honor) {
        add(Yaku::Tsuuiisou, 1);
    }
    if all_tiles_match(counts, TileType::is_terminal) {
        add(Yaku::Chinroutou, 1);
    }
    if all_tiles_match(counts, is_green_tile) {
        add(Yaku::Ryuuiisou, 1);
    }
    if closed {
        if let Some(pure) = chuuren(counts, shape, winning_tile) {
            add(
                if pure {
                    Yaku::JunseiChuurenPoutou
                } else {
                    Yaku::ChuurenPoutou
                },
                if pure && rules.double_yakuman { 2 } else { 1 },
            );
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn calculate_fu(
    shape: &HandShape,
    completion: Completion,
    groups: &[Group],
    context: &WinContext,
    closed: bool,
    wait: WaitType,
    yaku: &[YakuEntry],
    rules: &RuleConfig,
) -> FuBreakdown {
    if matches!(shape, HandShape::SevenPairs { .. }) {
        return FuBreakdown {
            base: 25,
            raw_total: 25,
            rounded_total: 25,
            ..FuBreakdown::default()
        };
    }
    if matches!(shape, HandShape::ThirteenOrphans { .. }) {
        return FuBreakdown::default();
    }
    let pinfu = yaku.iter().any(|entry| entry.yaku == Yaku::Pinfu);
    if pinfu && context.method == WinMethod::Tsumo {
        return FuBreakdown {
            base: 20,
            raw_total: 20,
            rounded_total: 20,
            ..FuBreakdown::default()
        };
    }
    let HandShape::Standard { pair, .. } = shape else {
        unreachable!()
    };
    let mut result = FuBreakdown {
        base: 20,
        ..FuBreakdown::default()
    };
    if closed && context.method == WinMethod::Ron {
        result.menzen_ron = 10;
    }
    if context.method == WinMethod::Tsumo {
        result.tsumo = 2;
    }
    result.pair = pair_fu(*pair, context, rules);
    result.wait = if matches!(
        wait,
        WaitType::Kanchan | WaitType::Penchan | WaitType::Tanki
    ) {
        2
    } else {
        0
    };
    result.melds = groups
        .iter()
        .map(|group| group_fu(group, completion, context.method))
        .sum();
    result.raw_total =
        result.base + result.menzen_ron + result.tsumo + result.melds + result.pair + result.wait;
    if result.raw_total == 20 && context.method == WinMethod::Ron {
        result.raw_total = 30;
    }
    result.rounded_total = result.raw_total.div_ceil(10) * 10;
    result
}

fn group_fu(group: &Group, completion: Completion, method: WinMethod) -> u16 {
    let (tile, kan) = match group.kind {
        GroupKind::Sequence(_) => return 0,
        GroupKind::Triplet(tile) => (tile, false),
        GroupKind::Kan(tile) => (tile, true),
    };
    let ron_open = method == WinMethod::Ron
        && matches!((completion, group.concealed_index), (Completion::Meld(a), Some(b)) if a == b);
    let open = group.open || ron_open;
    let mut fu = if kan {
        if open { 8 } else { 16 }
    } else if open {
        2
    } else {
        4
    };
    if tile.is_terminal_or_honor() {
        fu *= 2;
    }
    fu
}

fn pair_fu(pair: TileType, context: &WinContext, rules: &RuleConfig) -> u16 {
    let mut fu = 0;
    if matches!(pair, TileType::White | TileType::Green | TileType::Red) {
        fu += 2;
    }
    if pair == context.seat_wind && pair == context.round_wind {
        fu += if rules.double_wind_pair_fu { 4 } else { 2 };
    } else {
        if pair == context.seat_wind {
            fu += 2;
        }
        if pair == context.round_wind {
            fu += 2;
        }
    }
    fu
}

fn is_value_pair(pair: TileType, context: &WinContext) -> bool {
    matches!(pair, TileType::White | TileType::Green | TileType::Red)
        || pair == context.seat_wind
        || pair == context.round_wind
}

fn group_is_concealed_triplet(group: &Group, completion: Completion, method: WinMethod) -> bool {
    if !matches!(group.kind, GroupKind::Triplet(_) | GroupKind::Kan(_)) || group.open {
        return false;
    }
    !(method == WinMethod::Ron
        && matches!((completion, group.concealed_index), (Completion::Meld(a), Some(b)) if a == b))
}

fn all_tile_counts(concealed: &[Tile], fixed: &[Meld]) -> Result<TileCounts> {
    TileCounts::from_tiles(
        concealed
            .iter()
            .copied()
            .chain(fixed.iter().flat_map(|meld| meld.tiles.iter().copied())),
    )
}

fn validate_physical_tile_counts(concealed: &[Tile], fixed: &[Meld]) -> Result<()> {
    all_tile_counts(concealed, fixed).map(|_| ())
}

fn all_tiles_match(counts: &TileCounts, predicate: impl Fn(TileType) -> bool) -> bool {
    counts
        .iter()
        .all(|(tile, count)| count == 0 || predicate(tile))
}

fn count_indicator_dora(counts: &TileCounts, indicators: &[TileType]) -> u8 {
    indicators
        .iter()
        .map(|indicator| counts[indicator.dora()])
        .sum()
}

fn count_red_dora(concealed: &[Tile], fixed: &[Meld]) -> u8 {
    concealed
        .iter()
        .chain(fixed.iter().flat_map(|meld| meld.tiles.iter()))
        .filter(|tile| tile.red)
        .count() as u8
}

fn duplicate_sequence_pairs(starts: &[TileType]) -> usize {
    let mut frequencies = HashMap::new();
    for start in starts {
        *frequencies.entry(*start).or_insert(0usize) += 1;
    }
    frequencies.values().map(|count| count / 2).sum()
}

fn has_sanshoku_doujun(starts: &[TileType]) -> bool {
    (0..=6).any(|offset| {
        starts.contains(&TileType::ALL[offset])
            && starts.contains(&TileType::ALL[9 + offset])
            && starts.contains(&TileType::ALL[18 + offset])
    })
}

fn has_ittsu(starts: &[TileType]) -> bool {
    [0, 9, 18].into_iter().any(|base| {
        starts.contains(&TileType::ALL[base])
            && starts.contains(&TileType::ALL[base + 3])
            && starts.contains(&TileType::ALL[base + 6])
    })
}

fn has_sanshoku_doukou(triplets: &[TileType]) -> bool {
    (0..9).any(|offset| {
        triplets.contains(&TileType::ALL[offset])
            && triplets.contains(&TileType::ALL[9 + offset])
            && triplets.contains(&TileType::ALL[18 + offset])
    })
}

fn terminal_group_properties(pair: TileType, groups: &[Group]) -> (bool, bool, bool) {
    let pair_ok = pair.is_terminal_or_honor();
    let mut all_ok = pair_ok;
    let mut has_sequence = false;
    let mut has_honor = pair.is_honor();
    for group in groups {
        match group.kind {
            GroupKind::Sequence(start) => {
                has_sequence = true;
                let rank = start.rank().expect("suited sequence");
                all_ok &= rank == 1 || rank == 7;
            }
            GroupKind::Triplet(tile) | GroupKind::Kan(tile) => {
                all_ok &= tile.is_terminal_or_honor();
                has_honor |= tile.is_honor();
            }
        }
    }
    (all_ok, has_sequence, has_honor)
}

/// Some(one suit + whether honors are present), or None for multiple suits.
fn one_suit_only(counts: &TileCounts) -> Option<bool> {
    let suits: Vec<_> = (0..3)
        .filter(|&suit| {
            counts.as_array()[suit * 9..suit * 9 + 9]
                .iter()
                .any(|&c| c > 0)
        })
        .collect();
    if suits.len() != 1 {
        return None;
    }
    Some(counts.as_array()[27..].iter().any(|&count| count > 0))
}

fn is_green_tile(tile: TileType) -> bool {
    matches!(
        tile,
        TileType::S2 | TileType::S3 | TileType::S4 | TileType::S6 | TileType::S8 | TileType::Green
    )
}

fn chuuren(counts: &TileCounts, shape: &HandShape, winning_tile: TileType) -> Option<bool> {
    if !matches!(shape, HandShape::Standard { .. }) {
        return None;
    }
    let has_honors = counts.as_array()[27..].iter().any(|&count| count > 0);
    if has_honors {
        return None;
    }
    let suit = (0..3).find(|&suit| {
        counts.as_array()[suit * 9..suit * 9 + 9]
            .iter()
            .any(|&count| count > 0)
    })?;
    if (0..3).any(|other| {
        other != suit
            && counts.as_array()[other * 9..other * 9 + 9]
                .iter()
                .any(|&count| count > 0)
    }) {
        return None;
    }
    let slice = &counts.as_array()[suit * 9..suit * 9 + 9];
    if slice[0] < 3 || slice[8] < 3 || slice[1..8].iter().any(|&count| count == 0) {
        return None;
    }
    let mut before_win = *counts;
    before_win.remove(winning_tile).ok()?;
    let base = [3, 1, 1, 1, 1, 1, 1, 1, 3];
    Some(before_win.as_array()[suit * 9..suit * 9 + 9] == base)
}

fn better_score(candidate: &ScoreEvaluation, current: &ScoreEvaluation) -> bool {
    candidate.payment.total_received > current.payment.total_received
        || candidate.payment.total_received == current.payment.total_received
            && (candidate.yakuman_multiplier > current.yakuman_multiplier
                || candidate.total_han > current.total_han
                || candidate.total_han == current.total_han
                    && candidate.fu.rounded_total > current.fu.rounded_total)
}
