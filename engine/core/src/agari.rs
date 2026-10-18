use serde::{Deserialize, Serialize};

use crate::{MahjongError, Result, TileCounts, TileType};

const KOKUSHI_INDEXES: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShapeMeld {
    Sequence { start: TileType },
    Triplet { tile: TileType },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HandShape {
    Standard {
        pair: TileType,
        concealed_melds: Vec<ShapeMeld>,
    },
    SevenPairs {
        pairs: Vec<TileType>,
    },
    ThirteenOrphans {
        pair: TileType,
        thirteen_sided_wait: bool,
    },
}

/// Return all distinct concealed decompositions compatible with the fixed melds.
pub fn decompose_winning_hand(
    counts: &TileCounts,
    fixed_melds: u8,
    winning_tile: Option<TileType>,
) -> Result<Vec<HandShape>> {
    counts.validate()?;
    if fixed_melds > 4 {
        return Err(MahjongError::InvalidState(
            "fixed meld count exceeds four".into(),
        ));
    }
    let expected = 14usize.saturating_sub(fixed_melds as usize * 3);
    if counts.total() != expected {
        return Err(MahjongError::InvalidState(format!(
            "winning shape needs {expected} concealed tiles with {fixed_melds} fixed melds, got {}",
            counts.total()
        )));
    }

    let mut shapes = Vec::new();
    if fixed_melds == 0 {
        if let Some(shape) = seven_pairs(counts) {
            shapes.push(shape);
        }
        if let Some(shape) = thirteen_orphans(counts, winning_tile) {
            shapes.push(shape);
        }
    }

    let concealed_meld_count = 4 - fixed_melds as usize;
    for pair_index in 0..34 {
        if counts.as_array()[pair_index] < 2 {
            continue;
        }
        let mut work = counts.into_array();
        work[pair_index] -= 2;
        let mut melds = Vec::with_capacity(concealed_meld_count);
        collect_melds(
            &mut work,
            concealed_meld_count,
            &mut melds,
            &mut shapes,
            TileType::ALL[pair_index],
        );
    }
    shapes.sort_by_key(shape_sort_key);
    shapes.dedup();
    Ok(shapes)
}

pub fn is_winning_hand(counts: &TileCounts, fixed_melds: u8) -> Result<bool> {
    Ok(!decompose_winning_hand(counts, fixed_melds, None)?.is_empty())
}

fn collect_melds(
    counts: &mut [u8; 34],
    needed: usize,
    current: &mut Vec<ShapeMeld>,
    output: &mut Vec<HandShape>,
    pair: TileType,
) {
    let Some(index) = counts.iter().position(|&count| count > 0) else {
        if current.len() == needed {
            output.push(HandShape::Standard {
                pair,
                concealed_melds: current.clone(),
            });
        }
        return;
    };
    if current.len() >= needed {
        return;
    }

    if counts[index] >= 3 {
        counts[index] -= 3;
        current.push(ShapeMeld::Triplet {
            tile: TileType::ALL[index],
        });
        collect_melds(counts, needed, current, output, pair);
        current.pop();
        counts[index] += 3;
    }

    if index < 27 && index % 9 <= 6 && counts[index + 1] > 0 && counts[index + 2] > 0 {
        counts[index] -= 1;
        counts[index + 1] -= 1;
        counts[index + 2] -= 1;
        current.push(ShapeMeld::Sequence {
            start: TileType::ALL[index],
        });
        collect_melds(counts, needed, current, output, pair);
        current.pop();
        counts[index] += 1;
        counts[index + 1] += 1;
        counts[index + 2] += 1;
    }
}

fn seven_pairs(counts: &TileCounts) -> Option<HandShape> {
    let pairs: Vec<_> = counts
        .iter()
        .filter_map(|(tile, count)| (count == 2).then_some(tile))
        .collect();
    (pairs.len() == 7).then_some(HandShape::SevenPairs { pairs })
}

fn thirteen_orphans(counts: &TileCounts, winning_tile: Option<TileType>) -> Option<HandShape> {
    if counts
        .iter()
        .any(|(tile, count)| count > 0 && !tile.is_terminal_or_honor())
    {
        return None;
    }
    if KOKUSHI_INDEXES
        .iter()
        .any(|&index| counts.as_array()[index] == 0)
    {
        return None;
    }
    let pair_index = KOKUSHI_INDEXES
        .iter()
        .copied()
        .find(|&index| counts.as_array()[index] == 2)?;
    let pair = TileType::ALL[pair_index];
    let thirteen_sided_wait = winning_tile == Some(pair);
    Some(HandShape::ThirteenOrphans {
        pair,
        thirteen_sided_wait,
    })
}

fn shape_sort_key(shape: &HandShape) -> (u8, usize, usize) {
    match shape {
        HandShape::Standard {
            pair,
            concealed_melds,
        } => {
            let meld_hash = concealed_melds.iter().fold(0usize, |value, meld| {
                value * 37
                    + match meld {
                        ShapeMeld::Sequence { start } => start.index(),
                        ShapeMeld::Triplet { tile } => tile.index() + 34,
                    }
            });
            (0, pair.index(), meld_hash)
        }
        HandShape::SevenPairs { .. } => (1, 0, 0),
        HandShape::ThirteenOrphans { .. } => (2, 0, 0),
    }
}
