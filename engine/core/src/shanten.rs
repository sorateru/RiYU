use serde::{Deserialize, Serialize};

use crate::{MahjongError, Result, TileCounts};

const KOKUSHI_INDEXES: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShantenResult {
    /// Minimum over every enabled hand family. `-1` means complete.
    pub minimum: i8,
    pub standard: i8,
    pub seven_pairs: Option<i8>,
    pub thirteen_orphans: Option<i8>,
}

/// Calculate exact shanten for a concealed tile count and number of fixed melds.
///
/// `counts` includes the drawn tile when present. Kans count as one fixed meld.
pub fn calculate_shanten(counts: &TileCounts, fixed_melds: u8) -> Result<ShantenResult> {
    counts.validate()?;
    if fixed_melds > 4 {
        return Err(MahjongError::InvalidState(format!(
            "fixed meld count {fixed_melds} exceeds four"
        )));
    }
    let maximum_concealed = 14usize.saturating_sub(fixed_melds as usize * 3);
    if counts.total() > maximum_concealed {
        return Err(MahjongError::InvalidState(format!(
            "{} concealed tiles exceed structural maximum {maximum_concealed} with {fixed_melds} fixed melds",
            counts.total()
        )));
    }

    let standard = standard_shanten(*counts.as_array(), fixed_melds);
    let (seven_pairs, thirteen_orphans) = if fixed_melds == 0 {
        (
            Some(seven_pairs_shanten(counts)),
            Some(thirteen_orphans_shanten(counts)),
        )
    } else {
        (None, None)
    };
    let minimum = seven_pairs
        .into_iter()
        .chain(thirteen_orphans)
        .fold(standard, i8::min);
    Ok(ShantenResult {
        minimum,
        standard,
        seven_pairs,
        thirteen_orphans,
    })
}

fn standard_shanten(mut counts: [u8; 34], fixed_melds: u8) -> i8 {
    let mut best = 8;
    search_standard(&mut counts, 0, fixed_melds as i8, 0, 0, &mut best);
    best
}

fn search_standard(
    counts: &mut [u8; 34],
    start: usize,
    melds: i8,
    pairs: i8,
    taatsu: i8,
    best: &mut i8,
) {
    let Some(index) = (start..34).find(|&index| counts[index] > 0) else {
        let usable_taatsu = taatsu.min((4 - melds).max(0));
        *best = (*best).min(8 - melds * 2 - usable_taatsu - pairs.min(1));
        return;
    };

    // Treat this tile as isolated. This branch is necessary for mixed shapes.
    counts[index] -= 1;
    search_standard(counts, index, melds, pairs, taatsu, best);
    counts[index] += 1;

    if melds < 4 && counts[index] >= 3 {
        counts[index] -= 3;
        search_standard(counts, index, melds + 1, pairs, taatsu, best);
        counts[index] += 3;
    }

    if melds < 4 && index < 27 && index % 9 <= 6 && counts[index + 1] > 0 && counts[index + 2] > 0 {
        counts[index] -= 1;
        counts[index + 1] -= 1;
        counts[index + 2] -= 1;
        search_standard(counts, index, melds + 1, pairs, taatsu, best);
        counts[index] += 1;
        counts[index + 1] += 1;
        counts[index + 2] += 1;
    }

    if counts[index] >= 2 {
        counts[index] -= 2;
        if pairs == 0 {
            search_standard(counts, index, melds, 1, taatsu, best);
        }
        search_standard(counts, index, melds, pairs, taatsu + 1, best);
        counts[index] += 2;
    }

    if index < 27 && index % 9 <= 7 && counts[index + 1] > 0 {
        counts[index] -= 1;
        counts[index + 1] -= 1;
        search_standard(counts, index, melds, pairs, taatsu + 1, best);
        counts[index] += 1;
        counts[index + 1] += 1;
    }

    if index < 27 && index % 9 <= 6 && counts[index + 2] > 0 {
        counts[index] -= 1;
        counts[index + 2] -= 1;
        search_standard(counts, index, melds, pairs, taatsu + 1, best);
        counts[index] += 1;
        counts[index + 2] += 1;
    }
}

fn seven_pairs_shanten(counts: &TileCounts) -> i8 {
    let pair_count = counts
        .as_array()
        .iter()
        .filter(|&&count| count >= 2)
        .count() as i8;
    let unique_count = counts.unique_count() as i8;
    6 - pair_count + (7 - unique_count).max(0)
}

fn thirteen_orphans_shanten(counts: &TileCounts) -> i8 {
    let unique = KOKUSHI_INDEXES
        .iter()
        .filter(|&&index| counts.as_array()[index] > 0)
        .count() as i8;
    let pair = KOKUSHI_INDEXES
        .iter()
        .any(|&index| counts.as_array()[index] >= 2) as i8;
    13 - unique - pair
}
