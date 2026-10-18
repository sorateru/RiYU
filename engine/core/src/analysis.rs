use std::cmp::Ordering;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    GameState, MahjongError, Result, ShantenResult, Tile, TileCounts, TileType, TurnPhase,
    calculate_shanten,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UkeireTile {
    pub tile: TileType,
    /// Copies not present in the supplied known-tile count.
    pub remaining: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscardAnalysis {
    pub discard: Tile,
    pub shanten: ShantenResult,
    pub ukeire: Vec<UkeireTile>,
    pub ukeire_count: u16,
}

/// Exact effective tiles: adding one must reduce minimum shanten.
pub fn calculate_ukeire(
    hand: &TileCounts,
    known: &TileCounts,
    fixed_melds: u8,
) -> Result<Vec<UkeireTile>> {
    let current = calculate_shanten(hand, fixed_melds)?.minimum;
    let mut result = Vec::new();
    for tile in TileType::ALL {
        if hand[tile] >= 4 || known[tile] >= 4 {
            continue;
        }
        let mut with_tile = *hand;
        with_tile.add(tile)?;
        if calculate_shanten(&with_tile, fixed_melds)?.minimum < current {
            result.push(UkeireTile {
                tile,
                remaining: 4 - known[tile],
            });
        }
    }
    Ok(result)
}

/// Rank every physically distinct discard (red and normal five remain distinct).
/// This is strict tile efficiency, not a learned recommendation.
pub fn analyze_discards(
    concealed_tiles: &[Tile],
    known: &TileCounts,
    fixed_melds: u8,
) -> Result<Vec<DiscardAnalysis>> {
    let counts = TileCounts::from_tiles(concealed_tiles.iter().copied())?;
    let expected = 14usize.saturating_sub(fixed_melds as usize * 3);
    if concealed_tiles.len() != expected {
        return Err(MahjongError::InvalidState(format!(
            "discard analysis needs {expected} concealed tiles with {fixed_melds} fixed melds, got {}",
            concealed_tiles.len()
        )));
    }

    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for &discard in concealed_tiles {
        if !seen.insert(discard) {
            continue;
        }
        let mut after = counts;
        after.remove(discard.kind)?;
        let shanten = calculate_shanten(&after, fixed_melds)?;
        let ukeire = calculate_ukeire(&after, known, fixed_melds)?;
        let ukeire_count = ukeire.iter().map(|entry| entry.remaining as u16).sum();
        output.push(DiscardAnalysis {
            discard,
            shanten,
            ukeire,
            ukeire_count,
        });
    }
    output.sort_by(compare_discards);
    Ok(output)
}

pub fn analyze_observer_discards(state: &GameState) -> Result<Vec<DiscardAnalysis>> {
    if !matches!(
        state.phase,
        TurnPhase::AwaitingDiscard { player } if player == state.observer()
    ) {
        return Err(MahjongError::InvalidState(
            "observer does not currently have a discard decision".into(),
        ));
    }
    let mut tiles = state.hand.clone();
    if let Some(tile) = state.drawn_tile {
        tiles.push(tile);
    }
    let fixed_melds = state.players[state.observer().index()].melds.len() as u8;
    analyze_discards(&tiles, &known_tiles(state)?, fixed_melds)
}

fn known_tiles(state: &GameState) -> Result<TileCounts> {
    let mut tiles = state.hand.clone();
    tiles.extend(state.drawn_tile);
    tiles.extend(state.dora_indicators.iter().copied());
    for player in &state.players {
        tiles.extend(
            player
                .discards
                .iter()
                .filter(|discard| discard.called_by.is_none())
                .map(|discard| discard.tile),
        );
        for meld in &player.melds {
            tiles.extend(meld.tiles.iter().copied());
        }
    }
    TileCounts::from_tiles(tiles)
}

fn compare_discards(left: &DiscardAnalysis, right: &DiscardAnalysis) -> Ordering {
    left.shanten
        .minimum
        .cmp(&right.shanten.minimum)
        .then_with(|| right.ukeire_count.cmp(&left.ukeire_count))
        // Prefer keeping a red five when base efficiency ties.
        .then_with(|| left.discard.red.cmp(&right.discard.red))
        .then_with(|| left.discard.kind.index().cmp(&right.discard.kind.index()))
}
