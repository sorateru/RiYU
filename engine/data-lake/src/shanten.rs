use std::collections::HashMap;

use mahjong_core::{ShantenResult, TileCounts};

// For a single suit, store the maximum number of incomplete groups for each
// (complete groups, used head) combination. Unused tiles are always permitted.
type Frontier = [[i8; 2]; 5];
const EMPTY: Frontier = [[0, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]];

#[derive(Default)]
pub struct SuitMemo {
    cache: HashMap<(bool, u32), Frontier>,
}

impl SuitMemo {
    pub fn analyze(&mut self, counts: &TileCounts, fixed: u8) -> ShantenResult {
        let raw = counts.as_array();
        let mut combined = EMPTY;
        let max = 4 - fixed as usize;
        for (i, suit) in raw.chunks(9).enumerate() {
            let key = suit
                .iter()
                .enumerate()
                .fold(0, |key, (j, &n)| key | (u32::from(n) << (j * 3)));
            let block = self.frontier(i == 3, key);
            let mut next = [[-1; 2]; 5];
            for (a, left) in combined.iter().enumerate().take(max + 1) {
                for (b, right) in block.iter().enumerate().take(max + 1 - a) {
                    for (head_a, &left_groups) in left.iter().enumerate() {
                        for (head_b, &right_groups) in right.iter().enumerate().take(2 - head_a) {
                            if left_groups >= 0 && right_groups >= 0 {
                                next[a + b][head_a + head_b] =
                                    next[a + b][head_a + head_b].max(left_groups + right_groups);
                            }
                        }
                    }
                }
            }
            combined = next;
        }
        let mut best = 8;
        for (m, values) in combined.iter().enumerate().take(max + 1) {
            for (head, &incomplete) in values.iter().enumerate() {
                if incomplete >= 0 {
                    best = best.min(
                        8 - 2 * (m as i8 + fixed as i8)
                            - incomplete.min((max - m) as i8)
                            - head as i8,
                    );
                }
            }
        }
        let (seven_pairs, thirteen_orphans) = if fixed == 0 {
            let pairs = raw.iter().filter(|&&n| n >= 2).count() as i8;
            let unique = counts.unique_count() as i8;
            let terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
            let distinct = terminals.iter().filter(|&&t| raw[t] > 0).count() as i8;
            let pair = terminals.iter().any(|&t| raw[t] >= 2);
            (
                Some(6 - pairs + (7 - unique).max(0)),
                Some(13 - distinct - i8::from(pair)),
            )
        } else {
            (None, None)
        };
        ShantenResult {
            minimum: best
                .min(seven_pairs.unwrap_or(best))
                .min(thirteen_orphans.unwrap_or(best)),
            standard: best,
            seven_pairs,
            thirteen_orphans,
        }
    }

    fn frontier(&mut self, honor: bool, key: u32) -> Frontier {
        if key == 0 {
            return EMPTY;
        }
        if let Some(&f) = self.cache.get(&(honor, key)) {
            return f;
        }
        let i = key.trailing_zeros() / 3;
        let first = 1 << (i * 3);
        let count = (key / first) & 7;
        let has_next = i < 8 && (key / (first << 3)) & 7 != 0;
        let has_skip = i < 7 && (key / (first << 6)) & 7 != 0;
        let mut output = self.frontier(honor, key - first);
        let mut add = |amount: u32, meld: usize, head: usize, incomplete: i8| {
            let smaller = self.frontier(honor, key - amount);
            for (m, values) in smaller.iter().enumerate().take(5 - meld) {
                for (h, &value) in values.iter().enumerate().take(2 - head) {
                    if value >= 0 {
                        output[m + meld][h + head] =
                            output[m + meld][h + head].max(value + incomplete);
                    }
                }
            }
        };
        if count >= 3 {
            add(first * 3, 1, 0, 0);
        }
        if count >= 2 {
            add(first * 2, 0, 1, 0);
            add(first * 2, 0, 0, 1);
        }
        if !honor {
            if has_next && has_skip {
                add(first + (first << 3) + (first << 6), 1, 0, 0);
            }
            if has_next {
                add(first + (first << 3), 0, 0, 1);
            }
            if has_skip {
                add(first + (first << 6), 0, 0, 1);
            }
        }
        if self.cache.len() < 65_536 {
            self.cache.insert((honor, key), output);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_core::{TileType, calculate_shanten};

    #[test]
    fn suit_frontier_matches_exhaustive_search() {
        let mut rng = 37_u64;
        let mut memo = SuitMemo::default();
        for fixed in 0..=4u8 {
            for example in 0..1200 {
                let size = 13 + (example % 2) - usize::from(fixed) * 3;
                let mut counts = TileCounts::empty();
                for _ in 0..size {
                    loop {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let domain = if example % 3 == 0 { 9 } else { 34 };
                        let tile = TileType::ALL[((rng >> 32) as usize) % domain];
                        if counts[tile] < 4 {
                            counts.add(tile).unwrap();
                            break;
                        }
                    }
                }
                assert_eq!(
                    memo.analyze(&counts, fixed),
                    calculate_shanten(&counts, fixed).unwrap(),
                    "counts={counts:?} fixed={fixed}"
                );
            }
        }
    }
}
