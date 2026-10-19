use serde::{Deserialize, Serialize};

use crate::{MahjongError, Result};

use crate::MatchLength;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KazoeLimit {
    Sanbaiman,
    Yakuman,
}

/// Rule differences are data, never global flags hidden in rule code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfig {
    pub player_count: u8,
    pub starting_score: i32,
    pub target_score: i32,
    /// Number of red fives in manzu, pinzu and souzu respectively.
    pub red_fives: [u8; 3],
    pub open_tanyao: bool,
    /// A pair that is both seat and round wind is worth four fu when true.
    #[serde(default = "default_true")]
    pub double_wind_pair_fu: bool,
    pub kiriage_mangan: bool,
    pub kazoe_limit: KazoeLimit,
    pub allow_double_ron: bool,
    pub triple_ron_is_draw: bool,
    /// Count common double-yakuman variants and daisuushii as two units.
    #[serde(default = "default_true")]
    pub double_yakuman: bool,
    #[serde(default = "default_match_length")]
    pub match_length: MatchLength,
    #[serde(default = "default_true")]
    pub allow_west_extension: bool,
    #[serde(default = "default_true")]
    pub agari_yame: bool,
    #[serde(default = "default_true")]
    pub tenpai_yame: bool,
    #[serde(default = "default_true")]
    pub bankruptcy_ends_match: bool,
    /// Permit kokushi musou to rob a concealed kan.
    #[serde(default = "default_true")]
    pub allow_kokushi_ankan_chankan: bool,
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            player_count: 4,
            starting_score: 25_000,
            target_score: 30_000,
            red_fives: [1, 1, 1],
            open_tanyao: true,
            double_wind_pair_fu: true,
            kiriage_mangan: false,
            kazoe_limit: KazoeLimit::Yakuman,
            allow_double_ron: true,
            triple_ron_is_draw: true,
            double_yakuman: true,
            match_length: MatchLength::EastSouth,
            allow_west_extension: true,
            agari_yame: true,
            tenpai_yame: true,
            bankruptcy_ends_match: true,
            allow_kokushi_ankan_chankan: true,
        }
    }
}

const fn default_true() -> bool {
    true
}

const fn default_match_length() -> MatchLength {
    MatchLength::EastSouth
}

impl RuleConfig {
    pub fn validate(&self) -> Result<()> {
        if self.player_count != 4 {
            return Err(MahjongError::InvalidRules(
                "Phase 1 supports exactly four players".into(),
            ));
        }
        if self.starting_score <= 0 || self.target_score <= 0 {
            return Err(MahjongError::InvalidRules(
                "starting_score and target_score must be positive".into(),
            ));
        }
        if self.red_fives.iter().any(|&count| count > 1) {
            return Err(MahjongError::InvalidRules(
                "Phase 1 supports at most one red five per suit".into(),
            ));
        }
        Ok(())
    }
}
