use serde::{Deserialize, Serialize};

use crate::{
    Action, DiscardAnalysis, GameState, Result, ShantenResult, TileCounts, UkeireTile,
    analyze_observer_discards, calculate_shanten, calculate_ukeire, legal_actions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricStatus {
    Exact,
    ModelEstimate,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisValue<T> {
    pub status: MetricStatus,
    pub value: Option<T>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelActionScore {
    pub action: Action,
    /// Raw model probability only after calibration; otherwise callers must
    /// label this as a policy score rather than a probability.
    pub policy_score: f32,
}

impl<T> AnalysisValue<T> {
    fn exact(value: T) -> Self {
        Self {
            status: MetricStatus::Exact,
            value: Some(value),
            reason: None,
        }
    }

    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: MetricStatus::Unavailable,
            value: None,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateAnalysis {
    pub schema_version: u16,
    pub shanten: AnalysisValue<ShantenResult>,
    pub ukeire: AnalysisValue<Vec<UkeireTile>>,
    pub legal_actions: AnalysisValue<Vec<Action>>,
    /// Sorted by exact shanten then live ukeire. This is not an AI policy.
    pub discard_efficiency: AnalysisValue<Vec<DiscardAnalysis>>,
    pub model_policy: AnalysisValue<Vec<ModelActionScore>>,
    pub estimated_win_rate: AnalysisValue<f32>,
    pub estimated_deal_in_rate: AnalysisValue<f32>,
    pub estimated_rank_value: AnalysisValue<f32>,
}

pub fn analyze_state(state: &GameState) -> Result<StateAnalysis> {
    state.validate()?;
    let observer = state.observer();
    let fixed_melds = state.players[observer.index()].melds.len() as u8;
    let mut tiles = state.hand.clone();
    tiles.extend(state.drawn_tile);
    let structural_13 = 13usize.saturating_sub(fixed_melds as usize * 3);
    let structural_14 = 14usize.saturating_sub(fixed_melds as usize * 3);

    let (shanten, ukeire) = if tiles.len() == structural_13 || tiles.len() == structural_14 {
        let counts = TileCounts::from_tiles(tiles.iter().copied())?;
        let shanten = AnalysisValue::exact(calculate_shanten(&counts, fixed_melds)?);
        let ukeire = if tiles.len() == structural_13 {
            AnalysisValue::exact(calculate_ukeire(
                &counts,
                &known_tile_counts(state)?,
                fixed_melds,
            )?)
        } else {
            AnalysisValue::unavailable("14枚状態では打牌候補ごとに受け入れを算出します")
        };
        (shanten, ukeire)
    } else {
        (
            AnalysisValue::unavailable("局が未開始か、手牌枚数が解析可能な状態ではありません"),
            AnalysisValue::unavailable("局が未開始か、手牌枚数が解析可能な状態ではありません"),
        )
    };

    let legal = legal_actions(state)?;
    let discard_efficiency = match analyze_observer_discards(state) {
        Ok(candidates) => AnalysisValue::exact(candidates),
        Err(_) => AnalysisValue::unavailable("現在は観測者の打牌手番ではありません"),
    };
    let no_model = "学習済みで評価済みのモデルが読み込まれていません";
    Ok(StateAnalysis {
        schema_version: state.schema_version,
        shanten,
        ukeire,
        legal_actions: AnalysisValue::exact(legal.actions),
        discard_efficiency,
        model_policy: AnalysisValue::unavailable(no_model),
        estimated_win_rate: AnalysisValue::unavailable(no_model),
        estimated_deal_in_rate: AnalysisValue::unavailable(no_model),
        estimated_rank_value: AnalysisValue::unavailable(no_model),
    })
}

fn known_tile_counts(state: &GameState) -> Result<TileCounts> {
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
        tiles.extend(
            player
                .melds
                .iter()
                .flat_map(|meld| meld.tiles.iter().copied()),
        );
    }
    TileCounts::from_tiles(tiles)
}
