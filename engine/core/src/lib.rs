//! Authoritative, deterministic state model for four-player riichi mahjong.
//!
//! This crate deliberately contains no neural-network code.  AI components may
//! consume [`GameState`], but only the rule engine is allowed to mutate it.

mod action;
mod agari;
mod analysis;
mod counts;
mod error;
mod event;
mod legal;
mod player;
mod points;
mod rules;
mod scoring;
mod shanten;
mod state;
mod tile;

pub use action::{Action, KanAction, Meld, MeldKind};
pub use agari::{HandShape, ShapeMeld, decompose_winning_hand, is_winning_hand};
pub use analysis::{
    DiscardAnalysis, UkeireTile, analyze_discards, analyze_observer_discards, calculate_ukeire,
};
pub use counts::TileCounts;
pub use error::{MahjongError, Result};
pub use event::{DrawGameReason, EventKind, GameEvent, RiichiStep, RoundEndReason};
pub use legal::{LegalActionReport, legal_actions, winning_tile_types};
pub use player::PlayerId;
pub use points::{Payment, ScoreLimit, WinMethod, calculate_points};
pub use rules::{KazoeLimit, RuleConfig};
pub use scoring::{
    FuBreakdown, RiichiDeclaration, ScoreEvaluation, WaitType, WinContext, Yaku, YakuEntry,
    score_hand,
};
pub use shanten::{ShantenResult, calculate_shanten};
pub use state::{
    CompleteGameState, DiscardRecord, FuritenState, GameState, InformationScope,
    PlayerPublicState, PrivateGameState, RiichiState, RoundState, RoundWind, TurnPhase,
};
pub use tile::{Tile, TileType};

/// Current JSON contract version. Breaking schema changes must increment this.
pub const SCHEMA_VERSION: u16 = 1;

/// Deserialize and validate one observation state.
pub fn game_state_from_json(input: &str) -> Result<GameState> {
    let state: GameState = serde_json::from_str(input)?;
    state.validate()?;
    Ok(state)
}

/// Serialize one observation state using a stable, human-readable layout.
pub fn game_state_to_json(state: &GameState) -> Result<String> {
    state.validate()?;
    Ok(serde_json::to_string_pretty(state)?)
}

/// Rebuild a state from a perspective and an ordered event stream.
pub fn replay_events(
    observer: PlayerId,
    rules: RuleConfig,
    events: impl IntoIterator<Item = GameEvent>,
) -> Result<GameState> {
    let mut state = GameState::new(observer, rules)?;
    for event in events {
        state.apply_event(event)?;
    }
    Ok(state)
}
