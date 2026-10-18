use thiserror::Error;

pub type Result<T> = std::result::Result<T, MahjongError>;

#[derive(Debug, Error)]
pub enum MahjongError {
    #[error("invalid player id {0}; expected 0..=3")]
    InvalidPlayer(u8),

    #[error("invalid tile: {0}")]
    InvalidTile(String),

    #[error("invalid rule configuration: {0}")]
    InvalidRules(String),

    #[error("invalid action: {0}")]
    InvalidAction(String),

    #[error("winning hand has no yaku")]
    NoYaku,

    #[error("invalid event: {0}")]
    InvalidEvent(String),

    #[error("invalid game state: {0}")]
    InvalidState(String),

    #[error("event sequence mismatch: expected {expected}, received {actual}")]
    SequenceMismatch { expected: u64, actual: u64 },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
