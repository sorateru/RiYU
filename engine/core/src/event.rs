use serde::{Deserialize, Serialize};

use crate::{Meld, PlayerId, RoundState, Tile};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiichiStep {
    Declared,
    Accepted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawGameReason {
    Exhaustive,
    NineTerminals,
    FourWinds,
    FourRiichi,
    FourKan,
    TripleRon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundEndReason {
    Ron,
    Tsumo,
    Draw,
    AbortiveDraw,
}

/// Events contain observations, not secrets. For an opponent draw, `tile` must
/// be `null`; the same event from that player's perspective must contain it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    RoundStart {
        round: RoundState,
        scores: [i32; 4],
        initial_hand: Vec<Tile>,
        dora_indicator: Tile,
        remaining_tiles: u16,
    },
    Draw {
        player: PlayerId,
        tile: Option<Tile>,
        #[serde(default)]
        rinshan: bool,
    },
    Discard {
        player: PlayerId,
        tile: Tile,
        tsumogiri: bool,
        #[serde(default)]
        riichi_declaration: bool,
    },
    Chi {
        player: PlayerId,
        meld: Meld,
    },
    Pon {
        player: PlayerId,
        meld: Meld,
    },
    Kan {
        player: PlayerId,
        meld: Meld,
    },
    Riichi {
        player: PlayerId,
        step: RiichiStep,
    },
    DoraReveal {
        tile: Tile,
    },
    /// A response decision that matters for temporary/riichi furiten.
    Pass {
        player: PlayerId,
        ron_available: bool,
    },
    Ron {
        winners: Vec<PlayerId>,
        loser: PlayerId,
        score_deltas: [i32; 4],
    },
    Tsumo {
        winner: PlayerId,
        score_deltas: [i32; 4],
    },
    DrawGame {
        reason: DrawGameReason,
        score_deltas: [i32; 4],
    },
    RoundEnd {
        reason: RoundEndReason,
        scores: [i32; 4],
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameEvent {
    /// Monotonic sequence over the entire match, beginning at zero.
    pub sequence: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}
