use serde::{Deserialize, Serialize};

use crate::{MahjongError, PlayerId, Result, Tile};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeldKind {
    Chi,
    Pon,
    Daiminkan,
    Ankan,
    Shouminkan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meld {
    pub kind: MeldKind,
    pub tiles: Vec<Tile>,
    /// The tile originally taken from another player's discard.
    pub called_tile: Option<Tile>,
    /// The player whose discard was called. Kept for shouminkan provenance.
    pub from: Option<PlayerId>,
}

impl Meld {
    pub fn validate(&self) -> Result<()> {
        for tile in &self.tiles {
            tile.validate()?;
        }
        if let Some(player) = self.from {
            player.validate()?;
        }

        let expected_len = match self.kind {
            MeldKind::Chi | MeldKind::Pon => 3,
            MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Shouminkan => 4,
        };
        if self.tiles.len() != expected_len {
            return Err(MahjongError::InvalidAction(format!(
                "{:?} requires {expected_len} tiles, got {}",
                self.kind,
                self.tiles.len()
            )));
        }

        match self.kind {
            MeldKind::Chi | MeldKind::Pon | MeldKind::Daiminkan => {
                if self.called_tile.is_none() || self.from.is_none() {
                    return Err(MahjongError::InvalidAction(
                        "open meld requires called_tile and from".into(),
                    ));
                }
            }
            MeldKind::Ankan => {
                if self.called_tile.is_some() || self.from.is_some() {
                    return Err(MahjongError::InvalidAction(
                        "ankan cannot have called_tile or from".into(),
                    ));
                }
            }
            MeldKind::Shouminkan => {
                if self.called_tile.is_none() || self.from.is_none() {
                    return Err(MahjongError::InvalidAction(
                        "shouminkan must retain the original pon provenance".into(),
                    ));
                }
            }
        }

        if let Some(called) = self.called_tile {
            if !self.tiles.contains(&called) {
                return Err(MahjongError::InvalidAction(
                    "called_tile is not present in meld tiles".into(),
                ));
            }
        }

        match self.kind {
            MeldKind::Chi => validate_chi(&self.tiles),
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Shouminkan => {
                let kind = self.tiles[0].kind;
                if self.tiles.iter().any(|tile| tile.kind != kind) {
                    return Err(MahjongError::InvalidAction(
                        "pon/kan tiles must have the same base tile kind".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Tiles supplied by the caller, excluding the one public discard.
    pub fn consumed_from_hand(&self) -> Result<Vec<Tile>> {
        self.validate()?;
        let mut consumed = self.tiles.clone();
        if matches!(
            self.kind,
            MeldKind::Chi | MeldKind::Pon | MeldKind::Daiminkan
        ) {
            let called = self.called_tile.expect("validated open meld");
            let position = consumed
                .iter()
                .position(|tile| *tile == called)
                .ok_or_else(|| {
                    MahjongError::InvalidAction("called tile missing from meld".into())
                })?;
            consumed.remove(position);
        }
        Ok(consumed)
    }
}

fn validate_chi(tiles: &[Tile]) -> Result<()> {
    let mut indexes: Vec<usize> = tiles.iter().map(|tile| tile.kind.index()).collect();
    if indexes.iter().any(|&index| index >= 27) {
        return Err(MahjongError::InvalidAction(
            "chi cannot contain honor tiles".into(),
        ));
    }
    indexes.sort_unstable();
    if indexes[0] / 9 != indexes[2] / 9
        || indexes[1] != indexes[0] + 1
        || indexes[2] != indexes[1] + 1
    {
        return Err(MahjongError::InvalidAction(
            "chi tiles must form one suited sequence".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KanAction {
    Daiminkan {
        called_tile: Tile,
        from: PlayerId,
        consume: [Tile; 3],
    },
    Ankan {
        tiles: [Tile; 4],
    },
    Shouminkan {
        tile: Tile,
        meld_index: usize,
    },
}

/// An AI action describes intent. It never mutates state directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    Discard {
        tile: Tile,
        tsumogiri: bool,
    },
    Riichi {
        discard: Tile,
        tsumogiri: bool,
    },
    Chi {
        called_tile: Tile,
        from: PlayerId,
        consume: [Tile; 2],
    },
    Pon {
        called_tile: Tile,
        from: PlayerId,
        consume: [Tile; 2],
    },
    Kan {
        action: KanAction,
    },
    Ron {
        target: PlayerId,
    },
    Tsumo,
    Pass,
}

impl Action {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Discard { tile, .. } | Self::Riichi { discard: tile, .. } => tile.validate(),
            Self::Chi {
                called_tile,
                from,
                consume,
            } => {
                let meld = Meld {
                    kind: MeldKind::Chi,
                    tiles: vec![*called_tile, consume[0], consume[1]],
                    called_tile: Some(*called_tile),
                    from: Some(*from),
                };
                meld.validate()
            }
            Self::Pon {
                called_tile,
                from,
                consume,
            } => {
                let meld = Meld {
                    kind: MeldKind::Pon,
                    tiles: vec![*called_tile, consume[0], consume[1]],
                    called_tile: Some(*called_tile),
                    from: Some(*from),
                };
                meld.validate()
            }
            Self::Kan { action } => match action {
                KanAction::Daiminkan {
                    called_tile,
                    from,
                    consume,
                } => Meld {
                    kind: MeldKind::Daiminkan,
                    tiles: vec![*called_tile, consume[0], consume[1], consume[2]],
                    called_tile: Some(*called_tile),
                    from: Some(*from),
                }
                .validate(),
                KanAction::Ankan { tiles } => Meld {
                    kind: MeldKind::Ankan,
                    tiles: tiles.to_vec(),
                    called_tile: None,
                    from: None,
                }
                .validate(),
                KanAction::Shouminkan { tile, .. } => tile.validate(),
            },
            Self::Ron { target } => target.validate(),
            Self::Tsumo | Self::Pass => Ok(()),
        }
    }
}
