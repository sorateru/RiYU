use serde::{Deserialize, Serialize};

use crate::{MahjongError, Result};

/// The 34 base tile kinds. Red fives are represented by [`Tile::red`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TileType {
    M1,
    M2,
    M3,
    M4,
    M5,
    M6,
    M7,
    M8,
    M9,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    East,
    South,
    West,
    North,
    White,
    Green,
    Red,
}

impl TileType {
    pub const ALL: [Self; 34] = [
        Self::M1,
        Self::M2,
        Self::M3,
        Self::M4,
        Self::M5,
        Self::M6,
        Self::M7,
        Self::M8,
        Self::M9,
        Self::P1,
        Self::P2,
        Self::P3,
        Self::P4,
        Self::P5,
        Self::P6,
        Self::P7,
        Self::P8,
        Self::P9,
        Self::S1,
        Self::S2,
        Self::S3,
        Self::S4,
        Self::S5,
        Self::S6,
        Self::S7,
        Self::S8,
        Self::S9,
        Self::East,
        Self::South,
        Self::West,
        Self::North,
        Self::White,
        Self::Green,
        Self::Red,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(index: usize) -> Result<Self> {
        Self::ALL.get(index).copied().ok_or_else(|| {
            MahjongError::InvalidTile(format!("tile index {index} is outside 0..34"))
        })
    }

    pub const fn is_honor(self) -> bool {
        self.index() >= 27
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::M1 | Self::M9 | Self::P1 | Self::P9 | Self::S1 | Self::S9
        )
    }

    pub const fn is_terminal_or_honor(self) -> bool {
        self.is_terminal() || self.is_honor()
    }

    /// Rank within a numbered suit, or `None` for honors.
    pub const fn rank(self) -> Option<u8> {
        let index = self.index();
        if index < 27 {
            Some((index % 9 + 1) as u8)
        } else {
            None
        }
    }

    /// Tile represented by this dora indicator.
    pub const fn dora(self) -> Self {
        match self {
            Self::M1 => Self::M2,
            Self::M2 => Self::M3,
            Self::M3 => Self::M4,
            Self::M4 => Self::M5,
            Self::M5 => Self::M6,
            Self::M6 => Self::M7,
            Self::M7 => Self::M8,
            Self::M8 => Self::M9,
            Self::M9 => Self::M1,
            Self::P1 => Self::P2,
            Self::P2 => Self::P3,
            Self::P3 => Self::P4,
            Self::P4 => Self::P5,
            Self::P5 => Self::P6,
            Self::P6 => Self::P7,
            Self::P7 => Self::P8,
            Self::P8 => Self::P9,
            Self::P9 => Self::P1,
            Self::S1 => Self::S2,
            Self::S2 => Self::S3,
            Self::S3 => Self::S4,
            Self::S4 => Self::S5,
            Self::S5 => Self::S6,
            Self::S6 => Self::S7,
            Self::S7 => Self::S8,
            Self::S8 => Self::S9,
            Self::S9 => Self::S1,
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
            Self::North => Self::East,
            Self::White => Self::Green,
            Self::Green => Self::Red,
            Self::Red => Self::White,
        }
    }

    pub const fn supports_red(self) -> bool {
        matches!(self, Self::M5 | Self::P5 | Self::S5)
    }

    pub const fn suit_index(self) -> Option<usize> {
        match self {
            Self::M1
            | Self::M2
            | Self::M3
            | Self::M4
            | Self::M5
            | Self::M6
            | Self::M7
            | Self::M8
            | Self::M9 => Some(0),
            Self::P1
            | Self::P2
            | Self::P3
            | Self::P4
            | Self::P5
            | Self::P6
            | Self::P7
            | Self::P8
            | Self::P9 => Some(1),
            Self::S1
            | Self::S2
            | Self::S3
            | Self::S4
            | Self::S5
            | Self::S6
            | Self::S7
            | Self::S8
            | Self::S9 => Some(2),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tile {
    pub kind: TileType,
    #[serde(default)]
    pub red: bool,
}

impl Tile {
    pub const fn normal(kind: TileType) -> Self {
        Self { kind, red: false }
    }

    pub fn red(kind: TileType) -> Result<Self> {
        let tile = Self { kind, red: true };
        tile.validate()?;
        Ok(tile)
    }

    pub fn validate(self) -> Result<()> {
        if self.red && !self.kind.supports_red() {
            return Err(MahjongError::InvalidTile(format!(
                "{:?} cannot be marked as a red five",
                self.kind
            )));
        }
        Ok(())
    }
}
