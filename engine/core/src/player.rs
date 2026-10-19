use serde::{Deserialize, Serialize};

use crate::{MahjongError, Result};

/// Absolute seat index in table order: 0, 1, 2, 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlayerId(u8);

impl PlayerId {
    pub const ALL: [Self; 4] = [Self(0), Self(1), Self(2), Self(3)];

    pub fn new(value: u8) -> Result<Self> {
        if value < 4 {
            Ok(Self(value))
        } else {
            Err(MahjongError::InvalidPlayer(value))
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn next(self) -> Self {
        Self((self.0 + 1) % 4)
    }

    pub fn validate(self) -> Result<()> {
        Self::new(self.0).map(|_| ())
    }
}
