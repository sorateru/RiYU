use std::ops::{Index, IndexMut};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::{MahjongError, Result, Tile, TileType};

/// Red-insensitive count of the 34 base tile kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCounts([u8; 34]);

impl Default for TileCounts {
    fn default() -> Self {
        Self([0; 34])
    }
}

impl TileCounts {
    pub const fn empty() -> Self {
        Self([0; 34])
    }

    pub fn from_tiles(tiles: impl IntoIterator<Item = Tile>) -> Result<Self> {
        let mut counts = Self::empty();
        for tile in tiles {
            tile.validate()?;
            counts.add(tile.kind)?;
        }
        Ok(counts)
    }

    pub fn from_array(counts: [u8; 34]) -> Result<Self> {
        let value = Self(counts);
        value.validate()?;
        Ok(value)
    }

    pub const fn as_array(&self) -> &[u8; 34] {
        &self.0
    }

    pub const fn into_array(self) -> [u8; 34] {
        self.0
    }

    pub fn total(self) -> usize {
        self.0.iter().map(|&count| count as usize).sum()
    }

    pub fn unique_count(self) -> usize {
        self.0.iter().filter(|&&count| count > 0).count()
    }

    pub fn add(&mut self, tile: TileType) -> Result<()> {
        let count = &mut self.0[tile.index()];
        if *count >= 4 {
            return Err(MahjongError::InvalidTile(format!(
                "cannot add a fifth copy of {tile:?}"
            )));
        }
        *count += 1;
        Ok(())
    }

    pub fn remove(&mut self, tile: TileType) -> Result<()> {
        let count = &mut self.0[tile.index()];
        if *count == 0 {
            return Err(MahjongError::InvalidTile(format!(
                "cannot remove absent tile {tile:?}"
            )));
        }
        *count -= 1;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        for (index, &count) in self.0.iter().enumerate() {
            if count > 4 {
                return Err(MahjongError::InvalidTile(format!(
                    "{} copies of {:?}; maximum is four",
                    count,
                    TileType::ALL[index]
                )));
            }
        }
        Ok(())
    }

    pub fn iter(self) -> impl Iterator<Item = (TileType, u8)> {
        TileType::ALL.into_iter().zip(self.0)
    }
}

impl Index<TileType> for TileCounts {
    type Output = u8;

    fn index(&self, index: TileType) -> &Self::Output {
        &self.0[index.index()]
    }
}

impl IndexMut<TileType> for TileCounts {
    fn index_mut(&mut self, index: TileType) -> &mut Self::Output {
        &mut self.0[index.index()]
    }
}

impl Serialize for TileCounts {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TileCounts {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values = Vec::<u8>::deserialize(deserializer)?;
        let array: [u8; 34] = values.try_into().map_err(|values: Vec<u8>| {
            D::Error::custom(format!(
                "tile count array needs 34 entries, got {}",
                values.len()
            ))
        })?;
        Self::from_array(array).map_err(D::Error::custom)
    }
}
