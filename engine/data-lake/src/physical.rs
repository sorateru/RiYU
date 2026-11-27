use anyhow::{Result, ensure};

use crate::source::{PhysicalMeld, PhysicalTile, Start};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Zone {
    Unseen,
    Hand(u8),
    River(u8),
    Meld(u8),
    Indicator,
}

pub struct Ledger {
    zones: [Zone; 136],
    pub hand: [Vec<PhysicalTile>; 4],
    pub drawn: [Option<PhysicalTile>; 4],
    pub melds: [Vec<PhysicalMeld>; 4],
    pub latest_discard: Option<(u8, PhysicalTile)>,
}

impl Ledger {
    pub fn new(start: &Start) -> Result<Self> {
        let mut ledger = Self {
            zones: [Zone::Unseen; 136],
            hand: start.hands.clone(),
            drawn: [None; 4],
            melds: std::array::from_fn(|_| Vec::new()),
            latest_discard: None,
        };
        for (seat, hand) in start.hands.iter().enumerate() {
            ensure!(hand.len() == 13, "initial hand must have 13 tiles");
            for &tile in hand {
                ledger.move_tile(tile, Zone::Unseen, Zone::Hand(seat as u8))?;
            }
        }
        ledger.dora(start.dora)?;
        Ok(ledger)
    }

    fn move_tile(&mut self, tile: PhysicalTile, from: Zone, to: Zone) -> Result<()> {
        let zone = &mut self.zones[tile.id as usize];
        ensure!(
            *zone == from,
            "tile zone conservation failed: id={}, expected {from:?}, found {zone:?}",
            tile.id
        );
        *zone = to;
        Ok(())
    }

    pub fn dora(&mut self, tile: PhysicalTile) -> Result<()> {
        self.move_tile(tile, Zone::Unseen, Zone::Indicator)
    }

    pub fn draw(&mut self, seat: u8, tile: PhysicalTile) -> Result<()> {
        ensure!(self.drawn[seat as usize].is_none(), "unresolved draw");
        self.move_tile(tile, Zone::Unseen, Zone::Hand(seat))?;
        self.drawn[seat as usize] = Some(tile);
        self.latest_discard = None;
        Ok(())
    }

    pub fn discard(&mut self, seat: u8, tile: PhysicalTile) -> Result<bool> {
        let tsumogiri = self.drawn[seat as usize] == Some(tile);
        self.move_tile(tile, Zone::Hand(seat), Zone::River(seat))?;
        self.integrate_draw(seat);
        self.remove(seat, tile)?;
        self.latest_discard = Some((seat, tile));
        Ok(tsumogiri)
    }

    fn integrate_draw(&mut self, seat: u8) {
        if let Some(tile) = self.drawn[seat as usize].take() {
            self.hand[seat as usize].push(tile);
        }
    }

    fn remove(&mut self, seat: u8, tile: PhysicalTile) -> Result<()> {
        let hand = &mut self.hand[seat as usize];
        let index = hand
            .iter()
            .position(|&t| t == tile)
            .ok_or_else(|| anyhow::anyhow!("physical tile absent from hand"))?;
        hand.swap_remove(index);
        Ok(())
    }

    pub fn call(&mut self, seat: u8, meld: &PhysicalMeld) -> Result<()> {
        self.integrate_draw(seat);
        if meld.kind == "shouminkan" {
            let added = meld
                .added
                .ok_or_else(|| anyhow::anyhow!("missing added tile"))?;
            let prior = self.melds[seat as usize]
                .iter()
                .position(|m| {
                    m.kind == "pon"
                        && m.called == meld.called
                        && m.from == meld.from
                        && m.tiles.iter().all(|t| meld.tiles.contains(t))
                        && !m.tiles.contains(&added)
                })
                .ok_or_else(|| anyhow::anyhow!("physical prior pon not found"))?;
            self.move_tile(added, Zone::Hand(seat), Zone::Meld(seat))?;
            self.remove(seat, added)?;
            self.melds[seat as usize][prior] = meld.clone();
        } else {
            if let Some(called) = meld.called {
                let from = meld
                    .from
                    .ok_or_else(|| anyhow::anyhow!("call lacks source seat"))?;
                ensure!(
                    self.latest_discard == Some((from, called)),
                    "physical called discard mismatch"
                );
                self.move_tile(called, Zone::River(from), Zone::Meld(seat))?;
            }
            for &t in &meld.tiles {
                if Some(t) != meld.called {
                    self.move_tile(t, Zone::Hand(seat), Zone::Meld(seat))?;
                    self.remove(seat, t)?;
                }
            }
            self.melds[seat as usize].push(meld.clone());
        }
        self.latest_discard = None;
        Ok(())
    }

    pub fn cancel_added_kan(&mut self, seat: u8, tile: PhysicalTile) -> Result<()> {
        let meld = self.melds[seat as usize]
            .iter_mut()
            .find(|m| m.added == Some(tile))
            .ok_or_else(|| anyhow::anyhow!("robbed added kan is absent"))?;
        meld.kind = "pon";
        meld.added = None;
        meld.tiles.retain(|&t| t != tile);
        self.move_tile(tile, Zone::Meld(seat), Zone::River(seat))
    }
}
