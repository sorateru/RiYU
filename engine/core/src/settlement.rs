use serde::{Deserialize, Serialize};

use crate::{
    DrawGameReason, GameState, MahjongError, Payment, PlayerId, Result, RuleConfig, TileType,
    WinMethod, calculate_points,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Responsibility {
    pub liable: PlayerId,
    pub yakuman_multiplier: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RonClaim {
    pub winner: PlayerId,
    pub payment: Payment,
    pub responsibility: Option<Responsibility>,
}

/// Settle ron claims ordered by turn distance from the discarder. The first
/// winner receives the riichi sticks; each claim receives its own honba.
pub fn settle_ron(
    loser: PlayerId,
    claims: &[RonClaim],
    honba: u16,
    riichi_sticks: u16,
    rules: &RuleConfig,
) -> Result<[i32; 4]> {
    loser.validate()?;
    rules.validate()?;
    if claims.is_empty() || claims.len() > 3 {
        return Err(MahjongError::InvalidAction(
            "ron settlement requires one to three claims".into(),
        ));
    }
    if claims.len() > 1 && !rules.allow_double_ron {
        return Err(MahjongError::InvalidAction(
            "multiple ron is disabled".into(),
        ));
    }
    if claims.len() == 3 && rules.triple_ron_is_draw {
        return Err(MahjongError::InvalidAction(
            "triple ron is an abortive draw under these rules".into(),
        ));
    }

    let mut deltas = [0_i32; 4];
    for (index, claim) in claims.iter().enumerate() {
        claim.winner.validate()?;
        if claim.winner == loser || claim.payment.method != WinMethod::Ron {
            return Err(MahjongError::InvalidAction(
                "invalid ron winner or payment method".into(),
            ));
        }
        let total = claim
            .payment
            .ron_payment
            .ok_or_else(|| MahjongError::InvalidAction("ron payment lacks ron amount".into()))?;
        let mut from_loser = total;
        let mut from_liable = 0_u32;
        if let Some(responsibility) = claim.responsibility {
            responsibility.liable.validate()?;
            if responsibility.yakuman_multiplier == 0 {
                return Err(MahjongError::InvalidAction(
                    "responsibility multiplier must be positive".into(),
                ));
            }
            let component = calculate_points(
                0,
                20,
                responsibility.yakuman_multiplier,
                claim.payment.dealer,
                WinMethod::Ron,
                0,
                0,
                rules,
            )?
            .ron_payment
            .expect("ron component");
            let hand_payment = total.saturating_sub(u32::from(honba) * 300);
            if component > hand_payment {
                return Err(MahjongError::InvalidAction(
                    "responsibility exceeds the yakuman value of the hand".into(),
                ));
            }
            if responsibility.liable != loser {
                from_liable = component / 2;
                from_loser -= from_liable;
            }
        }
        transfer(&mut deltas, loser, claim.winner, from_loser)?;
        if let Some(responsibility) = claim.responsibility {
            if from_liable > 0 {
                transfer(
                    &mut deltas,
                    responsibility.liable,
                    claim.winner,
                    from_liable,
                )?;
            }
        }
        if index == 0 {
            deltas[claim.winner.index()] += i32::from(riichi_sticks) * 1_000;
        }
    }
    Ok(deltas)
}

/// Settle tsumo, shifting the responsibility-yakuman portion entirely to the
/// liable player and leaving non-liability and honba payments unchanged.
pub fn settle_tsumo(
    winner: PlayerId,
    dealer: PlayerId,
    payment: &Payment,
    responsibility: Option<Responsibility>,
    riichi_sticks: u16,
    rules: &RuleConfig,
) -> Result<[i32; 4]> {
    winner.validate()?;
    dealer.validate()?;
    rules.validate()?;
    if payment.method != WinMethod::Tsumo {
        return Err(MahjongError::InvalidAction(
            "tsumo settlement requires a tsumo payment".into(),
        ));
    }
    let mut deltas = [0_i32; 4];
    for payer in PlayerId::ALL {
        if payer == winner {
            continue;
        }
        let amount = if payment.dealer || payer != dealer {
            payment.tsumo_from_non_dealer
        } else {
            payment.tsumo_from_dealer
        }
        .ok_or_else(|| MahjongError::InvalidAction("tsumo payment is incomplete".into()))?;
        transfer(&mut deltas, payer, winner, amount)?;
    }

    if let Some(responsibility) = responsibility {
        responsibility.liable.validate()?;
        if responsibility.liable == winner || responsibility.yakuman_multiplier == 0 {
            return Err(MahjongError::InvalidAction(
                "invalid tsumo responsibility".into(),
            ));
        }
        let component = calculate_points(
            0,
            20,
            responsibility.yakuman_multiplier,
            payment.dealer,
            WinMethod::Tsumo,
            0,
            0,
            rules,
        )?;
        let mut shifted = 0_u32;
        for payer in PlayerId::ALL {
            if payer == winner {
                continue;
            }
            let amount = if component.dealer || payer != dealer {
                component.tsumo_from_non_dealer
            } else {
                component.tsumo_from_dealer
            }
            .expect("complete component");
            refund(&mut deltas, payer, winner, amount)?;
            shifted += amount;
        }
        transfer(&mut deltas, responsibility.liable, winner, shifted)?;
    }
    deltas[winner.index()] += i32::from(riichi_sticks) * 1_000;
    Ok(deltas)
}

/// Detect abortive draws that follow entirely from public state. Nine
/// terminals and triple ron require an explicit decision/result event.
pub fn detect_automatic_abortive_draw(state: &GameState) -> Option<DrawGameReason> {
    if state
        .players
        .iter()
        .all(|player| player.riichi.is_accepted())
    {
        return Some(DrawGameReason::FourRiichi);
    }

    let first_discards: Option<Vec<_>> = state
        .players
        .iter()
        .map(|player| {
            (player.discards.len() == 1 && player.melds.is_empty())
                .then(|| player.discards[0].tile.kind)
        })
        .collect();
    if let Some(discards) = first_discards {
        if discards.iter().all(|kind| {
            *kind == discards[0]
                && matches!(
                    kind,
                    TileType::East | TileType::South | TileType::West | TileType::North
                )
        }) {
            return Some(DrawGameReason::FourWinds);
        }
    }

    let mut kans_by_player = [0_u8; 4];
    for (index, player) in state.players.iter().enumerate() {
        kans_by_player[index] = player
            .melds
            .iter()
            .filter(|meld| {
                matches!(
                    meld.kind,
                    crate::MeldKind::Daiminkan
                        | crate::MeldKind::Ankan
                        | crate::MeldKind::Shouminkan
                )
            })
            .count() as u8;
    }
    if let Some(pending) = &state.pending_kan {
        kans_by_player[pending.player.index()] += 1;
    }
    let total: u8 = kans_by_player.iter().sum();
    let declarers = kans_by_player.iter().filter(|&&count| count > 0).count();
    (total >= 4 && declarers >= 2).then_some(DrawGameReason::FourKan)
}

fn transfer(deltas: &mut [i32; 4], from: PlayerId, to: PlayerId, amount: u32) -> Result<()> {
    let amount = i32::try_from(amount)
        .map_err(|_| MahjongError::InvalidAction("payment exceeds i32 range".into()))?;
    deltas[from.index()] -= amount;
    deltas[to.index()] += amount;
    Ok(())
}

fn refund(deltas: &mut [i32; 4], payer: PlayerId, winner: PlayerId, amount: u32) -> Result<()> {
    let amount = i32::try_from(amount)
        .map_err(|_| MahjongError::InvalidAction("payment exceeds i32 range".into()))?;
    deltas[payer.index()] += amount;
    deltas[winner.index()] -= amount;
    Ok(())
}
