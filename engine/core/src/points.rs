use serde::{Deserialize, Serialize};

use crate::{KazoeLimit, MahjongError, Result, RuleConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WinMethod {
    Ron,
    Tsumo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreLimit {
    Normal,
    Mangan,
    Haneman,
    Baiman,
    Sanbaiman,
    KazoeYakuman,
    Yakuman { multiplier: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payment {
    pub han: u8,
    pub fu: u16,
    pub limit: ScoreLimit,
    pub basic_points: u32,
    pub dealer: bool,
    pub method: WinMethod,
    pub ron_payment: Option<u32>,
    pub tsumo_from_dealer: Option<u32>,
    pub tsumo_from_non_dealer: Option<u32>,
    /// Hand payment plus honba, excluding riichi sticks.
    pub total_payment: u32,
    /// Amount received including riichi sticks.
    pub total_received: u32,
}

/// Calculate the standard four-player payment from already established han/fu.
/// Yaku detection is deliberately separate.
#[allow(clippy::too_many_arguments)]
pub fn calculate_points(
    han: u8,
    fu: u16,
    yakuman_multiplier: u8,
    dealer: bool,
    method: WinMethod,
    honba: u16,
    riichi_sticks: u16,
    rules: &RuleConfig,
) -> Result<Payment> {
    rules.validate()?;
    if yakuman_multiplier == 0 && han == 0 {
        return Err(MahjongError::InvalidAction(
            "a winning hand needs at least one han or yakuman".into(),
        ));
    }
    if yakuman_multiplier == 0 && (fu < 20 || fu > 110 || (fu != 25 && fu % 10 != 0)) {
        return Err(MahjongError::InvalidAction(format!(
            "invalid fu value {fu}"
        )));
    }

    let (limit, basic_points) = determine_limit(han, fu, yakuman_multiplier, rules);
    let honba = honba as u32;
    let riichi_award = riichi_sticks as u32 * 1_000;

    let (ron_payment, tsumo_from_dealer, tsumo_from_non_dealer, total_payment) = match method {
        WinMethod::Ron => {
            let base = if dealer {
                round_up_100(basic_points * 6)
            } else {
                round_up_100(basic_points * 4)
            };
            let payment = base + honba * 300;
            (Some(payment), None, None, payment)
        }
        WinMethod::Tsumo if dealer => {
            let each = round_up_100(basic_points * 2) + honba * 100;
            (None, None, Some(each), each * 3)
        }
        WinMethod::Tsumo => {
            let from_dealer = round_up_100(basic_points * 2) + honba * 100;
            let from_non_dealer = round_up_100(basic_points) + honba * 100;
            (
                None,
                Some(from_dealer),
                Some(from_non_dealer),
                from_dealer + from_non_dealer * 2,
            )
        }
    };

    Ok(Payment {
        han,
        fu,
        limit,
        basic_points,
        dealer,
        method,
        ron_payment,
        tsumo_from_dealer,
        tsumo_from_non_dealer,
        total_payment,
        total_received: total_payment + riichi_award,
    })
}

fn determine_limit(
    han: u8,
    fu: u16,
    yakuman_multiplier: u8,
    rules: &RuleConfig,
) -> (ScoreLimit, u32) {
    if yakuman_multiplier > 0 {
        return (
            ScoreLimit::Yakuman {
                multiplier: yakuman_multiplier,
            },
            8_000 * yakuman_multiplier as u32,
        );
    }
    if han >= 13 {
        return match rules.kazoe_limit {
            KazoeLimit::Yakuman => (ScoreLimit::KazoeYakuman, 8_000),
            KazoeLimit::Sanbaiman => (ScoreLimit::Sanbaiman, 6_000),
        };
    }
    if han >= 11 {
        return (ScoreLimit::Sanbaiman, 6_000);
    }
    if han >= 8 {
        return (ScoreLimit::Baiman, 4_000);
    }
    if han >= 6 {
        return (ScoreLimit::Haneman, 3_000);
    }
    let raw = (fu as u32).saturating_mul(1u32.checked_shl((han + 2) as u32).unwrap_or(u32::MAX));
    let mangan = han >= 5
        || raw >= 2_000
        || (rules.kiriage_mangan && ((han == 4 && fu == 30) || (han == 3 && fu == 60)));
    if mangan {
        (ScoreLimit::Mangan, 2_000)
    } else {
        (ScoreLimit::Normal, raw)
    }
}

const fn round_up_100(points: u32) -> u32 {
    points.div_ceil(100) * 100
}
