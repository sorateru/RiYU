use std::collections::BTreeMap;
use std::io::Read;

use anyhow::{Context, Result, bail, ensure};
use bzip2::read::BzDecoder;
use mahjong_core::{MatchLength, Meld, MeldKind, PlayerId, RuleConfig, Tile, TileType};
use quick_xml::{Reader, events::Event};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PhysicalTile {
    pub id: u8,
    pub kind: u8,
    pub red: bool,
}

impl PhysicalTile {
    fn new(id: u8, red_fives: bool) -> Result<Self> {
        ensure!(id < 136, "invalid physical tile {id}");
        Ok(Self {
            id,
            kind: id / 4,
            red: red_fives && matches!(id, 16 | 52 | 88),
        })
    }

    pub fn semantic(self) -> Tile {
        Tile {
            kind: TileType::ALL[self.kind as usize],
            red: self.red,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PhysicalMeld {
    pub kind: &'static str,
    pub tiles: Vec<PhysicalTile>,
    pub called: Option<PhysicalTile>,
    pub from: Option<u8>,
    pub added: Option<PhysicalTile>,
}

impl PhysicalMeld {
    pub fn semantic(&self) -> Result<Meld> {
        let kind = match self.kind {
            "chi" => MeldKind::Chi,
            "pon" => MeldKind::Pon,
            "daiminkan" => MeldKind::Daiminkan,
            "ankan" => MeldKind::Ankan,
            "shouminkan" => MeldKind::Shouminkan,
            other => bail!("unsupported meld {other}"),
        };
        Ok(Meld {
            kind,
            tiles: self.tiles.iter().map(|t| t.semantic()).collect(),
            called_tile: self.called.map(PhysicalTile::semantic),
            from: self.from.map(PlayerId::new).transpose()?,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Start {
    pub hand_number: u8,
    pub honba: u16,
    pub sticks: u16,
    pub dealer: u8,
    pub dice: [u8; 2],
    pub scores: [i32; 4],
    pub hands: [Vec<PhysicalTile>; 4],
    pub dora: PhysicalTile,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreChange {
    pub before: [i32; 4],
    pub delta: [i32; 4],
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Kind {
    Start(Start),
    Draw {
        seat: u8,
        tile: PhysicalTile,
    },
    Discard {
        seat: u8,
        tile: PhysicalTile,
    },
    Call {
        seat: u8,
        raw_code: u32,
        meld: PhysicalMeld,
    },
    Riichi {
        seat: u8,
        step: u8,
        scores: Option<[i32; 4]>,
    },
    Dora {
        tile: PhysicalTile,
    },
    Win {
        seat: u8,
        from: u8,
        tile: PhysicalTile,
        hand: Vec<PhysicalTile>,
        fu: u16,
        points: i32,
        limit_code: u8,
        yaku: Vec<[u16; 2]>,
        yakuman: Vec<u16>,
        dora: Vec<PhysicalTile>,
        ura: Vec<PhysicalTile>,
        responsibility: Option<u8>,
        scores: ScoreChange,
    },
    DrawGame {
        reason: String,
        hands: [Option<Vec<PhysicalTile>>; 4],
        scores: ScoreChange,
    },
    Players {
        ranks: Option<Vec<u8>>,
        ratings: Option<Vec<f64>>,
    },
    Disconnect {
        seat: u8,
    },
    Shuffle {
        seed: String,
    },
    Rules {
        code: u32,
        lobby: Option<u32>,
    },
    MatchStart {
        dealer: u8,
    },
    Other,
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalEvent {
    pub source_ordinal: u32,
    pub round: Option<u16>,
    pub round_seq: Option<u16>,
    pub tag: String,
    pub attrs: BTreeMap<String, String>,
    #[serde(flatten)]
    pub kind: Kind,
}

#[derive(Debug, Serialize)]
pub struct Match {
    pub id: String,
    pub payload_sha256: String,
    pub locator: String,
    pub source_hour: Option<String>,
    pub source_version: Option<String>,
    pub root_attrs: BTreeMap<String, String>,
    pub raw_rule: u32,
    pub player_count: u8,
    pub final_scores: Option<[i32; 4]>,
    pub final_adjustments: Option<[i32; 4]>, // tenths, as recorded by Tenhou
    pub final_ranks: Option<[u8; 4]>,
    pub events: Vec<CanonicalEvent>,
}

impl Match {
    /// Explicitly supported source codes for the first replay profile.
    pub fn rules(&self) -> Option<RuleConfig> {
        if !matches!(self.raw_rule, 169 | 225) {
            return None;
        }
        Some(RuleConfig {
            starting_score: 25_000,
            target_score: 30_000,
            red_fives: [1; 3],
            open_tanyao: true,
            double_wind_pair_fu: true,
            kiriage_mangan: false,
            double_yakuman: false,
            allow_kokushi_ankan_chankan: false,
            match_length: if self.raw_rule & 8 == 0 {
                MatchLength::EastOnly
            } else {
                MatchLength::EastSouth
            },
            ..RuleConfig::default()
        })
    }
}

const MAX_XML_BYTES: usize = 2 * 1024 * 1024;

pub fn decode(locator: String, compressed: &[u8]) -> Result<Match> {
    let mut xml = Vec::new();
    BzDecoder::new(compressed)
        .take(MAX_XML_BYTES as u64 + 1)
        .read_to_end(&mut xml)?;
    ensure!(xml.len() <= MAX_XML_BYTES, "oversized XML payload");
    normalize(locator, &xml)
}

fn attributes(e: &quick_xml::events::BytesStart<'_>) -> Result<BTreeMap<String, String>> {
    e.attributes()
        .map(|entry| {
            let entry = entry?;
            Ok((
                std::str::from_utf8(entry.key.as_ref())?.to_owned(),
                entry.unescape_value()?.into_owned(),
            ))
        })
        .collect()
}

pub(crate) fn normalize(locator: String, xml: &[u8]) -> Result<Match> {
    let mut reader = Reader::from_reader(xml);
    reader.config_mut().trim_text(true);
    let mut elements = Vec::new();
    let mut root_attrs = BTreeMap::new();
    let mut root_open = false;
    let mut root_seen = false;
    loop {
        match reader.read_event()? {
            Event::Start(e) if e.name().as_ref() == b"mjloggm" && !root_seen => {
                root_seen = true;
                root_open = true;
                root_attrs = attributes(&e)?;
            }
            Event::Empty(e) if root_open => {
                elements.push((
                    std::str::from_utf8(e.name().as_ref())?.to_owned(),
                    attributes(&e)?,
                ));
            }
            Event::End(e) if e.name().as_ref() == b"mjloggm" && root_open => root_open = false,
            Event::Eof => break,
            Event::Decl(_) | Event::Comment(_) => {}
            Event::Text(t) if t.as_ref().is_empty() => {}
            other => bail!(
                "unsupported XML structure: {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }
    ensure!(root_seen && !root_open, "incomplete mjloggm root");
    let raw_rule: u32 = get(
        &elements
            .iter()
            .find(|e| e.0 == "GO")
            .context("missing GO")?
            .1,
        "type",
    )?
    .parse()?;
    let red_fives = raw_rule & 2 == 0;
    let player_count = if raw_rule & 16 == 0 { 4 } else { 3 };
    let mut round = None;
    let mut round_seq = 0;
    let mut events = Vec::with_capacity(elements.len());
    let mut final_scores: Option<[i32; 4]> = None;
    let mut final_adjustments: Option<[i32; 4]> = None;
    for (source_ordinal, (tag, attrs)) in elements.into_iter().enumerate() {
        if tag == "INIT" {
            round = Some(round.map_or(0, |i| i + 1));
            round_seq = 0;
        }
        let kind = parse_kind(&tag, &attrs, red_fives)
            .with_context(|| format!("source event {source_ordinal} ({tag})"))?;
        if let Some(owari) = attrs.get("owari") {
            let values: Vec<&str> = owari.split(',').collect();
            if player_count == 4 || values.len() == 8 {
                ensure!(values.len() == 8, "invalid owari length");
                final_scores = Some(
                    (0..4)
                        .map(|i| scaled(values[2 * i], 100))
                        .collect::<Result<Vec<_>>>()?
                        .try_into()
                        .unwrap(),
                );
                final_adjustments = Some(
                    (0..4)
                        .map(|i| scaled(values[2 * i + 1], 10))
                        .collect::<Result<Vec<_>>>()?
                        .try_into()
                        .unwrap(),
                );
            }
        }
        events.push(CanonicalEvent {
            source_ordinal: source_ordinal.try_into()?,
            round,
            round_seq: round.map(|_| round_seq),
            tag,
            attrs,
            kind,
        });
        if round.is_some() {
            round_seq += 1;
        }
    }
    let dealer = events.iter().find_map(|event| match &event.kind {
        Kind::MatchStart { dealer } => Some(*dealer),
        _ => None,
    });
    let final_ranks = (player_count == 4)
        .then_some(final_scores)
        .flatten()
        .zip(dealer)
        .map(|(scores, dealer)| {
            let mut order = [0, 1, 2, 3];
            order.sort_by_key(|&s| (std::cmp::Reverse(scores[s]), (s + 4 - dealer as usize) % 4));
            let mut ranks = [0; 4];
            for (rank, seat) in order.into_iter().enumerate() {
                ranks[seat] = rank as u8 + 1;
            }
            ranks
        });
    let payload_sha256 = digest(xml);
    let id = digest(format!("riyu-match-pilot-v1\0tenhou\0{payload_sha256}").as_bytes())[..32]
        .to_owned();
    let source_hour = locator
        .get(..10)
        .filter(|s| s.bytes().all(|c| c.is_ascii_digit()))
        .map(str::to_owned);
    let source_version = root_attrs.get("ver").cloned();
    Ok(Match {
        id,
        payload_sha256,
        locator,
        source_hour,
        source_version,
        root_attrs,
        raw_rule,
        player_count,
        final_scores,
        final_adjustments,
        final_ranks,
        events,
    })
}

fn get<'a>(attrs: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    attrs
        .get(key)
        .map(String::as_str)
        .with_context(|| format!("missing {key}"))
}

fn list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    s.split(',')
        .filter(|v| !v.is_empty())
        .map(|v| v.parse().map_err(Into::into))
        .collect()
}

fn scaled(s: &str, scale: i32) -> Result<i32> {
    let v: f64 = s.parse()?;
    let v = v * f64::from(scale);
    ensure!(
        v.is_finite() && v.abs() <= f64::from(i32::MAX) && (v - v.round()).abs() < 0.000001,
        "invalid scaled value"
    );
    Ok(v.round() as i32)
}

fn fours(s: &str) -> Result<[i32; 4]> {
    s.split(',')
        .map(|v| scaled(v, 100))
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected four scores"))
}

fn score_change(s: &str) -> Result<ScoreChange> {
    let v: Vec<_> = s
        .split(',')
        .map(|v| scaled(v, 100))
        .collect::<Result<_>>()?;
    ensure!(v.len() == 8, "expected eight sc values");
    Ok(ScoreChange {
        before: std::array::from_fn(|i| v[i * 2]),
        delta: std::array::from_fn(|i| v[i * 2 + 1]),
    })
}

fn parse_kind(tag: &str, a: &BTreeMap<String, String>, red: bool) -> Result<Kind> {
    let tiles = |s: &str| {
        list::<u8>(s)?
            .into_iter()
            .map(|id| PhysicalTile::new(id, red))
            .collect::<Result<Vec<_>>>()
    };
    let attr_tiles = |key: &str| a.get(key).map(|s| tiles(s)).transpose();
    let seat = |key| -> Result<u8> {
        let s: u8 = get(a, key)?.parse()?;
        ensure!(s < 4, "invalid seat");
        Ok(s)
    };
    let tile = |id| PhysicalTile::new(id, red);
    let score = || score_change(get(a, "sc")?);
    if let Some((first, suffix)) = tag.split_at_checked(1) {
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            if let Some(seat) = ["T", "U", "V", "W"].iter().position(|&x| x == first) {
                return Ok(Kind::Draw {
                    seat: seat as u8,
                    tile: tile(suffix.parse()?)?,
                });
            }
            if let Some(seat) = ["D", "E", "F", "G", "d", "e", "f", "g"]
                .iter()
                .position(|&x| x == first)
            {
                return Ok(Kind::Discard {
                    seat: seat as u8 % 4,
                    tile: tile(suffix.parse()?)?,
                });
            }
        }
    }
    Ok(match tag {
        "GO" => Kind::Rules {
            code: get(a, "type")?.parse()?,
            lobby: a.get("lobby").map(|x| x.parse()).transpose()?,
        },
        "SHUFFLE" => Kind::Shuffle {
            seed: get(a, "seed")?.to_owned(),
        },
        "TAIKYOKU" => Kind::MatchStart {
            dealer: seat("oya")?,
        },
        "UN" => Kind::Players {
            ranks: a.get("dan").map(|s| list(s)).transpose()?,
            ratings: a.get("rate").map(|s| list(s)).transpose()?,
        },
        "BYE" => Kind::Disconnect { seat: seat("who")? },
        "INIT" => {
            let s: Vec<u16> = list(get(a, "seed")?)?;
            ensure!(s.len() >= 6, "incomplete INIT seed");
            Kind::Start(Start {
                hand_number: s[0].try_into()?,
                honba: s[1],
                sticks: s[2],
                dealer: seat("oya")?,
                dice: [s[3].try_into()?, s[4].try_into()?],
                scores: fours(get(a, "ten")?)?,
                hands: (0..4)
                    .map(|i| Ok(attr_tiles(&format!("hai{i}"))?.unwrap_or_default()))
                    .collect::<Result<Vec<_>>>()?
                    .try_into()
                    .unwrap(),
                dora: tile(s[5].try_into()?)?,
            })
        }
        "REACH" => {
            let step = get(a, "step")?.parse()?;
            ensure!(matches!(step, 1 | 2), "invalid REACH step");
            Kind::Riichi {
                seat: seat("who")?,
                step,
                scores: a.get("ten").map(|s| fours(s)).transpose()?,
            }
        }
        "N" => {
            let seat = seat("who")?;
            let raw_code = get(a, "m")?.parse()?;
            Kind::Call {
                seat,
                raw_code,
                meld: decode_meld(raw_code, seat, red)?,
            }
        }
        "DORA" => Kind::Dora {
            tile: tile(get(a, "hai")?.parse()?)?,
        },
        "AGARI" => {
            let ten: Vec<u16> = list(get(a, "ten")?)?;
            let yaku: Vec<u16> = list(a.get("yaku").map_or("", String::as_str))?;
            ensure!(
                ten.len() >= 3 && yaku.len() % 2 == 0,
                "invalid AGARI score or yaku"
            );
            Kind::Win {
                seat: seat("who")?,
                from: seat("fromWho")?,
                tile: tile(get(a, "machi")?.parse()?)?,
                hand: tiles(get(a, "hai")?)?,
                fu: ten[0],
                points: ten[1].into(),
                limit_code: ten[2].try_into()?,
                yaku: yaku.chunks_exact(2).map(|p| [p[0], p[1]]).collect(),
                yakuman: list(a.get("yakuman").map_or("", String::as_str))?,
                dora: attr_tiles("doraHai")?.unwrap_or_default(),
                ura: attr_tiles("doraHaiUra")?.unwrap_or_default(),
                responsibility: a.get("paoWho").map(|_| seat("paoWho")).transpose()?,
                scores: score()?,
            }
        }
        "RYUUKYOKU" => Kind::DrawGame {
            reason: a
                .get("type")
                .cloned()
                .unwrap_or_else(|| "exhaustive".into()),
            hands: (0..4)
                .map(|i| attr_tiles(&format!("hai{i}")))
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .unwrap(),
            scores: score()?,
        },
        _ => Kind::Other,
    })
}

fn decode_meld(m: u32, seat: u8, red: bool) -> Result<PhysicalMeld> {
    let relative = (m & 3) as u8;
    let source = (seat + relative) % 4;
    let tile = |id: u32| PhysicalTile::new(id.try_into()?, red);
    let (kind, ids, called_index, from, added) = if m & 4 != 0 {
        let encoded = m >> 10;
        let base = encoded / 3;
        let base = base / 7 * 9 + base % 7;
        let ids = (0..3)
            .map(|i| 4 * (base + i) + ((m >> (3 + i * 2)) & 3))
            .collect();
        ("chi", ids, Some(encoded % 3), Some(source), None)
    } else if m & 0x18 != 0 {
        let unused = (m >> 5) & 3;
        let encoded = m >> 9;
        let base = 4 * (encoded / 3);
        let pon: Vec<_> = (0..4).filter(|&c| c != unused).map(|c| base + c).collect();
        if m & 8 != 0 {
            ("pon", pon, Some(encoded % 3), Some(source), None)
        } else {
            let called = pon[encoded as usize % 3] - base;
            (
                "shouminkan",
                (base..base + 4).collect(),
                Some(called),
                Some(source),
                Some(tile(base + unused)?),
            )
        }
    } else if m & 0x20 != 0 {
        ("kita", vec![m >> 8], None, None, None)
    } else {
        let id = m >> 8;
        let base = id & !3;
        if relative == 0 {
            ("ankan", (base..base + 4).collect(), None, None, None)
        } else {
            (
                "daiminkan",
                (base..base + 4).collect(),
                Some(id % 4),
                Some(source),
                None,
            )
        }
    };
    let tiles = ids.into_iter().map(tile).collect::<Result<Vec<_>>>()?;
    Ok(PhysicalMeld {
        kind,
        called: called_index.map(|i| tiles[i as usize]),
        from,
        added,
        tiles,
    })
}
