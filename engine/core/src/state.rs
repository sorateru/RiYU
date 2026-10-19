use std::array;

use serde::{Deserialize, Serialize};

use crate::{
    EventKind, GameEvent, MahjongError, Meld, MeldKind, PlayerId, Result, RiichiStep,
    RoundEndReason, RuleConfig, SCHEMA_VERSION, Tile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundWind {
    East,
    South,
    West,
    North,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundState {
    pub wind: RoundWind,
    /// Kyoku number within the wind, in 1..=4.
    pub hand_number: u8,
    pub honba: u16,
    pub riichi_sticks: u16,
    pub dealer: PlayerId,
}

impl RoundState {
    pub fn validate(&self) -> Result<()> {
        if !(1..=4).contains(&self.hand_number) {
            return Err(MahjongError::InvalidState(format!(
                "hand_number must be in 1..=4, got {}",
                self.hand_number
            )));
        }
        self.dealer.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiichiState {
    None,
    Declared,
    DoubleDeclared,
    Accepted,
    DoubleAccepted,
}

impl RiichiState {
    pub const fn is_declared(self) -> bool {
        matches!(self, Self::Declared | Self::DoubleDeclared)
    }

    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted | Self::DoubleAccepted)
    }

    pub const fn is_double(self) -> bool {
        matches!(self, Self::DoubleDeclared | Self::DoubleAccepted)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FuritenState {
    /// Cleared on the observer's next draw unless riichi furiten applies.
    pub temporary: bool,
    /// Persists through the round after passing a legal ron while in riichi.
    pub riichi: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscardRecord {
    pub tile: Tile,
    pub tsumogiri: bool,
    pub riichi_declaration: bool,
    /// Called discards remain in the river history but are not counted twice.
    pub called_by: Option<PlayerId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerPublicState {
    pub discards: Vec<DiscardRecord>,
    pub melds: Vec<Meld>,
    pub riichi: RiichiState,
}

impl Default for PlayerPublicState {
    fn default() -> Self {
        Self {
            discards: Vec::new(),
            melds: Vec::new(),
            riichi: RiichiState::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum TurnPhase {
    NotStarted,
    AwaitingDraw {
        player: PlayerId,
    },
    AwaitingRinshanDraw {
        player: PlayerId,
    },
    AwaitingDiscard {
        player: PlayerId,
    },
    AwaitingResponse {
        discarder: PlayerId,
        tile: Tile,
    },
    AwaitingKanResponse {
        declarer: PlayerId,
        tile: Tile,
        ankan: bool,
    },
    AwaitingRoundEnd {
        reason: RoundEndReason,
    },
    Ended,
}

/// A concealed/added kan remains provisional until the chankan response
/// window closes.  `replace_meld_index` is set only for shouminkan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingKan {
    pub player: PlayerId,
    pub meld: Meld,
    pub replace_meld_index: Option<usize>,
}

/// Explicit declaration of what this state may know.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InformationScope {
    pub observer: PlayerId,
    /// Always true for a player observation; documented rather than inferred.
    pub observer_hand_visible: bool,
    /// Must remain false for normal inference input.
    pub opponent_hands_visible: bool,
    /// Must remain false for normal inference input.
    pub wall_order_visible: bool,
}

impl InformationScope {
    fn player_observation(observer: PlayerId) -> Self {
        Self {
            observer,
            observer_hand_visible: true,
            opponent_hands_visible: false,
            wall_order_visible: false,
        }
    }

    fn validate(&self) -> Result<()> {
        self.observer.validate()?;
        if !self.observer_hand_visible || self.opponent_hands_visible || self.wall_order_visible {
            return Err(MahjongError::InvalidState(
                "GameState must contain only a player's observable information".into(),
            ));
        }
        Ok(())
    }
}

/// The sole input state accepted by normal inference.
///
/// It intentionally has no fields for opponent concealed hands or wall order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameState {
    pub schema_version: u16,
    pub rules: RuleConfig,
    pub information: InformationScope,
    pub round: Option<RoundState>,
    pub scores: [i32; 4],
    /// Observer's concealed tiles, excluding `drawn_tile`.
    pub hand: Vec<Tile>,
    pub drawn_tile: Option<Tile>,
    pub dora_indicators: Vec<Tile>,
    pub players: [PlayerPublicState; 4],
    /// Public eligibility for ippatsu after an accepted riichi.
    #[serde(default)]
    pub ippatsu_eligible: [bool; 4],
    /// Non-river furiten belongs to the observer and is not public opponent data.
    #[serde(default)]
    pub observer_furiten: FuritenState,
    pub remaining_tiles: u16,
    /// Number of draw events since the current round began.
    pub turn: u16,
    pub current_actor: Option<PlayerId>,
    pub phase: TurnPhase,
    #[serde(default)]
    pub pending_kan: Option<PendingKan>,
    pub last_event: Option<GameEvent>,
    pub event_history: Vec<GameEvent>,
    pub next_event_sequence: u64,
}

impl GameState {
    pub fn new(observer: PlayerId, rules: RuleConfig) -> Result<Self> {
        observer.validate()?;
        rules.validate()?;
        let starting_score = rules.starting_score;
        let state = Self {
            schema_version: SCHEMA_VERSION,
            rules,
            information: InformationScope::player_observation(observer),
            round: None,
            scores: [starting_score; 4],
            hand: Vec::new(),
            drawn_tile: None,
            dora_indicators: Vec::new(),
            players: array::from_fn(|_| PlayerPublicState::default()),
            ippatsu_eligible: [false; 4],
            observer_furiten: FuritenState::default(),
            remaining_tiles: 0,
            turn: 0,
            current_actor: None,
            phase: TurnPhase::NotStarted,
            pending_kan: None,
            last_event: None,
            event_history: Vec::new(),
            next_event_sequence: 0,
        };
        state.validate()?;
        Ok(state)
    }

    pub const fn observer(&self) -> PlayerId {
        self.information.observer
    }

    /// Atomically apply one event. On failure `self` is unchanged.
    pub fn apply_event(&mut self, event: GameEvent) -> Result<()> {
        if event.sequence != self.next_event_sequence {
            return Err(MahjongError::SequenceMismatch {
                expected: self.next_event_sequence,
                actual: event.sequence,
            });
        }
        let mut candidate = self.clone();
        candidate.apply_event_inner(&event.kind)?;
        candidate.next_event_sequence += 1;
        candidate.last_event = Some(event.clone());
        candidate.event_history.push(event);
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }

    fn apply_event_inner(&mut self, event: &EventKind) -> Result<()> {
        match event {
            EventKind::RoundStart {
                round,
                scores,
                initial_hand,
                dora_indicator,
                remaining_tiles,
            } => self.apply_round_start(
                round.clone(),
                *scores,
                initial_hand.clone(),
                *dora_indicator,
                *remaining_tiles,
            ),
            EventKind::Draw {
                player,
                tile,
                rinshan,
            } => self.apply_draw(*player, *tile, *rinshan),
            EventKind::Discard {
                player,
                tile,
                tsumogiri,
                riichi_declaration,
            } => self.apply_discard(*player, *tile, *tsumogiri, *riichi_declaration),
            EventKind::Chi { player, meld } => {
                self.apply_open_meld(*player, meld.clone(), MeldKind::Chi)
            }
            EventKind::Pon { player, meld } => {
                self.apply_open_meld(*player, meld.clone(), MeldKind::Pon)
            }
            EventKind::Kan { player, meld } => self.apply_kan(*player, meld.clone()),
            EventKind::Riichi { player, step } => self.apply_riichi(*player, *step),
            EventKind::DoraReveal { tile } => {
                self.require_active_round()?;
                tile.validate()?;
                self.dora_indicators.push(*tile);
                Ok(())
            }
            EventKind::Pass {
                player,
                ron_available,
            } => self.apply_pass(*player, *ron_available),
            EventKind::Ron {
                winners,
                loser,
                score_deltas,
            } => self.apply_ron(winners, *loser, *score_deltas),
            EventKind::Tsumo {
                winner,
                score_deltas,
            } => self.apply_tsumo(*winner, *score_deltas),
            EventKind::DrawGame {
                reason,
                score_deltas,
            } => {
                self.require_active_round()?;
                self.add_score_deltas(*score_deltas);
                self.phase = TurnPhase::AwaitingRoundEnd {
                    reason: if *reason == crate::DrawGameReason::Exhaustive {
                        RoundEndReason::Draw
                    } else {
                        RoundEndReason::AbortiveDraw
                    },
                };
                self.current_actor = None;
                Ok(())
            }
            EventKind::RoundEnd { reason, scores } => {
                let TurnPhase::AwaitingRoundEnd { reason: actual } = self.phase else {
                    return Err(MahjongError::InvalidEvent(
                        "round_end requires a preceding result event".into(),
                    ));
                };
                if *reason != actual {
                    return Err(MahjongError::InvalidEvent(format!(
                        "round_end reason {:?} does not match {:?}",
                        reason, actual
                    )));
                }
                if *scores != self.scores {
                    return Err(MahjongError::InvalidEvent(
                        "round_end scores do not match applied score deltas".into(),
                    ));
                }
                self.phase = TurnPhase::Ended;
                self.current_actor = None;
                Ok(())
            }
        }
    }

    fn apply_round_start(
        &mut self,
        round: RoundState,
        scores: [i32; 4],
        initial_hand: Vec<Tile>,
        dora_indicator: Tile,
        remaining_tiles: u16,
    ) -> Result<()> {
        if !matches!(self.phase, TurnPhase::NotStarted | TurnPhase::Ended) {
            return Err(MahjongError::InvalidEvent(
                "round_start cannot occur during an active round".into(),
            ));
        }
        round.validate()?;
        dora_indicator.validate()?;
        if initial_hand.len() != 13 {
            return Err(MahjongError::InvalidEvent(format!(
                "initial observer hand must contain 13 tiles, got {}",
                initial_hand.len()
            )));
        }
        for tile in &initial_hand {
            tile.validate()?;
        }
        if remaining_tiles > 70 {
            return Err(MahjongError::InvalidEvent(format!(
                "remaining live-wall tiles cannot exceed 70, got {remaining_tiles}"
            )));
        }

        let dealer = round.dealer;
        self.round = Some(round);
        self.scores = scores;
        self.hand = initial_hand;
        self.drawn_tile = None;
        self.dora_indicators = vec![dora_indicator];
        self.players = array::from_fn(|_| PlayerPublicState::default());
        self.ippatsu_eligible = [false; 4];
        self.observer_furiten = FuritenState::default();
        self.remaining_tiles = remaining_tiles;
        self.turn = 0;
        self.current_actor = Some(dealer);
        self.phase = TurnPhase::AwaitingDraw { player: dealer };
        self.pending_kan = None;
        Ok(())
    }

    fn apply_draw(&mut self, player: PlayerId, tile: Option<Tile>, rinshan: bool) -> Result<()> {
        self.require_active_round()?;
        player.validate()?;
        let closes_pending_kan = matches!(
            self.phase,
            TurnPhase::AwaitingKanResponse { declarer, .. } if declarer == player && rinshan
        );
        match self.phase {
            TurnPhase::AwaitingDraw { player: expected } if expected == player && !rinshan => {}
            TurnPhase::AwaitingRinshanDraw { player: expected }
                if expected == player && rinshan => {}
            TurnPhase::AwaitingResponse { discarder, .. }
                if discarder.next() == player && !rinshan => {}
            TurnPhase::AwaitingKanResponse { declarer, .. } if declarer == player && rinshan => {}
            _ => {
                return Err(MahjongError::InvalidEvent(format!(
                    "player {} cannot draw in phase {:?}",
                    player.get(),
                    self.phase
                )));
            }
        }
        if closes_pending_kan {
            self.finalize_pending_kan()?;
        }
        if self.remaining_tiles == 0 {
            return Err(MahjongError::InvalidEvent(
                "cannot draw from an empty live wall".into(),
            ));
        }
        if player == self.observer() {
            let tile = tile.ok_or_else(|| {
                MahjongError::InvalidEvent("observer draw must include the tile".into())
            })?;
            tile.validate()?;
            if self.drawn_tile.is_some() {
                return Err(MahjongError::InvalidState(
                    "observer already has an unresolved drawn tile".into(),
                ));
            }
            self.drawn_tile = Some(tile);
            if !self.observer_furiten.riichi {
                self.observer_furiten.temporary = false;
            }
        } else if tile.is_some() {
            return Err(MahjongError::InvalidEvent(
                "opponent draw must hide the tile in GameState events".into(),
            ));
        }
        self.remaining_tiles -= 1;
        self.turn += 1;
        self.current_actor = Some(player);
        self.phase = TurnPhase::AwaitingDiscard { player };
        Ok(())
    }

    fn apply_discard(
        &mut self,
        player: PlayerId,
        tile: Tile,
        tsumogiri: bool,
        riichi_declaration: bool,
    ) -> Result<()> {
        self.require_discard_turn(player)?;
        tile.validate()?;
        if riichi_declaration && !self.players[player.index()].riichi.is_declared() {
            return Err(MahjongError::InvalidEvent(
                "riichi discard requires a preceding riichi declaration".into(),
            ));
        }
        if player == self.observer() {
            self.remove_observer_discard(tile, tsumogiri)?;
        }
        self.players[player.index()].discards.push(DiscardRecord {
            tile,
            tsumogiri,
            riichi_declaration,
            called_by: None,
        });
        if self.players[player.index()].riichi.is_accepted() && !riichi_declaration {
            self.ippatsu_eligible[player.index()] = false;
        }
        self.current_actor = None;
        self.phase = TurnPhase::AwaitingResponse {
            discarder: player,
            tile,
        };
        Ok(())
    }

    fn apply_open_meld(
        &mut self,
        player: PlayerId,
        meld: Meld,
        expected_kind: MeldKind,
    ) -> Result<()> {
        player.validate()?;
        meld.validate()?;
        if meld.kind != expected_kind {
            return Err(MahjongError::InvalidEvent(format!(
                "event requires {:?} meld, got {:?}",
                expected_kind, meld.kind
            )));
        }
        let (discarder, tile) = self.response_context()?;
        if meld.from != Some(discarder) || meld.called_tile != Some(tile) {
            return Err(MahjongError::InvalidEvent(
                "meld source does not match the latest discard".into(),
            ));
        }
        if player == discarder {
            return Err(MahjongError::InvalidEvent(
                "a player cannot call their own discard".into(),
            ));
        }
        if expected_kind == MeldKind::Chi && discarder.next() != player {
            return Err(MahjongError::InvalidEvent(
                "chi is only legal for the next player".into(),
            ));
        }
        if player == self.observer() {
            for consumed in meld.consumed_from_hand()? {
                self.remove_observer_tile(consumed)?;
            }
            self.merge_drawn_tile_into_hand();
        }
        self.mark_latest_discard_called(discarder, player)?;
        self.players[player.index()].melds.push(meld);
        self.cancel_all_ippatsu();
        self.current_actor = Some(player);
        self.phase = TurnPhase::AwaitingDiscard { player };
        Ok(())
    }

    fn apply_kan(&mut self, player: PlayerId, meld: Meld) -> Result<()> {
        player.validate()?;
        meld.validate()?;
        let mut pending_kan = None;
        match meld.kind {
            MeldKind::Daiminkan => {
                let (discarder, tile) = self.response_context()?;
                if meld.from != Some(discarder) || meld.called_tile != Some(tile) {
                    return Err(MahjongError::InvalidEvent(
                        "daiminkan source does not match the latest discard".into(),
                    ));
                }
                if player == self.observer() {
                    for consumed in meld.consumed_from_hand()? {
                        self.remove_observer_tile(consumed)?;
                    }
                }
                self.mark_latest_discard_called(discarder, player)?;
                self.players[player.index()].melds.push(meld);
            }
            MeldKind::Ankan => {
                self.require_discard_turn(player)?;
                if player == self.observer() {
                    for tile in &meld.tiles {
                        self.remove_observer_tile(*tile)?;
                    }
                }
                let robbed_tile = meld.tiles[0];
                pending_kan = Some((meld, None, robbed_tile, true));
            }
            MeldKind::Shouminkan => {
                self.require_discard_turn(player)?;
                let called = meld.called_tile.ok_or_else(|| {
                    MahjongError::InvalidEvent("shouminkan lacks called tile".into())
                })?;
                let position = self.players[player.index()]
                    .melds
                    .iter()
                    .position(|existing| {
                        existing.kind == MeldKind::Pon
                            && existing.called_tile == Some(called)
                            && existing.from == meld.from
                    })
                    .ok_or_else(|| {
                        MahjongError::InvalidEvent(
                            "shouminkan requires an existing matching pon".into(),
                        )
                    })?;
                if player == self.observer() {
                    let added = *meld
                        .tiles
                        .iter()
                        .find(|tile| {
                            self.players[player.index()].melds[position]
                                .tiles
                                .iter()
                                .filter(|existing| *existing == *tile)
                                .count()
                                < meld
                                    .tiles
                                    .iter()
                                    .filter(|candidate| *candidate == *tile)
                                    .count()
                        })
                        .ok_or_else(|| {
                            MahjongError::InvalidEvent("cannot identify added kan tile".into())
                        })?;
                    self.remove_observer_tile(added)?;
                }
                let robbed_tile = *meld
                    .tiles
                    .iter()
                    .find(|tile| {
                        self.players[player.index()].melds[position]
                            .tiles
                            .iter()
                            .filter(|existing| *existing == *tile)
                            .count()
                            < meld
                                .tiles
                                .iter()
                                .filter(|candidate| *candidate == *tile)
                                .count()
                    })
                    .ok_or_else(|| {
                        MahjongError::InvalidEvent("cannot identify added kan tile".into())
                    })?;
                pending_kan = Some((meld, Some(position), robbed_tile, false));
            }
            MeldKind::Chi | MeldKind::Pon => {
                return Err(MahjongError::InvalidEvent(
                    "kan event requires a kan meld".into(),
                ));
            }
        }
        self.cancel_all_ippatsu();
        self.merge_drawn_tile_into_hand();
        if let Some((meld, replace_meld_index, tile, ankan)) = pending_kan {
            self.pending_kan = Some(PendingKan {
                player,
                meld,
                replace_meld_index,
            });
            self.current_actor = None;
            self.phase = TurnPhase::AwaitingKanResponse {
                declarer: player,
                tile,
                ankan,
            };
        } else {
            self.current_actor = Some(player);
            self.phase = TurnPhase::AwaitingRinshanDraw { player };
        }
        Ok(())
    }

    fn finalize_pending_kan(&mut self) -> Result<()> {
        let pending = self.pending_kan.take().ok_or_else(|| {
            MahjongError::InvalidState("kan response phase lacks pending kan".into())
        })?;
        let melds = &mut self.players[pending.player.index()].melds;
        if let Some(position) = pending.replace_meld_index {
            let slot = melds.get_mut(position).ok_or_else(|| {
                MahjongError::InvalidState("pending shouminkan pon no longer exists".into())
            })?;
            if slot.kind != MeldKind::Pon {
                return Err(MahjongError::InvalidState(
                    "pending shouminkan must replace a pon".into(),
                ));
            }
            *slot = pending.meld;
        } else {
            melds.push(pending.meld);
        }
        Ok(())
    }

    fn cancel_pending_kan_after_ron(&mut self, robbed_tile: Tile) -> Result<()> {
        let Some(pending) = self.pending_kan.take() else {
            return Ok(());
        };
        if pending.replace_meld_index.is_none() && pending.player == self.observer() {
            let mut skipped_robbed = false;
            for tile in pending.meld.tiles {
                if !skipped_robbed && tile == robbed_tile {
                    skipped_robbed = true;
                } else {
                    self.hand.push(tile);
                }
            }
            if !skipped_robbed {
                return Err(MahjongError::InvalidState(
                    "robbed tile is absent from pending concealed kan".into(),
                ));
            }
        }
        Ok(())
    }

    fn apply_riichi(&mut self, player: PlayerId, step: RiichiStep) -> Result<()> {
        player.validate()?;
        match step {
            RiichiStep::Declared => {
                self.require_discard_turn(player)?;
                if self.players[player.index()].riichi != RiichiState::None {
                    return Err(MahjongError::InvalidEvent(
                        "riichi was already declared".into(),
                    ));
                }
                let double = self.players[player.index()].discards.is_empty()
                    && self.players.iter().all(|state| state.melds.is_empty());
                self.players[player.index()].riichi = if double {
                    RiichiState::DoubleDeclared
                } else {
                    RiichiState::Declared
                };
            }
            RiichiStep::Accepted => {
                if !matches!(self.phase, TurnPhase::AwaitingResponse { discarder, .. } if discarder == player)
                    || !self.players[player.index()].riichi.is_declared()
                {
                    return Err(MahjongError::InvalidEvent(
                        "riichi acceptance requires that player's declaration discard".into(),
                    ));
                }
                if self.scores[player.index()] < 1_000 {
                    return Err(MahjongError::InvalidEvent(
                        "player lacks 1000 points for riichi".into(),
                    ));
                }
                self.scores[player.index()] -= 1_000;
                self.round.as_mut().expect("active round").riichi_sticks += 1;
                self.players[player.index()].riichi =
                    if self.players[player.index()].riichi.is_double() {
                        RiichiState::DoubleAccepted
                    } else {
                        RiichiState::Accepted
                    };
                self.ippatsu_eligible[player.index()] = true;
            }
        }
        Ok(())
    }

    fn apply_ron(
        &mut self,
        winners: &[PlayerId],
        loser: PlayerId,
        score_deltas: [i32; 4],
    ) -> Result<()> {
        let (discarder, robbed_tile, chankan) = self.ron_response_context()?;
        loser.validate()?;
        if loser != discarder || winners.is_empty() {
            return Err(MahjongError::InvalidEvent(
                "ron needs at least one winner and the latest discarder as loser".into(),
            ));
        }
        if winners.len() > 2 && self.rules.triple_ron_is_draw {
            return Err(MahjongError::InvalidEvent(
                "triple ron is configured as an abortive draw".into(),
            ));
        }
        if winners.len() > 1 && !self.rules.allow_double_ron {
            return Err(MahjongError::InvalidEvent(
                "multiple ron is disabled by the rule configuration".into(),
            ));
        }
        for winner in winners {
            winner.validate()?;
            if *winner == loser {
                return Err(MahjongError::InvalidEvent(
                    "ron winner cannot be the loser".into(),
                ));
            }
            if *winner == self.observer()
                && (self.observer_furiten.temporary || self.observer_furiten.riichi)
            {
                return Err(MahjongError::InvalidEvent(
                    "observer cannot ron while temporary/riichi furiten".into(),
                ));
            }
        }
        if chankan {
            self.cancel_pending_kan_after_ron(robbed_tile)?;
        }
        self.add_score_deltas(score_deltas);
        self.phase = TurnPhase::AwaitingRoundEnd {
            reason: RoundEndReason::Ron,
        };
        self.current_actor = None;
        Ok(())
    }

    fn apply_pass(&mut self, player: PlayerId, ron_available: bool) -> Result<()> {
        let (discarder, _, _) = self.ron_response_context()?;
        player.validate()?;
        if player == discarder {
            return Err(MahjongError::InvalidEvent(
                "discarder cannot pass on their own discard".into(),
            ));
        }
        if player == self.observer() && ron_available {
            if self.players[player.index()].riichi.is_accepted() {
                self.observer_furiten.riichi = true;
            } else {
                self.observer_furiten.temporary = true;
            }
        }
        Ok(())
    }

    fn apply_tsumo(&mut self, winner: PlayerId, score_deltas: [i32; 4]) -> Result<()> {
        self.require_discard_turn(winner)?;
        self.add_score_deltas(score_deltas);
        self.phase = TurnPhase::AwaitingRoundEnd {
            reason: RoundEndReason::Tsumo,
        };
        self.current_actor = None;
        Ok(())
    }

    fn add_score_deltas(&mut self, deltas: [i32; 4]) {
        for (score, delta) in self.scores.iter_mut().zip(deltas) {
            *score += delta;
        }
    }

    fn cancel_all_ippatsu(&mut self) {
        self.ippatsu_eligible = [false; 4];
    }

    fn require_active_round(&self) -> Result<()> {
        if self.round.is_none()
            || matches!(
                self.phase,
                TurnPhase::NotStarted | TurnPhase::AwaitingRoundEnd { .. } | TurnPhase::Ended
            )
        {
            return Err(MahjongError::InvalidEvent(
                "event requires an active round".into(),
            ));
        }
        Ok(())
    }

    fn require_discard_turn(&self, player: PlayerId) -> Result<()> {
        player.validate()?;
        if !matches!(self.phase, TurnPhase::AwaitingDiscard { player: expected } if expected == player)
        {
            return Err(MahjongError::InvalidEvent(format!(
                "player {} does not currently have a discard turn",
                player.get()
            )));
        }
        Ok(())
    }

    fn response_context(&self) -> Result<(PlayerId, Tile)> {
        match self.phase {
            TurnPhase::AwaitingResponse { discarder, tile } => Ok((discarder, tile)),
            _ => Err(MahjongError::InvalidEvent(
                "event requires a response window after a discard".into(),
            )),
        }
    }

    fn ron_response_context(&self) -> Result<(PlayerId, Tile, bool)> {
        match self.phase {
            TurnPhase::AwaitingResponse { discarder, tile } => Ok((discarder, tile, false)),
            TurnPhase::AwaitingKanResponse { declarer, tile, .. } => Ok((declarer, tile, true)),
            _ => Err(MahjongError::InvalidEvent(
                "ron/pass requires a discard or kan response window".into(),
            )),
        }
    }

    fn remove_observer_discard(&mut self, tile: Tile, tsumogiri: bool) -> Result<()> {
        match (self.drawn_tile, tsumogiri) {
            (Some(drawn), true) if drawn == tile => {
                self.drawn_tile = None;
                Ok(())
            }
            (Some(_), true) => Err(MahjongError::InvalidEvent(
                "tsumogiri tile does not match the observer's drawn tile".into(),
            )),
            (Some(drawn), false) => {
                self.remove_concealed_tile(tile)?;
                self.hand.push(drawn);
                self.drawn_tile = None;
                Ok(())
            }
            (None, false) => self.remove_concealed_tile(tile),
            (None, true) => Err(MahjongError::InvalidEvent(
                "tsumogiri is impossible without a drawn tile".into(),
            )),
        }
    }

    fn remove_concealed_tile(&mut self, tile: Tile) -> Result<()> {
        let position = self
            .hand
            .iter()
            .position(|candidate| *candidate == tile)
            .ok_or_else(|| {
                MahjongError::InvalidEvent(format!("observer does not hold tile {tile:?}"))
            })?;
        self.hand.remove(position);
        Ok(())
    }

    fn remove_observer_tile(&mut self, tile: Tile) -> Result<()> {
        if self.drawn_tile == Some(tile) {
            self.drawn_tile = None;
            Ok(())
        } else {
            self.remove_concealed_tile(tile)
        }
    }

    fn merge_drawn_tile_into_hand(&mut self) {
        if let Some(tile) = self.drawn_tile.take() {
            self.hand.push(tile);
        }
    }

    fn mark_latest_discard_called(&mut self, discarder: PlayerId, caller: PlayerId) -> Result<()> {
        let record = self.players[discarder.index()]
            .discards
            .last_mut()
            .ok_or_else(|| MahjongError::InvalidState("discard history is empty".into()))?;
        if record.called_by.is_some() {
            return Err(MahjongError::InvalidEvent(
                "latest discard was already called".into(),
            ));
        }
        record.called_by = Some(caller);
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(MahjongError::InvalidState(format!(
                "unsupported schema_version {}; expected {SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        self.rules.validate()?;
        self.information.validate()?;
        if let Some(round) = &self.round {
            round.validate()?;
        } else if !matches!(self.phase, TurnPhase::NotStarted) {
            return Err(MahjongError::InvalidState(
                "non-initial phase requires round metadata".into(),
            ));
        }
        for tile in &self.hand {
            tile.validate()?;
        }
        if let Some(tile) = self.drawn_tile {
            tile.validate()?;
        }
        for tile in &self.dora_indicators {
            tile.validate()?;
        }
        for player in &self.players {
            for discard in &player.discards {
                discard.tile.validate()?;
                if let Some(caller) = discard.called_by {
                    caller.validate()?;
                }
            }
            for meld in &player.melds {
                meld.validate()?;
            }
        }
        match (&self.pending_kan, self.phase) {
            (Some(pending), TurnPhase::AwaitingKanResponse { declarer, .. })
                if pending.player == declarer =>
            {
                pending.player.validate()?;
                pending.meld.validate()?;
            }
            (Some(_), _) => {
                return Err(MahjongError::InvalidState(
                    "pending kan is only valid during its response window".into(),
                ));
            }
            (None, TurnPhase::AwaitingKanResponse { .. }) => {
                return Err(MahjongError::InvalidState(
                    "kan response phase requires a pending kan".into(),
                ));
            }
            (None, _) => {}
        }
        if self.observer_furiten.riichi
            && !self.players[self.observer().index()].riichi.is_accepted()
        {
            return Err(MahjongError::InvalidState(
                "riichi furiten requires accepted riichi".into(),
            ));
        }
        for (index, eligible) in self.ippatsu_eligible.iter().enumerate() {
            if *eligible && !self.players[index].riichi.is_accepted() {
                return Err(MahjongError::InvalidState(
                    "ippatsu eligibility requires accepted riichi".into(),
                ));
            }
        }
        if self.remaining_tiles > 70 {
            return Err(MahjongError::InvalidState(
                "remaining_tiles exceeds the four-player live wall".into(),
            ));
        }
        self.validate_phase_actor()?;
        self.validate_history()?;
        self.validate_known_tile_counts()?;
        Ok(())
    }

    fn validate_phase_actor(&self) -> Result<()> {
        let expected = match self.phase {
            TurnPhase::AwaitingDraw { player }
            | TurnPhase::AwaitingRinshanDraw { player }
            | TurnPhase::AwaitingDiscard { player } => {
                player.validate()?;
                Some(player)
            }
            TurnPhase::AwaitingResponse { discarder, tile } => {
                discarder.validate()?;
                tile.validate()?;
                None
            }
            TurnPhase::AwaitingKanResponse {
                declarer,
                tile,
                ankan: _,
            } => {
                declarer.validate()?;
                tile.validate()?;
                None
            }
            TurnPhase::NotStarted | TurnPhase::AwaitingRoundEnd { .. } | TurnPhase::Ended => None,
        };
        if self.current_actor != expected {
            return Err(MahjongError::InvalidState(format!(
                "current_actor {:?} does not match phase {:?}",
                self.current_actor, self.phase
            )));
        }
        Ok(())
    }

    fn validate_history(&self) -> Result<()> {
        if self.event_history.len() as u64 != self.next_event_sequence {
            return Err(MahjongError::InvalidState(
                "event_history length does not match next_event_sequence".into(),
            ));
        }
        for (expected, event) in self.event_history.iter().enumerate() {
            if event.sequence != expected as u64 {
                return Err(MahjongError::InvalidState(
                    "event_history sequence is not contiguous from zero".into(),
                ));
            }
        }
        if self.last_event.as_ref() != self.event_history.last() {
            return Err(MahjongError::InvalidState(
                "last_event does not match event_history".into(),
            ));
        }
        Ok(())
    }

    fn validate_known_tile_counts(&self) -> Result<()> {
        let mut base_counts = [0_u8; 34];
        let mut red_counts = [0_u8; 3];
        let mut add = |tile: Tile| -> Result<()> {
            base_counts[tile.kind.index()] += 1;
            if base_counts[tile.kind.index()] > 4 {
                return Err(MahjongError::InvalidState(format!(
                    "more than four visible/held copies of {:?}",
                    tile.kind
                )));
            }
            if tile.red {
                let suit = tile.kind.suit_index().expect("red tile validation");
                red_counts[suit] += 1;
                if red_counts[suit] > self.rules.red_fives[suit] {
                    return Err(MahjongError::InvalidState(format!(
                        "red five count exceeds rules for suit {suit}"
                    )));
                }
            }
            Ok(())
        };

        for tile in &self.hand {
            add(*tile)?;
        }
        if let Some(tile) = self.drawn_tile {
            add(tile)?;
        }
        for tile in &self.dora_indicators {
            add(*tile)?;
        }
        for player in &self.players {
            for discard in &player.discards {
                if discard.called_by.is_none() {
                    add(discard.tile)?;
                }
            }
            for meld in &player.melds {
                for tile in &meld.tiles {
                    add(*tile)?;
                }
            }
        }
        if let Some(pending) = &self.pending_kan {
            if pending.replace_meld_index.is_some() {
                let existing = self.players[pending.player.index()]
                    .melds
                    .get(pending.replace_meld_index.expect("checked"))
                    .ok_or_else(|| {
                        MahjongError::InvalidState(
                            "pending shouminkan references a missing pon".into(),
                        )
                    })?;
                for tile in &pending.meld.tiles {
                    let pending_count = pending
                        .meld
                        .tiles
                        .iter()
                        .filter(|candidate| *candidate == tile)
                        .count();
                    let existing_count = existing
                        .tiles
                        .iter()
                        .filter(|candidate| *candidate == tile)
                        .count();
                    if pending_count > existing_count {
                        add(*tile)?;
                        break;
                    }
                }
            } else {
                for tile in &pending.meld.tiles {
                    add(*tile)?;
                }
            }
        }
        Ok(())
    }
}

/// Secret simulator/oracle data. Normal inference APIs never accept this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateGameState {
    pub concealed_hands: [Vec<Tile>; 4],
    pub live_wall: Vec<Tile>,
    pub dead_wall: Vec<Tile>,
}

/// Complete information is a separate top-level type to prevent accidental
/// inclusion in model inputs or observation JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompleteGameState {
    pub schema_version: u16,
    pub public: GameState,
    pub private: PrivateGameState,
}

impl CompleteGameState {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(MahjongError::InvalidState(
                "complete state schema version mismatch".into(),
            ));
        }
        self.public.validate()?;
        for tile in self
            .private
            .concealed_hands
            .iter()
            .flatten()
            .chain(&self.private.live_wall)
            .chain(&self.private.dead_wall)
        {
            tile.validate()?;
        }
        Ok(())
    }
}
