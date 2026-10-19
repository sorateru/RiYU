use std::array;

use crate::{
    Action, CompleteGameState, DrawGameReason, EventKind, GameEvent, GameState, KanAction,
    MahjongError, Meld, MeldKind, PlayerId, PrivateGameState, Result, RiichiStep, RonClaim,
    RoundEndReason, RoundState, RuleConfig, SCHEMA_VERSION, Tile, TileCounts, TileType, TurnPhase,
    WinMethod, calculate_shanten, legal_actions, noten_penalty, score_observer_win, settle_ron,
    settle_tsumo,
};

/// Deterministic four-player simulator used by local self-play workers.
/// It projects every event into four separate observation states.
#[derive(Debug, Clone)]
pub struct FourPlayerSimulator {
    rules: RuleConfig,
    states: [GameState; 4],
    live_wall: Vec<Tile>,
    dead_wall: Vec<Tile>,
    pending_riichi: Option<PlayerId>,
}

impl FourPlayerSimulator {
    pub fn new(seed: u64, rules: RuleConfig, round: RoundState, scores: [i32; 4]) -> Result<Self> {
        rules.validate()?;
        round.validate()?;
        let mut wall = build_wall(&rules);
        shuffle(&mut wall, seed);
        let mut dead_wall = wall.split_off(122);
        let mut hands: [Vec<Tile>; 4] = array::from_fn(|_| Vec::with_capacity(14));
        for _ in 0..13 {
            for player in PlayerId::ALL {
                hands[player.index()].push(wall.pop().expect("136 tile wall"));
            }
        }
        let dora_indicator = dead_wall[0];
        let mut built = Vec::with_capacity(4);
        for observer in PlayerId::ALL {
            let mut state = GameState::new(observer, rules.clone())?;
            state.apply_event(GameEvent {
                sequence: 0,
                kind: EventKind::RoundStart {
                    round: round.clone(),
                    scores,
                    initial_hand: hands[observer.index()].clone(),
                    dora_indicator,
                    remaining_tiles: wall.len() as u16,
                },
            })?;
            built.push(state);
        }
        let states: [GameState; 4] = built
            .try_into()
            .map_err(|_| MahjongError::InvalidState("failed to construct four states".into()))?;
        dead_wall.shrink_to_fit();
        Ok(Self {
            rules,
            states,
            live_wall: wall,
            dead_wall,
            pending_riichi: None,
        })
    }

    pub fn observation(&self, player: PlayerId) -> &GameState {
        &self.states[player.index()]
    }

    pub fn legal_actions(&self, player: PlayerId) -> Result<Vec<Action>> {
        Ok(legal_actions(self.observation(player))?.actions)
    }

    pub fn complete_state(&self, observer: PlayerId) -> Result<CompleteGameState> {
        let concealed_hands = array::from_fn(|index| {
            let mut tiles = self.states[index].hand.clone();
            tiles.extend(self.states[index].drawn_tile);
            tiles
        });
        let complete = CompleteGameState {
            schema_version: SCHEMA_VERSION,
            public: self.states[observer.index()].clone(),
            private: PrivateGameState {
                concealed_hands,
                live_wall: self.live_wall.clone(),
                dead_wall: self.dead_wall.clone(),
            },
        };
        complete.validate()?;
        Ok(complete)
    }

    pub fn draw(&mut self) -> Result<()> {
        let phase = self.states[0].phase;
        let (player, rinshan) = match phase {
            TurnPhase::AwaitingDraw { player } => (player, false),
            TurnPhase::AwaitingResponse { discarder, .. } => (discarder.next(), false),
            TurnPhase::AwaitingRinshanDraw { player }
            | TurnPhase::AwaitingKanResponse {
                declarer: player, ..
            } => (player, true),
            _ => {
                return Err(MahjongError::InvalidAction(
                    "simulator is not waiting for a draw".into(),
                ));
            }
        };
        if self.live_wall.is_empty() {
            return self.finish_exhaustive_draw();
        }
        if let Some(riichi) = self.pending_riichi.take() {
            self.broadcast(EventKind::Riichi {
                player: riichi,
                step: RiichiStep::Accepted,
            })?;
        }
        let tile = if rinshan {
            let tile = self.dead_wall.pop().ok_or_else(|| {
                MahjongError::InvalidState("dead wall lacks a rinshan tile".into())
            })?;
            self.dead_wall
                .push(self.live_wall.pop().expect("checked live wall"));
            tile
        } else {
            self.live_wall.pop().expect("checked live wall")
        };
        self.broadcast_draw(player, tile, rinshan)
    }

    pub fn act(&mut self, player: PlayerId, action: Action) -> Result<()> {
        if !self.legal_actions(player)?.contains(&action) {
            return Err(MahjongError::InvalidAction(
                "action is not in the current legal action set".into(),
            ));
        }
        match action {
            Action::Discard { tile, tsumogiri } => self.broadcast(EventKind::Discard {
                player,
                tile,
                tsumogiri,
                riichi_declaration: false,
            }),
            Action::Riichi { discard, tsumogiri } => {
                self.broadcast(EventKind::Riichi {
                    player,
                    step: RiichiStep::Declared,
                })?;
                self.broadcast(EventKind::Discard {
                    player,
                    tile: discard,
                    tsumogiri,
                    riichi_declaration: true,
                })?;
                self.pending_riichi = Some(player);
                Ok(())
            }
            Action::Kan { action } => {
                let meld = self.turn_kan_meld(player, action)?;
                self.broadcast(EventKind::Kan { player, meld })
            }
            Action::Tsumo => self.finish_tsumo(player),
            Action::AbortiveDraw { reason } => self.finish_abortive_draw(reason),
            Action::Chi { .. } | Action::Pon { .. } | Action::Ron { .. } | Action::Pass => Err(
                MahjongError::InvalidAction("response action must use resolve_responses".into()),
            ),
        }
    }

    /// Resolve one response per player (omitted players are treated as pass).
    pub fn resolve_responses(&mut self, choices: &[(PlayerId, Action)]) -> Result<()> {
        let (discarder, _) = match self.states[0].phase {
            TurnPhase::AwaitingResponse { discarder, tile } => (discarder, tile),
            TurnPhase::AwaitingKanResponse { declarer, tile, .. } => (declarer, tile),
            _ => {
                return Err(MahjongError::InvalidAction(
                    "simulator is not in a response window".into(),
                ));
            }
        };
        for (player, action) in choices {
            if !self.legal_actions(*player)?.contains(action) {
                return Err(MahjongError::InvalidAction(format!(
                    "illegal response by player {}",
                    player.get()
                )));
            }
        }
        let mut winners: Vec<_> = choices
            .iter()
            .filter_map(|(player, action)| matches!(action, Action::Ron { .. }).then_some(*player))
            .collect();
        winners.sort_by_key(|winner| turn_distance(discarder, *winner));
        if !winners.is_empty() {
            self.pending_riichi = None;
            if winners.len() == 3 && self.rules.triple_ron_is_draw {
                return self.finish_abortive_draw(DrawGameReason::TripleRon);
            }
            let round = self.states[0].round.as_ref().expect("active round").clone();
            let claims = winners
                .iter()
                .map(|winner| {
                    Ok(RonClaim {
                        winner: *winner,
                        payment: score_observer_win(&self.states[winner.index()], WinMethod::Ron)?
                            .payment,
                        responsibility: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let deltas = settle_ron(
                discarder,
                &claims,
                round.honba,
                round.riichi_sticks,
                &self.rules,
            )?;
            self.broadcast(EventKind::Ron {
                winners,
                loser: discarder,
                score_deltas: deltas,
            })?;
            return self.finish_round(RoundEndReason::Ron);
        }

        for player in PlayerId::ALL {
            if player == discarder {
                continue;
            }
            let report = legal_actions(&self.states[player.index()])?;
            let ron_available = report
                .actions
                .iter()
                .any(|action| matches!(action, Action::Ron { .. }));
            let passed = choices
                .iter()
                .find(|(candidate, _)| *candidate == player)
                .is_none_or(|(_, action)| matches!(action, Action::Pass));
            if ron_available && passed {
                self.broadcast(EventKind::Pass {
                    player,
                    ron_available: true,
                })?;
            }
        }
        if let Some(riichi) = self.pending_riichi.take() {
            self.broadcast(EventKind::Riichi {
                player: riichi,
                step: RiichiStep::Accepted,
            })?;
        }

        let mut calls: Vec<_> = choices
            .iter()
            .filter(|(_, action)| {
                matches!(
                    action,
                    Action::Chi { .. } | Action::Pon { .. } | Action::Kan { .. }
                )
            })
            .collect();
        calls.sort_by_key(|(player, action)| {
            let priority = if matches!(action, Action::Chi { .. }) {
                1
            } else {
                0
            };
            (priority, turn_distance(discarder, *player))
        });
        if let Some((player, action)) = calls.first() {
            let event = response_event(*player, (*action).clone())?;
            self.broadcast(event)?;
        }
        Ok(())
    }

    fn turn_kan_meld(&self, player: PlayerId, action: KanAction) -> Result<Meld> {
        match action {
            KanAction::Ankan { tiles } => Ok(Meld {
                kind: MeldKind::Ankan,
                tiles: tiles.to_vec(),
                called_tile: None,
                from: None,
            }),
            KanAction::Shouminkan { tile, meld_index } => {
                let existing = self.states[player.index()].players[player.index()]
                    .melds
                    .get(meld_index)
                    .ok_or_else(|| MahjongError::InvalidAction("invalid pon index".into()))?;
                let mut tiles = existing.tiles.clone();
                tiles.push(tile);
                Ok(Meld {
                    kind: MeldKind::Shouminkan,
                    tiles,
                    called_tile: existing.called_tile,
                    from: existing.from,
                })
            }
            KanAction::Daiminkan { .. } => Err(MahjongError::InvalidAction(
                "daiminkan is only a response action".into(),
            )),
        }
    }

    fn finish_tsumo(&mut self, winner: PlayerId) -> Result<()> {
        let evaluation = score_observer_win(&self.states[winner.index()], WinMethod::Tsumo)?;
        let round = self.states[0].round.as_ref().expect("active round").clone();
        let deltas = settle_tsumo(
            winner,
            round.dealer,
            &evaluation.payment,
            None,
            round.riichi_sticks,
            &self.rules,
        )?;
        self.broadcast(EventKind::Tsumo {
            winner,
            score_deltas: deltas,
        })?;
        self.finish_round(RoundEndReason::Tsumo)
    }

    fn finish_exhaustive_draw(&mut self) -> Result<()> {
        let mut tenpai = [false; 4];
        for player in PlayerId::ALL {
            let state = &self.states[player.index()];
            let counts = TileCounts::from_tiles(state.hand.iter().copied())?;
            tenpai[player.index()] =
                calculate_shanten(&counts, state.players[player.index()].melds.len() as u8)?
                    .minimum
                    == 0;
        }
        self.broadcast(EventKind::DrawGame {
            reason: DrawGameReason::Exhaustive,
            score_deltas: noten_penalty(tenpai),
        })?;
        self.finish_round(RoundEndReason::Draw)
    }

    fn finish_abortive_draw(&mut self, reason: DrawGameReason) -> Result<()> {
        self.broadcast(EventKind::DrawGame {
            reason,
            score_deltas: [0; 4],
        })?;
        self.finish_round(RoundEndReason::AbortiveDraw)
    }

    fn finish_round(&mut self, reason: RoundEndReason) -> Result<()> {
        let scores = self.states[0].scores;
        self.broadcast(EventKind::RoundEnd { reason, scores })
    }

    fn broadcast(&mut self, kind: EventKind) -> Result<()> {
        for state in &mut self.states {
            state.apply_event(GameEvent {
                sequence: state.next_event_sequence,
                kind: kind.clone(),
            })?;
        }
        Ok(())
    }

    fn broadcast_draw(&mut self, player: PlayerId, tile: Tile, rinshan: bool) -> Result<()> {
        for state in &mut self.states {
            state.apply_event(GameEvent {
                sequence: state.next_event_sequence,
                kind: EventKind::Draw {
                    player,
                    tile: (state.observer() == player).then_some(tile),
                    rinshan,
                },
            })?;
        }
        Ok(())
    }
}

fn response_event(player: PlayerId, action: Action) -> Result<EventKind> {
    match action {
        Action::Chi {
            called_tile,
            from,
            consume,
        } => Ok(EventKind::Chi {
            player,
            meld: Meld {
                kind: MeldKind::Chi,
                tiles: vec![called_tile, consume[0], consume[1]],
                called_tile: Some(called_tile),
                from: Some(from),
            },
        }),
        Action::Pon {
            called_tile,
            from,
            consume,
        } => Ok(EventKind::Pon {
            player,
            meld: Meld {
                kind: MeldKind::Pon,
                tiles: vec![called_tile, consume[0], consume[1]],
                called_tile: Some(called_tile),
                from: Some(from),
            },
        }),
        Action::Kan {
            action:
                KanAction::Daiminkan {
                    called_tile,
                    from,
                    consume,
                },
        } => Ok(EventKind::Kan {
            player,
            meld: Meld {
                kind: MeldKind::Daiminkan,
                tiles: vec![called_tile, consume[0], consume[1], consume[2]],
                called_tile: Some(called_tile),
                from: Some(from),
            },
        }),
        _ => Err(MahjongError::InvalidAction(
            "action is not a callable response".into(),
        )),
    }
}

fn turn_distance(from: PlayerId, to: PlayerId) -> u8 {
    (to.get() + 4 - from.get()) % 4
}

fn build_wall(rules: &RuleConfig) -> Vec<Tile> {
    let mut wall = Vec::with_capacity(136);
    for kind in TileType::ALL {
        for copy in 0..4 {
            let red = kind
                .suit_index()
                .is_some_and(|suit| kind.rank() == Some(5) && copy < rules.red_fives[suit]);
            wall.push(Tile { kind, red });
        }
    }
    wall
}

fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed.max(1);
    for index in (1..items.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let target = (state as usize) % (index + 1);
        items.swap(index, target);
    }
}
