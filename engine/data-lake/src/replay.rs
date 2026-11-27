use std::{collections::BTreeSet, time::Instant};

use anyhow::{Context, Result, bail, ensure};
use mahjong_core::{
    Action, DrawGameReason, EventKind, GameEvent, GameState, PlayerId, RiichiStep, RoundEndReason,
    RoundState, RoundWind, TurnPhase, WinMethod, legal_actions_with_shanten, noten_penalty,
    score_observer_win_with_ura,
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    actions::{Candidate, called_action, candidates, execution, semantic, tenhou_filter},
    local::{self, LocalLabels},
    physical::Ledger,
    source::{CanonicalEvent, Kind, Match, Start, digest},
    teachers::{
        Attack, DefenseCandidate, FuritenUncertain, HandAnalysis, HiddenHand, Timings, attack,
        defense, elapsed, hidden_hand,
    },
};

#[derive(Serialize)]
pub struct Decision {
    id: String,
    actor: u8,
    source_ordinal: u32,
    observation_seq: usize,
    window: &'static str,
    observation: Value,
    state_hash: String,
    history_hash: String,
    action_set_hash: String,
    actions: Vec<Candidate>,
    execution: Vec<Vec<u8>>,
    legal_status: &'static str,
    choice: Option<u8>,
    choice_status: &'static str,
    evidence_ordinal: u32,
    resolution: &'static str,
    riichi_accepted: Option<bool>,
    riichi_acceptance_ordinal: Option<u32>,
    attack: Attack,
    defense: Vec<DefenseCandidate>,
    local_actions: Option<LocalLabels>,
    opponents: Vec<HiddenHand>,
    next_opponent_action: [Option<u32>; 4],
}

#[derive(Serialize)]
pub struct Checkpoint {
    pub sequence: usize,
    pub state: GameState,
}

#[derive(Serialize)]
pub struct RoundResult {
    round_index: u16,
    decisions: Vec<Decision>,
    safe_events: [Vec<GameEvent>; 4],
    checkpoints16: [Vec<Checkpoint>; 4],
    terminal_observations: [Value; 4],
    scores: [i32; 4],
    delta: [i32; 4],
    reason: String,
    wins: Vec<u32>,
    scored_wins: usize,
    score_checks_skipped: usize,
    sticks_carried: u16,
    dealer_continues: bool,
}

#[derive(Serialize)]
pub struct Replay {
    rounds: Vec<RoundResult>,
    pub timings: Timings,
    restore: Vec<[u64; 4]>,
}

struct Round<'a> {
    match_id: &'a str,
    index: u16,
    states: [GameState; 4],
    ledger: Ledger,
    uncertain: [FuritenUncertain; 4],
    events: [Vec<GameEvent>; 4],
    checkpoints: [Vec<Checkpoint>; 4],
    histories: [String; 4],
    decisions: Vec<Decision>,
    cache: &'a mut HandAnalysis,
    timing: &'a mut Timings,
    checkpoints_enabled: bool,
}

fn player(seat: u8) -> PlayerId {
    PlayerId::new(seat).expect("validated seat")
}

// Safe view contains only facts visible at this moment. Tile serialization comes
// from mahjong-core's semantic Tile and has no physical copy identifier.
fn snapshot(s: &GameState) -> Value {
    let mut hand = s.hand.clone();
    hand.sort_by_key(|t| (t.kind.index(), t.red));
    json!({
        "seat": s.observer().get(), "round": s.round, "scores": s.scores, "hand": hand,
        "drawn": s.drawn_tile, "players": s.players, "dora": s.dora_indicators,
        "ippatsu": s.ippatsu_eligible, "phase": s.phase, "remaining": s.remaining_tiles, "turn": s.turn,
        "pending_kan": s.pending_kan,
    })
}

fn compact_state(s: &GameState, last: &GameEvent) -> GameState {
    let mut compact = s.clone();
    compact.hand.sort_by_key(|t| (t.kind.index(), t.red));
    compact.observer_furiten = Default::default();
    let mut event = last.clone();
    event.sequence = 0;
    compact.last_event = Some(event.clone());
    compact.event_history = vec![event];
    compact.next_event_sequence = 1;
    compact
}

impl Round<'_> {
    fn apply(&mut self, mut make: impl FnMut(usize) -> EventKind) -> Result<()> {
        let t = Instant::now();
        for i in 0..4 {
            let state = &mut self.states[i];
            let kind = make(i);
            state.apply_event(GameEvent {
                sequence: state.next_event_sequence,
                kind: kind.clone(),
            })?;
            let sequence = self.events[i].len();
            let event = GameEvent {
                sequence: sequence as u64,
                kind,
            };
            self.histories[i] = digest(&serde_json::to_vec(&(self.histories[i].as_str(), &event))?);
            if self.checkpoints_enabled && sequence % 16 == 0 {
                self.checkpoints[i].push(Checkpoint {
                    sequence,
                    state: compact_state(state, &event),
                });
            }
            self.events[i].push(event);
        }
        self.timing.replay_us += elapsed(t);
        Ok(())
    }

    fn check_hands(&self) -> Result<()> {
        for (i, state) in self.states.iter().enumerate() {
            let a = mahjong_core::TileCounts::from_tiles(
                state.hand.iter().copied().chain(state.drawn_tile),
            )?;
            let b = mahjong_core::TileCounts::from_tiles(
                self.ledger.hand[i]
                    .iter()
                    .chain(self.ledger.drawn[i].iter())
                    .map(|t| t.semantic()),
            )?;
            ensure!(
                a == b
                    && state.drawn_tile == self.ledger.drawn[i].map(|t| t.semantic())
                    && state.hand.iter().filter(|t| t.red).count()
                        == self.ledger.hand[i].iter().filter(|t| t.red).count(),
                "observer/physical hand mismatch seat {i}"
            );
        }
        Ok(())
    }

    fn extract(
        &mut self,
        seat: u8,
        trigger: u32,
        evidence: u32,
        window: &'static str,
        selection: Option<Action>,
        status: &'static str,
    ) -> Result<()> {
        let i = seat as usize;
        let t = Instant::now();
        let state = &self.states[i];
        let mut report =
            legal_actions_with_shanten(state, |counts, melds| self.cache.shanten(*counts, melds))?;
        tenhou_filter(state, &self.ledger, &mut report.actions);
        let mut legal_status = "complete";
        if self.uncertain[i].any()
            && report
                .actions
                .iter()
                .any(|a| matches!(a, Action::Ron { .. }))
        {
            legal_status = "unknown_furiten";
        }
        let rows = candidates(report.actions, seat, &self.ledger);
        if window != "turn" && rows.len() == 1 && legal_status == "complete" && selection.is_none()
        {
            self.timing.legal_us += elapsed(t);
            return Ok(());
        }
        let choice = if let Some(choice) = selection {
            let key = semantic(&choice, seat, &self.ledger);
            let found = rows.iter().position(|a| a.semantic == key);
            ensure!(
                found.is_some(),
                "selected {choice:?} is absent from legal set"
            );
            found.map(|i| i as u8)
        } else if status == "definite_pass" && legal_status == "complete" {
            rows.iter()
                .position(|a| a.action == Action::Pass)
                .map(|i| i as u8)
        } else {
            None
        };
        let execution = rows
            .iter()
            .map(|a| execution(&a.action, seat, &self.ledger))
            .collect::<Result<_>>()?;
        self.timing.legal_us += elapsed(t);
        let t = Instant::now();
        let observation = snapshot(state);
        let state_hash = digest(&serde_json::to_vec(&observation)?);
        let action_set_hash = digest(&serde_json::to_vec(&rows)?);
        let id = digest(
            format!(
                "riyu-decision-pilot-v1:{}:{}:{trigger}:{seat}:{window}",
                self.match_id, self.index
            )
            .as_bytes(),
        )[..32]
            .to_owned();
        self.timing.projection_us += elapsed(t);
        let t = Instant::now();
        let attack = attack(state, &rows, self.cache)?;
        self.timing.attack_us += elapsed(t);
        let t = Instant::now();
        let opponents: Vec<HiddenHand> = self
            .states
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, s)| hidden_hand(s, &self.ledger, self.uncertain[j], self.cache))
            .collect::<Result<_>>()?;
        self.timing.belief_us += elapsed(t);
        let t = Instant::now();
        let defense = defense(&self.states, seat, &rows, &opponents, &self.uncertain)?;
        self.timing.defense_us += elapsed(t);
        let t = Instant::now();
        let local_actions = local::labels(
            &self.states,
            seat,
            &rows,
            &attack,
            &opponents,
            &self.uncertain,
            &self.ledger,
            self.cache,
        )?;
        self.timing.local_us += elapsed(t);
        let resolution = match (status, legal_status) {
            ("selected", "complete") => "selected",
            (_, "unknown_furiten") => "incomplete_legal_set",
            ("definite_pass", _) => "confirmed_pass",
            _ => "priority_censored",
        };
        self.decisions.push(Decision {
            id,
            actor: seat,
            source_ordinal: trigger,
            observation_seq: self.events[i].len() - 1,
            window,
            observation,
            state_hash,
            history_hash: self.histories[i].clone(),
            action_set_hash,
            actions: rows,
            execution,
            legal_status,
            choice,
            choice_status: status,
            evidence_ordinal: evidence,
            resolution,
            riichi_accepted: None,
            riichi_acceptance_ordinal: None,
            attack,
            defense,
            local_actions,
            opponents,
            next_opponent_action: [None; 4],
        });
        Ok(())
    }

    fn close_response(
        &mut self,
        trigger: u32,
        closer: &CanonicalEvent,
        winners: &[u8],
    ) -> Result<()> {
        let discarder = match self.states[0].phase {
            TurnPhase::AwaitingResponse { discarder, .. } => discarder,
            TurnPhase::AwaitingKanResponse { declarer, .. } => declarer,
            _ => return Ok(()),
        };
        let (tile, window) = match self.states[0].phase {
            TurnPhase::AwaitingResponse { tile, .. } => (tile, "discard_response"),
            TurnPhase::AwaitingKanResponse { tile, .. } => (tile, "kan_response"),
            _ => unreachable!(),
        };
        let passed = matches!(closer.kind, Kind::Draw { .. });
        let mut misses = [None; 4];
        for seat in 0..4u8 {
            if seat == discarder.get() {
                continue;
            }
            let i = seat as usize;
            let choice = if winners.contains(&seat) {
                Some(Action::Ron { target: discarder })
            } else if let Kind::Call {
                seat: caller,
                ref meld,
                ..
            } = closer.kind
            {
                if seat == caller && !matches!(meld.kind, "ankan" | "shouminkan") {
                    Some(called_action(meld, seat, &self.ledger)?)
                } else {
                    None
                }
            } else {
                None
            };
            let selected = choice.is_some();
            let selected_ron = matches!(choice, Some(Action::Ron { .. }));
            let result = if selected {
                "selected"
            } else if passed {
                "definite_pass"
            } else {
                "censored"
            };
            self.extract(seat, trigger, closer.source_ordinal, window, choice, result)?;
            if selected_ron {
                continue;
            }
            let state = &self.states[i];
            let (_, _, _, wait) = self.cache.hand(state)?;
            let structural_wait = wait.is_some_and(|mask| mask & (1 << tile.kind.index()) != 0);
            // In a concealed-kan window only the supported kokushi exception
            // could establish a miss. The initial Tenhou profiles disable it.
            let eligible_window = !matches!(
                state.phase,
                TurnPhase::AwaitingKanResponse { ankan: true, .. }
            );
            if structural_wait && eligible_window {
                misses[i] = Some(passed || selected);
            }
        }
        for (i, miss) in misses.into_iter().enumerate() {
            if let Some(definite) = miss {
                if definite {
                    let state = &mut self.states[i];
                    state.apply_event(GameEvent {
                        sequence: state.next_event_sequence,
                        kind: EventKind::Pass {
                            player: player(i as u8),
                            ron_available: true,
                        },
                    })?;
                } else if self.states[i].players[i].riichi.is_accepted() {
                    self.uncertain[i].riichi = true;
                } else {
                    self.uncertain[i].temporary = true;
                }
            }
        }
        Ok(())
    }
}

pub fn replay(
    source: &Match,
    use_cache: bool,
    checkpoints_enabled: bool,
    measure_restore: bool,
) -> Result<Replay> {
    let rules = source.rules().context("unsupported rule profile")?;
    let mut result = Replay {
        rounds: Vec::new(),
        timings: Timings::default(),
        restore: Vec::new(),
    };
    let mut cache = HandAnalysis::new(use_cache);
    let starts: Vec<usize> = source
        .events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e.kind, Kind::Start(_)).then_some(i))
        .collect();
    ensure!(!starts.is_empty(), "no rounds");
    let mut previous = None;
    for (index, &begin) in starts.iter().enumerate() {
        let end = starts
            .get(index + 1)
            .copied()
            .unwrap_or(source.events.len());
        let events = &source.events[begin..end];
        let Kind::Start(ref start) = events[0].kind else {
            unreachable!()
        };
        if let Some((number, dealer, honba, sticks, scores)) = previous {
            ensure!(
                (
                    start.hand_number,
                    start.dealer,
                    start.honba,
                    start.sticks,
                    start.scores
                ) == (number, dealer, honba, sticks, scores),
                "round progression or starting scores mismatch at round {index}"
            );
        }
        let phase = Round {
            match_id: &source.id,
            index: index.try_into()?,
            states: std::array::from_fn(|i| {
                GameState::new(player(i as u8), rules.clone()).expect("validated rule")
            }),
            ledger: Ledger::new(start)?,
            uncertain: Default::default(),
            events: std::array::from_fn(|_| Vec::new()),
            checkpoints: std::array::from_fn(|_| Vec::new()),
            histories: std::array::from_fn(|_| digest(b"riyu-visible-history-pilot-v1")),
            decisions: Vec::new(),
            cache: &mut cache,
            timing: &mut result.timings,
            checkpoints_enabled,
        };
        let round = replay_round(phase, start, events).with_context(|| format!("round {index}"))?;
        if measure_restore {
            let n = round.decisions.len();
            for ordinal in [n / 5, n / 2, n * 4 / 5] {
                if let Some(decision) = round.decisions.get(ordinal) {
                    result
                        .restore
                        .push(check_restore(&round, decision, &rules)?);
                }
            }
        }
        let advances = u8::from(!round.dealer_continues);
        let next_honba = if !round.wins.is_empty() && advances != 0 {
            0
        } else {
            start.honba + 1
        };
        previous = Some((
            start.hand_number + advances,
            (start.dealer + advances) % 4,
            next_honba,
            round.sticks_carried,
            round.scores,
        ));
        result.rounds.push(round);
    }
    let last = result.rounds.last().unwrap();
    if let Some(final_scores) = source.final_scores {
        let mut expected = last.scores;
        let top = source
            .final_ranks
            .context("missing source placement")?
            .iter()
            .position(|&r| r == 1)
            .unwrap();
        expected[top] += 1000 * i32::from(last.sticks_carried);
        ensure!(
            final_scores == expected,
            "final scores do not match terminal plus remaining sticks"
        );
    }
    result.timings.shanten_evaluations = cache.evaluations;
    result.timings.shanten_cache_hits = cache.hits;
    Ok(result)
}

fn check_restore(
    round: &RoundResult,
    decision: &Decision,
    rules: &mahjong_core::RuleConfig,
) -> Result<[u64; 4]> {
    let seat = decision.actor as usize;
    let mut nanos = [0; 4];
    for (i, interval) in [0, 16, 32, 64].into_iter().enumerate() {
        let start = Instant::now();
        let checkpoint = (interval != 0)
            .then(|| {
                round.checkpoints16[seat]
                    .iter()
                    .rev()
                    .find(|c| c.sequence <= decision.observation_seq && c.sequence % interval == 0)
            })
            .flatten();
        let (mut state, from) = if let Some(c) = checkpoint {
            (c.state.clone(), c.sequence + 1)
        } else {
            (GameState::new(player(decision.actor), rules.clone())?, 0)
        };
        for event in &round.safe_events[seat][from..=decision.observation_seq] {
            state.apply_event(GameEvent {
                sequence: state.next_event_sequence,
                kind: event.kind.clone(),
            })?;
        }
        let recovered = snapshot(&state);
        nanos[i] = start.elapsed().as_nanos() as u64;
        ensure!(
            recovered == decision.observation,
            "observer-only restore disagrees with live replay"
        );
    }
    Ok(nanos)
}

fn replay_round(mut r: Round<'_>, start: &Start, events: &[CanonicalEvent]) -> Result<RoundResult> {
    let wind = [
        RoundWind::East,
        RoundWind::South,
        RoundWind::West,
        RoundWind::North,
    ]
    .get((start.hand_number / 4) as usize)
    .copied()
    .context("round wind out of range")?;
    r.apply(|i| EventKind::RoundStart {
        round: RoundState {
            wind,
            hand_number: start.hand_number % 4 + 1,
            honba: start.honba,
            riichi_sticks: start.sticks,
            dealer: player(start.dealer),
        },
        scores: start.scores,
        initial_hand: start.hands[i].iter().map(|t| t.semantic()).collect(),
        dora_indicator: start.dora.semantic(),
        remaining_tiles: 70,
    })?;
    r.check_hands()?;
    let mut turn_trigger = events[0].source_ordinal;
    let mut response_trigger = 0;
    let mut rinshan = None;
    let mut declared = None;
    let mut acceptance = None;
    let mut turn_dora = Vec::new();
    let mut scheduled_call_discard: Option<(u8, bool)> = None;
    let mut i = 1;
    while i < events.len() {
        let event = &events[i];
        i += 1;
        let seq = event.source_ordinal;
        let step = (|| -> Result<Option<RoundResult>> {
            match event.kind {
                Kind::Draw { seat, tile } => {
                    r.close_response(response_trigger, event, &[])?;
                    accept(&mut r, acceptance.take())?;
                    let is_rinshan = rinshan.take() == Some(seat);
                    r.ledger.draw(seat, tile)?;
                    r.uncertain[seat as usize].temporary = false;
                    r.apply(|observer| EventKind::Draw { player: player(seat),
                        tile: (observer == seat as usize).then(|| tile.semantic()), rinshan: is_rinshan })?;
                    turn_trigger = seq;
                }
                Kind::Discard { seat, tile } => {
                    let tsumogiri = r.ledger.drawn[seat as usize] == Some(tile);
                    let riichi = declared.take() == Some(seat);
                    if !riichi {
                        r.extract(seat, turn_trigger, seq, "turn", Some(Action::Discard { tile: tile.semantic(), tsumogiri }), "selected")?;
                    }
                    reveal_turn_dora(&mut r, &mut turn_dora)?;
                    r.ledger.discard(seat, tile)?;
                    r.apply(|_| EventKind::Discard { player: player(seat), tile: tile.semantic(), tsumogiri,
                        tsumogiri_known: true, riichi_declaration: riichi })?;
                    if let Some((caller, normal_order)) = scheduled_call_discard.take() {
                        ensure!(caller == seat, "discard does not follow its open call");
                        let i = seat as usize;
                        let definite = std::mem::take(&mut r.states[i].observer_furiten.temporary);
                        // On the ordinary next turn the discard ended the
                        // missed-win window. Non-adjacent pon can also clear
                        // it in Tenhou; until turn order is fully resolved,
                        // preserve both possibilities for such calls.
                        r.uncertain[i].temporary = !normal_order && (r.uncertain[i].temporary || definite);
                    }
                    response_trigger = seq;
                }
                Kind::Riichi { seat, step, scores } => {
                    if step == 1 {
                        let discard = events[i..].iter().find(|e| !matches!(e.kind, Kind::Disconnect { .. } | Kind::Players { .. })).context("riichi declaration lacks discard")?;
                        let Kind::Discard { seat: actual, tile } = discard.kind else { bail!("riichi declaration lacks immediate discard"); };
                        ensure!(actual == seat, "riichi discard seat mismatch");
                        r.extract(seat, turn_trigger, seq, "turn", Some(Action::Riichi { discard: tile.semantic(),
                            tsumogiri: r.ledger.drawn[seat as usize] == Some(tile) }), "selected")?;
                        declared = Some(seat);
                        r.apply(|_| EventKind::Riichi { player: player(seat), step: RiichiStep::Declared })?;
                        ensure!(scores.is_none_or(|s| s == r.states[0].scores), "riichi scores mismatch");
                    } else {
                        ensure!(acceptance.replace((seat, scores)).is_none(), "multiple pending riichi acceptances");
                    }
                }
                Kind::Dora { tile } => {
                    if matches!(r.states[0].phase, TurnPhase::AwaitingDiscard { .. }) {
                        turn_dora.push(tile);
                    } else {
                        r.ledger.dora(tile)?;
                        r.apply(|_| EventKind::DoraReveal { tile: tile.semantic() })?;
                    }
                }
                Kind::Call { seat, ref meld, .. } => {
                    let action = called_action(meld, seat, &r.ledger)?;
                    if matches!(meld.kind, "ankan" | "shouminkan") {
                        r.extract(seat, turn_trigger, seq, "turn", Some(action), "selected")?;
                        reveal_turn_dora(&mut r, &mut turn_dora)?;
                    } else {
                        r.close_response(response_trigger, event, &[])?;
                        accept(&mut r, acceptance.take())?;
                    }
                    let m = meld.semantic()?;
                    r.ledger.call(seat, meld)?;
                    r.apply(|_| match meld.kind {
                        "chi" => EventKind::Chi { player: player(seat), meld: m.clone() },
                        "pon" => EventKind::Pon { player: player(seat), meld: m.clone() },
                        _ => EventKind::Kan { player: player(seat), meld: m.clone() },
                    })?;
                    if matches!(meld.kind, "chi" | "pon") {
                        turn_trigger = seq;
                        scheduled_call_discard = meld.from.map(|from| (seat, (from + 1) % 4 == seat));
                    } else {
                        rinshan = Some(seat);
                        response_trigger = seq;
                    }
                }
                Kind::Win { seat, from, .. } => {
                    let mut last = i;
                    while last < events.len() && matches!(events[last].kind, Kind::Win { .. }) { last += 1; }
                    let records = &events[i-1..last];
                    i = last;
                    let winners: Vec<u8> = records.iter().map(|e| match e.kind { Kind::Win { seat, .. } => seat, _ => unreachable!() }).collect();
                    ensure!(winners.iter().copied().collect::<BTreeSet<_>>().len() == winners.len(), "duplicate winner");
                    if seat == from {
                        ensure!(winners.len() == 1, "multiple tsumo winners");
                        r.extract(seat, turn_trigger, seq, "turn", Some(Action::Tsumo), "selected")?;
                    } else {
                        r.close_response(response_trigger, event, &winners)?;
                    }
                    reveal_turn_dora(&mut r, &mut turn_dora)?;
                    accept(&mut r, acceptance.take())?;
                    let baseline = r.states[0].scores;
                    let mut scores = baseline;
                    let mut scored = 0;
                    let mut skipped = 0;
                    for win in records {
                        let Kind::Win { seat: winner, from: loser, tile, ref hand, fu, points, ref dora, ref ura, responsibility, .. } = win.kind else { unreachable!() };
                        ensure!(loser == from, "multi-ron target mismatch");
                        let mut expected: Vec<_> = r.ledger.hand[winner as usize].iter().chain(r.ledger.drawn[winner as usize].iter()).map(|t| t.id).collect();
                        if winner != loser { expected.push(tile.id); }
                        let mut actual = hand.iter().map(|t| t.id).collect::<Vec<_>>();
                        expected.sort_unstable();
                        actual.sort_unstable();
                        ensure!(expected == actual, "AGARI physical hand mismatch");
                        if winner == loser {
                            ensure!(r.ledger.drawn[winner as usize] == Some(tile), "AGARI draw mismatch");
                        } else if let Some((discarder, discarded)) = r.ledger.latest_discard {
                            ensure!((discarder, discarded) == (loser, tile), "AGARI discard mismatch");
                        } else {
                            ensure!(matches!(r.states[0].phase, TurnPhase::AwaitingKanResponse { .. }) &&
                                r.ledger.melds[loser as usize].iter().any(|meld| meld.added == Some(tile)), "AGARI robbed tile mismatch");
                        }
                        let Kind::Win { scores: ref change, .. } = win.kind else { unreachable!() };
                        ensure!(change.before == baseline || change.before == scores, "AGARI score baseline mismatch");
                        scores = sum(scores, change.delta);
                        let win_state = &r.states[winner as usize];
                        if responsibility.is_some() || wins_have_pending_dora(win_state, dora) {
                            skipped += 1;
                        } else {
                            let calculation = score_observer_win_with_ura(win_state,
                                if winner == loser { WinMethod::Tsumo } else { WinMethod::Ron },
                                &ura.iter().map(|t| t.semantic().kind).collect::<Vec<_>>())?;
                            let base_payment = calculation.payment.total_payment as i32 -
                                300 * i32::from(start.honba);
                            ensure!((calculation.yakuman_multiplier != 0 || fu == calculation.fu.rounded_total)
                                && points == base_payment,
                                "AGARI scoring mismatch: source fu/points {fu}/{points}, calculated {}/{base_payment}",
                                calculation.fu.rounded_total);
                            scored += 1;
                        }
                    }
                    let delta = sub(scores, baseline);
                    let reason = if seat == from {
                        r.apply(|_| EventKind::Tsumo { winner: player(seat), score_deltas: delta })?;
                        RoundEndReason::Tsumo
                    } else {
                        if matches!(r.states[0].phase, TurnPhase::AwaitingKanResponse { .. }) {
                            let Kind::Win { tile, .. } = records[0].kind else { unreachable!() };
                            r.ledger.cancel_added_kan(from, tile)?;
                        }
                        r.apply(|_| EventKind::Ron { winners: winners.iter().map(|&w| player(w)).collect(),
                            loser: player(from), score_deltas: delta })?;
                        RoundEndReason::Ron
                    };
                    return Ok(Some(finish(&mut r, start, events, scores, reason,
                        if seat == from { "tsumo" } else { "ron" }, records.iter().map(|e| e.source_ordinal).collect(),
                        scored, skipped, 0, winners.contains(&start.dealer))?));
                }
                Kind::DrawGame { ref reason, ref hands, ref scores } => {
                    let reason_enum = match reason.as_str() {
                        "exhaustive" | "nm" => DrawGameReason::Exhaustive,
                        "yao9" => DrawGameReason::NineTerminals, "kaze4" => DrawGameReason::FourWinds,
                        "reach4" => DrawGameReason::FourRiichi, "kan4" => DrawGameReason::FourKan,
                        "ron3" => DrawGameReason::TripleRon,
                        other => bail!("unsupported draw reason {other}"),
                    };
                    if reason_enum == DrawGameReason::NineTerminals {
                        let seat: u8 = hands.iter().position(Option::is_some).context("nine terminals without revealed hand")?.try_into()?;
                        r.extract(seat, turn_trigger, seq, "turn", Some(Action::AbortiveDraw { reason: reason_enum }), "selected")?;
                    } else {
                        r.close_response(response_trigger, event, &[])?;
                    }
                    accept(&mut r, acceptance.take())?;
                    reveal_turn_dora(&mut r, &mut turn_dora)?;
                    ensure!(scores.before == r.states[0].scores, "draw score baseline mismatch");
                    let ordinary = reason == "exhaustive";
                    let is_exhaustive = reason_enum == DrawGameReason::Exhaustive;
                    let mut tenpai = [false; 4];
                    if is_exhaustive {
                        for (seat, hand) in hands.iter().enumerate() {
                            if let Some(hand) = hand {
                                let mut a: Vec<_> = hand.iter().map(|t| t.id).collect();
                                let mut b: Vec<_> = r.ledger.hand[seat].iter().map(|t| t.id).collect();
                                a.sort_unstable();
                                b.sort_unstable();
                                ensure!(a == b, "revealed draw hand disagrees with physical replay");
                            }
                            if ordinary { tenpai[seat] = hand.is_some(); }
                            else { tenpai[seat] = r.cache.hand(&r.states[seat])?.2 == 0; }
                        }
                    }
                    if ordinary {
                        ensure!(noten_penalty(tenpai) == scores.delta, "draw score delta mismatch");
                    } else if !is_exhaustive {
                        ensure!(scores.delta == [0; 4], "abortive draw score delta not zero");
                    }
                    let terminal = sum(scores.before, scores.delta);
                    let sticks = r.states[0].round.as_ref().unwrap().riichi_sticks;
                    r.apply(|_| EventKind::DrawGame { reason: reason_enum, score_deltas: scores.delta })?;
                    return Ok(Some(finish(&mut r, start, events, terminal,
                        if is_exhaustive { RoundEndReason::Draw } else { RoundEndReason::AbortiveDraw },
                        reason, Vec::new(), 0, 0, sticks, !is_exhaustive || tenpai[start.dealer as usize])?));
                }
                Kind::Players { .. } | Kind::Disconnect { .. } => {}
                Kind::Other => bail!("unmapped round event {}", event.tag),
                _ => bail!("unexpected match event inside round"),
            }
            r.check_hands()?;
            Ok(None)
        })().with_context(|| format!("event {} ({})", event.round_seq.unwrap_or(0), event.tag.chars().next().unwrap_or('?')))?;
        if let Some(result) = step {
            ensure!(
                events[i..]
                    .iter()
                    .all(|e| matches!(e.kind, Kind::Players { .. } | Kind::Disconnect { .. })),
                "unexpected gameplay event after terminal"
            );
            return Ok(result);
        }
    }
    bail!("missing terminal event")
}

fn wins_have_pending_dora(state: &GameState, source: &[crate::source::PhysicalTile]) -> bool {
    state
        .dora_indicators
        .iter()
        .copied()
        .ne(source.iter().map(|t| t.semantic()))
}

fn accept(r: &mut Round<'_>, acceptance: Option<(u8, Option<[i32; 4]>)>) -> Result<()> {
    if let Some((seat, scores)) = acceptance {
        r.apply(|_| EventKind::Riichi {
            player: player(seat),
            step: RiichiStep::Accepted,
        })?;
        ensure!(
            scores.is_none_or(|s| s == r.states[0].scores),
            "riichi scores mismatch"
        );
    }
    Ok(())
}

fn reveal_turn_dora(
    r: &mut Round<'_>,
    pending: &mut Vec<crate::source::PhysicalTile>,
) -> Result<()> {
    for tile in pending.drain(..) {
        r.ledger.dora(tile)?;
        r.apply(|_| EventKind::DoraReveal {
            tile: tile.semantic(),
        })?;
    }
    Ok(())
}

fn sum(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    std::array::from_fn(|i| a[i] - b[i])
}

#[allow(clippy::too_many_arguments)]
fn finish(
    r: &mut Round<'_>,
    start: &Start,
    events: &[CanonicalEvent],
    scores: [i32; 4],
    reason: RoundEndReason,
    source_reason: &str,
    wins: Vec<u32>,
    scored_wins: usize,
    score_checks_skipped: usize,
    sticks_carried: u16,
    dealer_continues: bool,
) -> Result<RoundResult> {
    r.apply(|_| EventKind::RoundEnd { reason, scores })?;
    r.check_hands()?;
    let mut next = [None; 4];
    let actions: Vec<_> = events
        .iter()
        .filter_map(|e| {
            let actor = match e.kind {
                Kind::Discard { seat, .. }
                | Kind::Call { seat, .. }
                | Kind::Win { seat, .. }
                | Kind::Riichi { seat, step: 1, .. } => seat,
                _ => return None,
            };
            Some((e.source_ordinal, actor))
        })
        .collect();
    let mut cursor = actions.len();
    let mut accepted = [None; 4];
    for event in events {
        if let Kind::Riichi { seat, step: 2, .. } = event.kind {
            accepted[seat as usize] = Some(event.source_ordinal);
        }
    }
    r.decisions.sort_by_key(|d| (d.source_ordinal, d.actor));
    for decision in r.decisions.iter_mut().rev() {
        while cursor > 0 && actions[cursor - 1].0 > decision.source_ordinal {
            cursor -= 1;
            let (ordinal, actor) = actions[cursor];
            next[actor as usize] = Some(ordinal);
        }
        decision.next_opponent_action = next;
        decision.next_opponent_action[decision.actor as usize] = None;
        if decision
            .choice
            .is_some_and(|i| matches!(decision.actions[i as usize].action, Action::Riichi { .. }))
        {
            let acceptance = accepted[decision.actor as usize]
                .filter(|&ordinal| ordinal > decision.evidence_ordinal);
            decision.riichi_accepted = Some(acceptance.is_some());
            decision.riichi_acceptance_ordinal = acceptance;
        }
    }
    Ok(RoundResult {
        round_index: r.index,
        decisions: std::mem::take(&mut r.decisions),
        safe_events: std::mem::take(&mut r.events),
        checkpoints16: std::mem::take(&mut r.checkpoints),
        terminal_observations: std::array::from_fn(|i| snapshot(&r.states[i])),
        scores,
        delta: sub(scores, start.scores),
        reason: source_reason.to_owned(),
        wins,
        scored_wins,
        score_checks_skipped,
        sticks_carried,
        dealer_continues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::normalize;

    fn log(permute: bool) -> Match {
        let own = vec![0, 32, 36, 68, 72, 104, 108, 112, 116, 4, 8, 12, 20];
        let draw = 24;
        let indicator = 124;
        let mut free = (0..136).filter(|t| !own.contains(t) && *t != draw && *t != indicator);
        let hands = [
            own.clone(),
            free.by_ref().take(13).collect(),
            free.by_ref().take(13).collect(),
            free.take(13).collect(),
        ];
        let map = |id: u8| {
            if permute && !matches!(id / 4, 4 | 13 | 22) {
                id ^ 1
            } else {
                id
            }
        };
        let csv = |ids: &[u8]| {
            ids.iter()
                .map(|&t| map(t).to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut xml = format!(
            r#"<mjloggm ver="test"><GO type="169" lobby="0"/><TAIKYOKU oya="0"/><INIT seed="0,0,0,1,2,{}" oya="0" ten="250,250,250,250""#,
            map(indicator)
        );
        for (seat, hand) in hands.iter().enumerate() {
            xml.push_str(&format!(r#" hai{seat}="{}""#, csv(hand)));
        }
        let mut revealed = hands[0].clone();
        revealed.push(draw);
        xml.push_str(&format!(
            r#"/><T{}/><RYUUKYOKU type="yao9" hai0="{}" sc="250,0,250,0,250,0,250,0"/></mjloggm>"#,
            map(draw),
            csv(&revealed)
        ));
        normalize("2020010100-synthetic".into(), xml.as_bytes()).unwrap()
    }

    #[test]
    fn physical_copy_swap_preserves_observation_and_legal_actions() {
        let a = log(false);
        let b = log(true);
        assert_ne!(a.payload_sha256, b.payload_sha256);
        let x = replay(&a, true, true, true).unwrap();
        let y = replay(&b, false, true, true).unwrap();
        let d = &x.rounds[0].decisions[0];
        let e = &y.rounds[0].decisions[0];
        assert_eq!(d.observation, e.observation);
        assert_eq!(d.state_hash, e.state_hash);
        assert_eq!(d.history_hash, e.history_hash);
        assert_eq!(d.action_set_hash, e.action_set_hash);
        assert_eq!(d.choice, e.choice);
        assert_ne!(d.execution, e.execution);
        assert_eq!(
            d.actions[d.choice.unwrap() as usize].action,
            Action::AbortiveDraw {
                reason: DrawGameReason::NineTerminals
            }
        );
        assert_eq!(
            serde_json::to_value(&d.attack).unwrap(),
            serde_json::to_value(&e.attack).unwrap()
        );
        for observer in 1..4 {
            assert!(matches!(
                x.rounds[0].safe_events[observer][1].kind,
                EventKind::Draw { tile: None, .. }
            ));
        }
    }

    #[test]
    fn alternate_opponent_hands_change_targets_but_not_the_actors_observation() {
        let a = log(false);
        let mut b = log(false);
        let start = b
            .events
            .iter_mut()
            .find_map(|e| match &mut e.kind {
                Kind::Start(s) => Some(s),
                _ => None,
            })
            .unwrap();
        let first = start.hands[1][0];
        let other = start.hands[3][0];
        assert_ne!(first.kind, other.kind);
        start.hands[1][0] = other;
        start.hands[3][0] = first;
        let x = replay(&a, true, true, false).unwrap();
        let y = replay(&b, true, true, false).unwrap();
        let d = &x.rounds[0].decisions[0];
        let e = &y.rounds[0].decisions[0];
        assert_eq!(d.actor, 0);
        assert_eq!(d.observation, e.observation);
        assert_eq!(
            (&d.state_hash, &d.history_hash, &d.action_set_hash),
            (&e.state_hash, &e.history_hash, &e.action_set_hash)
        );
        assert_ne!(
            serde_json::to_value(&d.opponents).unwrap(),
            serde_json::to_value(&e.opponents).unwrap()
        );
    }
}
