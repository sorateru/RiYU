use mahjong_core::{
    Action, FourPlayerSimulator, GameEvent, GameState, Meld, PlayerId, RoundState, RuleConfig,
    Tile, TileCounts, WinContext, analyze_state, calculate_shanten, game_state_from_json,
    game_state_to_json, legal_actions, replay_events, score_hand,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Validate and normalize a GameState JSON document.
#[pyfunction]
fn validate_state_json(state_json: &str) -> PyResult<String> {
    let state = game_state_from_json(state_json).map_err(value_error)?;
    game_state_to_json(&state).map_err(value_error)
}

/// Apply one observed event atomically and return the resulting GameState JSON.
#[pyfunction]
fn apply_event_json(state_json: &str, event_json: &str) -> PyResult<String> {
    let mut state = game_state_from_json(state_json).map_err(value_error)?;
    let event: GameEvent = serde_json::from_str(event_json).map_err(value_error)?;
    state.apply_event(event).map_err(value_error)?;
    game_state_to_json(&state).map_err(value_error)
}

/// Return strict rule analysis. Model-only metrics remain explicitly unavailable.
#[pyfunction]
fn analyze_state_json(state_json: &str) -> PyResult<String> {
    let state = game_state_from_json(state_json).map_err(value_error)?;
    let analysis = analyze_state(&state).map_err(value_error)?;
    serde_json::to_string_pretty(&analysis).map_err(value_error)
}

/// Return only authoritative legal actions without computing discard analysis.
#[pyfunction]
fn legal_actions_json(state_json: &str) -> PyResult<String> {
    let state = game_state_from_json(state_json).map_err(value_error)?;
    let report = legal_actions(&state).map_err(value_error)?;
    serde_json::to_string(&report.actions).map_err(value_error)
}

/// Calculate exact standard/chiitoitsu/kokushi shanten from a 34-count array.
#[pyfunction]
#[pyo3(signature = (counts_json, fixed_melds=0))]
fn calculate_shanten_json(counts_json: &str, fixed_melds: u8) -> PyResult<String> {
    let counts: TileCounts = serde_json::from_str(counts_json).map_err(value_error)?;
    let result = calculate_shanten(&counts, fixed_melds).map_err(value_error)?;
    serde_json::to_string(&result).map_err(value_error)
}

/// Score a complete hypothetical hand with the authoritative Rust rules.
///
/// This accepts an explicit hand rather than GameState so belief particles can
/// be evaluated without ever copying their hidden tiles into an observation.
#[pyfunction]
#[pyo3(signature = (concealed_json, melds_json, winning_tile_json, context_json, rules_json=None))]
fn score_hand_json(
    concealed_json: &str,
    melds_json: &str,
    winning_tile_json: &str,
    context_json: &str,
    rules_json: Option<&str>,
) -> PyResult<String> {
    let concealed: Vec<Tile> = serde_json::from_str(concealed_json).map_err(value_error)?;
    let melds: Vec<Meld> = serde_json::from_str(melds_json).map_err(value_error)?;
    let winning_tile: Tile = serde_json::from_str(winning_tile_json).map_err(value_error)?;
    let context: WinContext = serde_json::from_str(context_json).map_err(value_error)?;
    let rules = match rules_json {
        Some(json) => serde_json::from_str(json).map_err(value_error)?,
        None => RuleConfig::default(),
    };
    let evaluation =
        score_hand(&concealed, &melds, winning_tile, &context, &rules).map_err(value_error)?;
    serde_json::to_string(&evaluation).map_err(value_error)
}

/// Replay an event array from an empty observation state.
#[pyfunction]
#[pyo3(signature = (observer, events_json, rules_json=None))]
fn replay_events_json(
    observer: u8,
    events_json: &str,
    rules_json: Option<&str>,
) -> PyResult<String> {
    let observer = PlayerId::new(observer).map_err(value_error)?;
    let rules = match rules_json {
        Some(json) => serde_json::from_str(json).map_err(value_error)?,
        None => RuleConfig::default(),
    };
    let events: Vec<GameEvent> = serde_json::from_str(events_json).map_err(value_error)?;
    let state = replay_events(observer, rules, events).map_err(value_error)?;
    game_state_to_json(&state).map_err(value_error)
}

/// Stateful observation replay for bulk dataset conversion.
///
/// Keeping the Rust GameState alive avoids parsing and serializing the growing
/// event history after every event. JSON is emitted only at decision points.
#[pyclass(name = "ObservationReplayer")]
struct PyObservationReplayer {
    inner: GameState,
}

#[pymethods]
impl PyObservationReplayer {
    #[new]
    #[pyo3(signature = (observer, rules_json=None))]
    fn new(observer: u8, rules_json: Option<&str>) -> PyResult<Self> {
        let observer = PlayerId::new(observer).map_err(value_error)?;
        let rules = match rules_json {
            Some(json) => serde_json::from_str(json).map_err(value_error)?,
            None => RuleConfig::default(),
        };
        Ok(Self {
            inner: replay_events(observer, rules, []).map_err(value_error)?,
        })
    }

    fn state_json(&self) -> PyResult<String> {
        game_state_to_json(&self.inner).map_err(value_error)
    }

    fn apply_event_json(&mut self, event_json: &str) -> PyResult<()> {
        let event: GameEvent = serde_json::from_str(event_json).map_err(value_error)?;
        self.inner.apply_event(event).map_err(value_error)
    }

    fn legal_actions_json(&self) -> PyResult<String> {
        let report = legal_actions(&self.inner).map_err(value_error)?;
        serde_json::to_string(&report.actions).map_err(value_error)
    }

    fn next_event_sequence(&self) -> u64 {
        self.inner.next_event_sequence
    }
}

#[pyclass(name = "FourPlayerSimulator")]
struct PyFourPlayerSimulator {
    inner: FourPlayerSimulator,
}

#[pymethods]
impl PyFourPlayerSimulator {
    #[new]
    #[pyo3(signature = (seed, round_json, scores, rules_json=None))]
    fn new(
        seed: u64,
        round_json: &str,
        scores: [i32; 4],
        rules_json: Option<&str>,
    ) -> PyResult<Self> {
        let round: RoundState = serde_json::from_str(round_json).map_err(value_error)?;
        let rules = match rules_json {
            Some(json) => serde_json::from_str(json).map_err(value_error)?,
            None => RuleConfig::default(),
        };
        Ok(Self {
            inner: FourPlayerSimulator::new(seed, rules, round, scores).map_err(value_error)?,
        })
    }

    fn observation_json(&self, player: u8) -> PyResult<String> {
        let player = PlayerId::new(player).map_err(value_error)?;
        game_state_to_json(self.inner.observation(player)).map_err(value_error)
    }

    fn legal_actions_json(&self, player: u8) -> PyResult<String> {
        let player = PlayerId::new(player).map_err(value_error)?;
        let actions = self.inner.legal_actions(player).map_err(value_error)?;
        serde_json::to_string(&actions).map_err(value_error)
    }

    fn draw(&mut self) -> PyResult<()> {
        self.inner.draw().map_err(value_error)
    }

    fn act_json(&mut self, player: u8, action_json: &str) -> PyResult<()> {
        let player = PlayerId::new(player).map_err(value_error)?;
        let action: Action = serde_json::from_str(action_json).map_err(value_error)?;
        self.inner.act(player, action).map_err(value_error)
    }

    fn resolve_responses_json(&mut self, choices_json: &str) -> PyResult<()> {
        let raw: Vec<(u8, Action)> = serde_json::from_str(choices_json).map_err(value_error)?;
        let choices = raw
            .into_iter()
            .map(|(player, action)| Ok((PlayerId::new(player)?, action)))
            .collect::<mahjong_core::Result<Vec<_>>>()
            .map_err(value_error)?;
        self.inner.resolve_responses(&choices).map_err(value_error)
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_state_json, module)?)?;
    module.add_function(wrap_pyfunction!(apply_event_json, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_state_json, module)?)?;
    module.add_function(wrap_pyfunction!(legal_actions_json, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_shanten_json, module)?)?;
    module.add_function(wrap_pyfunction!(score_hand_json, module)?)?;
    module.add_function(wrap_pyfunction!(replay_events_json, module)?)?;
    module.add_class::<PyObservationReplayer>()?;
    module.add_class::<PyFourPlayerSimulator>()?;
    Ok(())
}
