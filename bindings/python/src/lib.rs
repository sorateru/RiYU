use mahjong_core::{
    Action, FourPlayerSimulator, GameEvent, PlayerId, RoundState, RuleConfig, TileCounts,
    analyze_state, calculate_shanten, game_state_from_json, game_state_to_json, replay_events,
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

/// Calculate exact standard/chiitoitsu/kokushi shanten from a 34-count array.
#[pyfunction]
#[pyo3(signature = (counts_json, fixed_melds=0))]
fn calculate_shanten_json(counts_json: &str, fixed_melds: u8) -> PyResult<String> {
    let counts: TileCounts = serde_json::from_str(counts_json).map_err(value_error)?;
    let result = calculate_shanten(&counts, fixed_melds).map_err(value_error)?;
    serde_json::to_string(&result).map_err(value_error)
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
    module.add_function(wrap_pyfunction!(calculate_shanten_json, module)?)?;
    module.add_function(wrap_pyfunction!(replay_events_json, module)?)?;
    module.add_class::<PyFourPlayerSimulator>()?;
    Ok(())
}
