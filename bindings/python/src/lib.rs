use mahjong_core::{
    GameEvent, PlayerId, RuleConfig, game_state_from_json, game_state_to_json, replay_events,
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

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_state_json, module)?)?;
    module.add_function(wrap_pyfunction!(apply_event_json, module)?)?;
    module.add_function(wrap_pyfunction!(replay_events_json, module)?)?;
    Ok(())
}
