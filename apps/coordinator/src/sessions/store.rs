use std::{collections::HashMap, sync::Mutex};

use tokio::time::Instant;
use uuid::Uuid;

use crate::models::{Session, SessionEndReason, SessionState};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PendingTerminal {
    pub state: SessionState,
    pub end_reason: SessionEndReason,
    pub error_code: Option<String>,
}

impl PendingTerminal {
    pub fn ended(end_reason: SessionEndReason) -> Self {
        Self {
            state: SessionState::Ended,
            end_reason,
            error_code: None,
        }
    }

    pub fn failed(error_code: &str, end_reason: SessionEndReason) -> Self {
        Self {
            state: SessionState::Failed,
            end_reason,
            error_code: Some(error_code.to_string()),
        }
    }
}

/// Rich session view used by the reconciler and internal handlers.
#[derive(Clone, Debug)]
pub struct SessionRecord {
    pub session_id: Uuid,
    pub room_name: String,
    pub model_name: String,
    pub function_call_id: String,
    pub state: SessionState,
    pub error_code: Option<String>,
    pub end_reason: Option<SessionEndReason>,
    pub created_at: Instant,
    pub ready_at: Option<Instant>,
    pub running_at: Option<Instant>,
    pub last_heartbeat_at: Option<Instant>,
    pub cancel_requested_at: Option<Instant>,
    pub cancel_dispatched_at: Option<Instant>,
    pub force_cancel_dispatched_at: Option<Instant>,
    pub pending_terminal: Option<PendingTerminal>,
}

impl SessionRecord {
    pub fn to_session(&self) -> Session {
        Session {
            session_id: self.session_id,
            room_name: self.room_name.clone(),
            model_name: self.model_name.clone(),
            state: self.state.clone(),
            error_code: self.error_code.clone(),
            end_reason: self.end_reason.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Update types
// ---------------------------------------------------------------------------

/// Partial update applied to a session by calling [`SessionStore::update`].
///
/// All fields are `Option`; `None` means "leave unchanged".
/// Fields of type `Option<Option<T>>` use `Some(None)` to explicitly clear a
/// value and `Some(Some(v))` to set it.
#[derive(Clone, Debug, Default)]
pub struct SessionUpdate {
    pub state: Option<SessionState>,
    /// `Some(None)` clears the error code.
    pub error_code: Option<Option<String>>,
    pub end_reason: Option<SessionEndReason>,
    /// Set `ready_at` once (ignored if already set).
    pub ready_at: Option<Instant>,
    /// Set `running_at` once (ignored if already set).
    pub running_at: Option<Instant>,
    pub last_heartbeat_at: Option<Instant>,
    pub cancel_requested_at: Option<Instant>,
    /// `Some(None)` clears.
    pub cancel_dispatched_at: Option<Option<Instant>>,
    /// `Some(None)` clears.
    pub force_cancel_dispatched_at: Option<Option<Instant>>,
    /// `Some(None)` clears.
    pub pending_terminal: Option<Option<PendingTerminal>>,
}

impl SessionUpdate {
    /// Build an update that transitions a session to `Canceling`.
    pub fn begin_canceling(disposition: PendingTerminal, now: Instant) -> Self {
        Self {
            state: Some(SessionState::Canceling),
            end_reason: Some(disposition.end_reason.clone()),
            cancel_requested_at: Some(now),
            cancel_dispatched_at: Some(None),
            force_cancel_dispatched_at: Some(None),
            pending_terminal: Some(Some(disposition)),
            ..Default::default()
        }
    }

    /// Build an update that moves a session to a terminal state, clearing all
    /// transient timing fields.
    pub fn finish(
        state: SessionState,
        error_code: Option<String>,
        end_reason: Option<SessionEndReason>,
    ) -> Self {
        Self {
            state: Some(state),
            error_code: Some(error_code),
            end_reason,
            cancel_dispatched_at: Some(None),
            force_cancel_dispatched_at: Some(None),
            pending_terminal: Some(None),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdateError {
    NotFound,
    InvalidTransition,
}

// ---------------------------------------------------------------------------
// Store trait
// ---------------------------------------------------------------------------

pub trait SessionStore: Send + Sync {
    fn create(
        &self,
        id: Uuid,
        model_name: String,
        room_name: String,
        function_call_id: String,
        now: Instant,
    ) -> Session;

    fn get(&self, id: &Uuid) -> Option<Session>;
    fn get_record(&self, id: &Uuid) -> Option<SessionRecord>;

    fn update(&self, id: &Uuid, update: SessionUpdate) -> Result<Session, UpdateError>;

    fn list_non_terminal(&self) -> Vec<SessionRecord>;
}

// ---------------------------------------------------------------------------
// In-memory implementation
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct InMemorySessionStore {
    sessions: Mutex<HashMap<Uuid, SessionRecord>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionStore for InMemorySessionStore {
    fn create(
        &self,
        id: Uuid,
        model_name: String,
        room_name: String,
        function_call_id: String,
        now: Instant,
    ) -> Session {
        let record = SessionRecord {
            session_id: id,
            room_name,
            model_name,
            function_call_id,
            state: SessionState::Starting,
            error_code: None,
            end_reason: None,
            created_at: now,
            ready_at: None,
            running_at: None,
            last_heartbeat_at: None,
            cancel_requested_at: None,
            cancel_dispatched_at: None,
            force_cancel_dispatched_at: None,
            pending_terminal: None,
        };
        let session = record.to_session();
        self.sessions.lock().unwrap().insert(id, record);
        session
    }

    fn get(&self, id: &Uuid) -> Option<Session> {
        self.sessions
            .lock()
            .unwrap()
            .get(id)
            .map(SessionRecord::to_session)
    }

    fn get_record(&self, id: &Uuid) -> Option<SessionRecord> {
        self.sessions.lock().unwrap().get(id).cloned()
    }

    fn update(&self, id: &Uuid, update: SessionUpdate) -> Result<Session, UpdateError> {
        let mut sessions = self.sessions.lock().unwrap();
        let record = sessions.get_mut(id).ok_or(UpdateError::NotFound)?;

        // Validate the requested state transition.
        if let Some(ref new_state) = update.state {
            if !is_valid_transition(&record.state, new_state) {
                return Err(UpdateError::InvalidTransition);
            }
        }

        // Apply state change.
        if let Some(new_state) = update.state {
            let going_terminal = new_state.is_terminal();
            record.state = new_state;

            // Entering a terminal state clears all transient timing fields.
            if going_terminal {
                record.ready_at = None;
                record.running_at = None;
                record.last_heartbeat_at = None;
                record.cancel_requested_at = None;
                record.cancel_dispatched_at = None;
                record.force_cancel_dispatched_at = None;
                record.pending_terminal = None;
            }
        }

        // Apply remaining fields (skipped when the terminal-state clear above already set them).
        if let Some(error_code) = update.error_code {
            record.error_code = error_code;
        }
        if let Some(end_reason) = update.end_reason {
            record.end_reason = Some(end_reason);
        }
        if let Some(ready_at) = update.ready_at {
            if record.ready_at.is_none() {
                record.ready_at = Some(ready_at);
            }
        }
        if let Some(running_at) = update.running_at {
            if record.running_at.is_none() {
                record.running_at = Some(running_at);
            }
        }
        if let Some(last_heartbeat_at) = update.last_heartbeat_at {
            record.last_heartbeat_at = Some(last_heartbeat_at);
        }
        if let Some(cancel_requested_at) = update.cancel_requested_at {
            record.cancel_requested_at = Some(cancel_requested_at);
        }
        if let Some(cancel_dispatched_at) = update.cancel_dispatched_at {
            record.cancel_dispatched_at = cancel_dispatched_at;
        }
        if let Some(force_cancel_dispatched_at) = update.force_cancel_dispatched_at {
            record.force_cancel_dispatched_at = force_cancel_dispatched_at;
        }
        if let Some(pending_terminal) = update.pending_terminal {
            record.pending_terminal = pending_terminal;
        }

        Ok(record.to_session())
    }

    fn list_non_terminal(&self) -> Vec<SessionRecord> {
        let mut records: Vec<SessionRecord> = self
            .sessions
            .lock()
            .unwrap()
            .values()
            .filter(|r| !r.state.is_terminal())
            .cloned()
            .collect();
        // Deterministic order for tests.
        records.sort_by(|a, b| a.function_call_id.cmp(&b.function_call_id));
        records
    }
}

fn is_valid_transition(from: &SessionState, to: &SessionState) -> bool {
    match (from, to) {
        // Nothing leaves a terminal state.
        (SessionState::Ended, _) | (SessionState::Failed, _) => false,
        // Canceling may only go terminal.
        (SessionState::Canceling, SessionState::Ended | SessionState::Failed) => true,
        (SessionState::Canceling, _) => false,
        // Normal forward transitions.
        (SessionState::Starting, SessionState::Ready | SessionState::Canceling) => true,
        (SessionState::Ready, SessionState::Running | SessionState::Canceling) => true,
        (SessionState::Running, SessionState::Paused | SessionState::Canceling) => true,
        (SessionState::Paused, SessionState::Running | SessionState::Canceling) => true,
        // Any non-terminal state may be forced directly to a terminal state (e.g. reconciler).
        (_, SessionState::Ended | SessionState::Failed) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn room_name_for(session_id: Uuid) -> String {
    format!("wm-{session_id}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use tokio::time::{advance, Duration, Instant};

    use super::*;
    use crate::{
        modal_dispatch::ModalExecutionStatus,
        sessions::reconciler::{reconcile, ReconcileCommand},
    };
    use crate::registry::ModelTimeouts;

    fn timeouts() -> ModelTimeouts {
        ModelTimeouts {
            startup_timeout_secs: 120,
            session_max_duration_secs: 3600,
            session_cancel_grace_secs: 30,
            worker_heartbeat_timeout_secs: 15,
        }
    }

    #[test]
    fn room_name_is_deterministic() {
        let session_id =
            Uuid::parse_str("4f3a4031-3b33-44c2-856a-d450980ec8cb").expect("valid uuid");
        assert_eq!(room_name_for(session_id), "wm-4f3a4031-3b33-44c2-856a-d450980ec8cb");
    }

    #[test]
    fn create_and_get_round_trips() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        let session = store.create(
            id,
            "yume".to_string(),
            room_name_for(id),
            "call-1".to_string(),
            Instant::now(),
        );
        assert_eq!(session.state, SessionState::Starting);
        assert_eq!(store.get(&id).expect("session exists").state, SessionState::Starting);
    }

    #[test]
    fn ready_must_precede_running() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        let now = Instant::now();
        store.create(id, "yume".to_string(), room_name_for(id), "call-1".to_string(), now);

        assert_eq!(
            store.update(&id, SessionUpdate { state: Some(SessionState::Running), ..Default::default() }),
            Err(UpdateError::InvalidTransition)
        );
        store.update(&id, SessionUpdate { state: Some(SessionState::Ready), ..Default::default() })
            .expect("ready transition ok");
        store.update(&id, SessionUpdate { state: Some(SessionState::Running), ..Default::default() })
            .expect("running transition ok");

        assert_eq!(store.get(&id).unwrap().state, SessionState::Running);
    }

    #[test]
    fn running_session_can_pause_and_resume() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        let now = Instant::now();
        store.create(id, "yume".to_string(), room_name_for(id), "call-1".to_string(), now);

        store.update(&id, SessionUpdate { state: Some(SessionState::Ready), ..Default::default() }).unwrap();
        store.update(&id, SessionUpdate { state: Some(SessionState::Running), ..Default::default() }).unwrap();
        store.update(&id, SessionUpdate { state: Some(SessionState::Paused), ..Default::default() }).unwrap();
        store.update(&id, SessionUpdate { state: Some(SessionState::Running), ..Default::default() }).unwrap();

        assert_eq!(store.get(&id).unwrap().state, SessionState::Running);
    }

    #[test]
    fn pause_requires_running_state() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        let now = Instant::now();
        store.create(id, "yume".to_string(), room_name_for(id), "call-1".to_string(), now);

        assert_eq!(
            store.update(&id, SessionUpdate { state: Some(SessionState::Paused), ..Default::default() }),
            Err(UpdateError::InvalidTransition)
        );
        store.update(&id, SessionUpdate { state: Some(SessionState::Ready), ..Default::default() }).unwrap();
        assert_eq!(
            store.update(&id, SessionUpdate { state: Some(SessionState::Paused), ..Default::default() }),
            Err(UpdateError::InvalidTransition)
        );
    }

    #[test]
    fn multiple_sessions_transition_independently() {
        let store = InMemorySessionStore::new();
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let now = Instant::now();
        store.create(first, "yume".to_string(), room_name_for(first), "call-1".to_string(), now);
        store.create(second, "waypoint".to_string(), room_name_for(second), "call-2".to_string(), now);

        store.update(&first, SessionUpdate { state: Some(SessionState::Ready), ..Default::default() }).unwrap();
        store.update(&first, SessionUpdate { state: Some(SessionState::Running), ..Default::default() }).unwrap();
        store.update(&second, SessionUpdate { state: Some(SessionState::Ready), ..Default::default() }).unwrap();

        assert_eq!(store.get(&first).unwrap().state, SessionState::Running);
        assert_eq!(store.get(&second).unwrap().state, SessionState::Ready);
    }

    #[test]
    fn non_terminal_list_excludes_finished_sessions() {
        let store = InMemorySessionStore::new();
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let now = Instant::now();
        store.create(first, "yume".to_string(), room_name_for(first), "call-1".to_string(), now);
        store.create(second, "waypoint".to_string(), room_name_for(second), "call-2".to_string(), now);

        store.update(&first, SessionUpdate::finish(SessionState::Ended, None, None)).unwrap();

        let non_terminal = store.list_non_terminal();
        assert_eq!(non_terminal.len(), 1);
        assert_eq!(non_terminal[0].session_id, second);
    }

    #[tokio::test(start_paused = true)]
    async fn startup_timeout_enters_canceling_and_uses_timeout_failure_on_termination() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        store.create(id, "yume".to_string(), room_name_for(id), "call-1".to_string(), Instant::now());

        advance(Duration::from_secs(121)).await;
        let record = store.get_record(&id).unwrap();
        let (update, commands) = reconcile(&record, Some(ModalExecutionStatus::Pending), Instant::now(), &timeouts());
        assert_eq!(
            commands,
            vec![ReconcileCommand::CancelModal {
                session_id: id,
                model_name: "yume".to_string(),
                function_call_id: "call-1".to_string(),
                force: false,
            }]
        );
        store.update(&id, update.unwrap()).unwrap();

        let session = store.get(&id).unwrap();
        assert_eq!(session.state, SessionState::Canceling);
        assert_eq!(session.end_reason, Some(SessionEndReason::StartupTimeout));

        // Simulate cancel dispatched.
        store.update(&id, SessionUpdate { cancel_dispatched_at: Some(Some(Instant::now())), ..Default::default() }).unwrap();

        let record = store.get_record(&id).unwrap();
        let (update, commands) = reconcile(&record, Some(ModalExecutionStatus::Terminated), Instant::now(), &timeouts());
        assert!(commands.is_empty());
        store.update(&id, update.unwrap()).unwrap();

        let session = store.get(&id).unwrap();
        assert_eq!(session.state, SessionState::Failed);
        assert_eq!(session.error_code.as_deref(), Some("STARTUP_TIMEOUT"));
        assert_eq!(session.end_reason, Some(SessionEndReason::StartupTimeout));
    }

    #[tokio::test(start_paused = true)]
    async fn ready_session_times_out_before_model_start() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        store.create(id, "yume".to_string(), room_name_for(id), "call-1".to_string(), Instant::now());
        store.update(&id, SessionUpdate { state: Some(SessionState::Ready), ready_at: Some(Instant::now()), ..Default::default() }).unwrap();

        advance(Duration::from_secs(3601)).await;
        let record = store.get_record(&id).unwrap();
        let (update, commands) = reconcile(&record, Some(ModalExecutionStatus::Pending), Instant::now(), &timeouts());
        assert_eq!(
            commands,
            vec![ReconcileCommand::CancelModal {
                session_id: id,
                model_name: "yume".to_string(),
                function_call_id: "call-1".to_string(),
                force: false,
            }]
        );
        store.update(&id, update.unwrap()).unwrap();

        let session = store.get(&id).unwrap();
        assert_eq!(session.state, SessionState::Canceling);
        assert_eq!(session.end_reason, Some(SessionEndReason::SessionTimeout));
    }

    #[tokio::test(start_paused = true)]
    async fn reconcile_only_affects_target_session() {
        let store = InMemorySessionStore::new();
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let now = Instant::now();
        store.create(first, "yume".to_string(), room_name_for(first), "call-1".to_string(), now);
        store.create(second, "waypoint".to_string(), room_name_for(second), "call-2".to_string(), now);

        advance(Duration::from_secs(121)).await;
        let record = store.get_record(&first).unwrap();
        let (update, commands) = reconcile(&record, Some(ModalExecutionStatus::Pending), Instant::now(), &timeouts());
        assert_eq!(
            commands,
            vec![ReconcileCommand::CancelModal {
                session_id: first,
                model_name: "yume".to_string(),
                function_call_id: "call-1".to_string(),
                force: false,
            }]
        );
        store.update(&first, update.unwrap()).unwrap();

        assert_eq!(store.get(&first).unwrap().state, SessionState::Canceling);
        assert_eq!(store.get(&second).unwrap().state, SessionState::Starting);
    }

    #[tokio::test(start_paused = true)]
    async fn paused_session_times_out_after_max_duration() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        let now = Instant::now();
        store.create(id, "yume".to_string(), room_name_for(id), "call-1".to_string(), now);
        store.update(&id, SessionUpdate { state: Some(SessionState::Ready), ready_at: Some(now), last_heartbeat_at: Some(now), ..Default::default() }).unwrap();
        store.update(&id, SessionUpdate { state: Some(SessionState::Running), running_at: Some(now), last_heartbeat_at: Some(now), ..Default::default() }).unwrap();
        store.update(&id, SessionUpdate { state: Some(SessionState::Paused), last_heartbeat_at: Some(now), ..Default::default() }).unwrap();

        advance(Duration::from_secs(3601)).await;
        let record = store.get_record(&id).unwrap();
        let (update, commands) = reconcile(&record, Some(ModalExecutionStatus::Pending), Instant::now(), &timeouts());
        assert_eq!(
            commands,
            vec![ReconcileCommand::CancelModal {
                session_id: id,
                model_name: "yume".to_string(),
                function_call_id: "call-1".to_string(),
                force: false,
            }]
        );
        store.update(&id, update.unwrap()).unwrap();

        let session = store.get(&id).unwrap();
        assert_eq!(session.state, SessionState::Canceling);
        assert_eq!(session.end_reason, Some(SessionEndReason::SessionTimeout));
    }

    #[tokio::test(start_paused = true)]
    async fn paused_session_times_out_when_heartbeat_stops() {
        let store = InMemorySessionStore::new();
        let id = Uuid::new_v4();
        let now = Instant::now();
        store.create(id, "yume".to_string(), room_name_for(id), "call-1".to_string(), now);
        store.update(&id, SessionUpdate { state: Some(SessionState::Ready), ready_at: Some(now), last_heartbeat_at: Some(now), ..Default::default() }).unwrap();
        store.update(&id, SessionUpdate { state: Some(SessionState::Running), running_at: Some(now), last_heartbeat_at: Some(now), ..Default::default() }).unwrap();
        store.update(&id, SessionUpdate { state: Some(SessionState::Paused), last_heartbeat_at: Some(now), ..Default::default() }).unwrap();

        advance(Duration::from_secs(16)).await;
        let record = store.get_record(&id).unwrap();
        let (update, commands) = reconcile(&record, Some(ModalExecutionStatus::Pending), Instant::now(), &timeouts());
        assert_eq!(
            commands,
            vec![ReconcileCommand::CancelModal {
                session_id: id,
                model_name: "yume".to_string(),
                function_call_id: "call-1".to_string(),
                force: false,
            }]
        );
        store.update(&id, update.unwrap()).unwrap();

        let session = store.get(&id).unwrap();
        assert_eq!(session.state, SessionState::Canceling);
        assert_eq!(session.end_reason, Some(SessionEndReason::WorkerHeartbeatTimeout));
    }
}
