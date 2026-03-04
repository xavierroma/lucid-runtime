use std::{collections::HashMap, time::Duration};

use tokio::time::Instant;
use uuid::Uuid;

use crate::models::{Session, SessionState};

pub const WORKER_DISCONNECTED_ERROR: &str = "WORKER_DISCONNECTED";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerState {
    Idle,
    Busy,
}

#[derive(Clone, Debug)]
pub struct WorkerAssignment {
    pub session_id: Uuid,
    pub room_name: String,
    pub worker_access_token: String,
}

#[derive(Clone, Debug)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub state: WorkerState,
    pub registered: bool,
    pub last_heartbeat: Option<Instant>,
    pub pending_assignment: Option<WorkerAssignment>,
}

#[derive(Clone, Debug)]
struct SessionRecord {
    session: Session,
    cancel_requested: bool,
}

#[derive(Clone, Debug)]
pub struct RuntimeState {
    sessions: HashMap<Uuid, SessionRecord>,
    active_session: Option<Uuid>,
    pub worker: WorkerInfo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerOperationError {
    UnknownWorker,
    WorkerUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionTransitionError {
    NotFound,
    InvalidState,
}

impl RuntimeState {
    pub fn new(worker_id: String) -> Self {
        Self {
            sessions: HashMap::new(),
            active_session: None,
            worker: WorkerInfo {
                worker_id,
                state: WorkerState::Idle,
                registered: false,
                last_heartbeat: None,
                pending_assignment: None,
            },
        }
    }

    pub fn room_name_for(session_id: Uuid) -> String {
        format!("wm-{session_id}")
    }

    pub fn create_assigned_session(
        &mut self,
        session_id: Uuid,
        worker_access_token: String,
    ) -> Session {
        let room_name = Self::room_name_for(session_id);

        let mut session = Session {
            session_id,
            room_name: room_name.clone(),
            state: SessionState::Created,
            error_code: None,
        };
        session.state = SessionState::Assigned;

        self.sessions.insert(
            session_id,
            SessionRecord {
                session: session.clone(),
                cancel_requested: false,
            },
        );
        self.active_session = Some(session_id);
        self.worker.state = WorkerState::Busy;
        self.worker.pending_assignment = Some(WorkerAssignment {
            session_id,
            room_name,
            worker_access_token,
        });

        session
    }

    pub fn can_create_session(&self, now: Instant, heartbeat_ttl: Duration) -> bool {
        self.active_session.is_none()
            && self.worker.state == WorkerState::Idle
            && self.worker.registered
            && self.worker_is_live(now, heartbeat_ttl)
    }

    pub fn get_session(&self, session_id: &Uuid) -> Option<Session> {
        self.sessions
            .get(session_id)
            .map(|record| record.session.clone())
    }

    pub fn request_end_session(&mut self, session_id: &Uuid) -> bool {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return false;
        };
        if record.session.state != SessionState::Ended {
            record.session.state = SessionState::Ended;
        }
        record.cancel_requested = true;

        let pending_assignment_for_session = self
            .worker
            .pending_assignment
            .as_ref()
            .map(|assignment| assignment.session_id == *session_id)
            .unwrap_or(false);
        if pending_assignment_for_session {
            self.worker.pending_assignment = None;
            self.active_session = None;
            self.worker.state = WorkerState::Idle;
            record.cancel_requested = false;
        }

        true
    }

    pub fn end_session(&mut self, session_id: &Uuid, error_code: Option<String>) -> bool {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return false;
        };

        if record.session.state != SessionState::Ended {
            record.session.state = SessionState::Ended;
            record.session.error_code = error_code;
        } else if record.session.error_code.is_none() {
            record.session.error_code = error_code;
        }
        record.cancel_requested = false;

        if self
            .worker
            .pending_assignment
            .as_ref()
            .map(|assignment| assignment.session_id == *session_id)
            .unwrap_or(false)
        {
            self.worker.pending_assignment = None;
        }

        if self.active_session == Some(*session_id) {
            self.active_session = None;
        }

        self.worker.state = WorkerState::Idle;
        true
    }

    pub fn register_worker(
        &mut self,
        worker_id: &str,
        now: Instant,
    ) -> Result<(), WorkerOperationError> {
        if !self.worker_id_matches(worker_id) {
            return Err(WorkerOperationError::UnknownWorker);
        }

        self.worker.registered = true;
        self.worker.last_heartbeat = Some(now);
        self.worker.state = if self.active_session.is_some() {
            WorkerState::Busy
        } else {
            WorkerState::Idle
        };

        Ok(())
    }

    pub fn heartbeat_worker(
        &mut self,
        worker_id: &str,
        now: Instant,
    ) -> Result<bool, WorkerOperationError> {
        if !self.worker_id_matches(worker_id) {
            return Err(WorkerOperationError::UnknownWorker);
        }

        self.worker.registered = true;
        self.worker.last_heartbeat = Some(now);

        let cancel_active_session = self
            .active_session
            .and_then(|session_id| self.sessions.get(&session_id))
            .map(|record| record.cancel_requested)
            .unwrap_or(false);
        Ok(cancel_active_session)
    }

    pub fn take_assignment(
        &mut self,
        worker_id: &str,
        now: Instant,
        heartbeat_ttl: Duration,
    ) -> Result<Option<WorkerAssignment>, WorkerOperationError> {
        if !self.worker_id_matches(worker_id) {
            return Err(WorkerOperationError::UnknownWorker);
        }

        self.expire_stale_worker(now, heartbeat_ttl);
        if !self.worker.registered || !self.worker_is_live(now, heartbeat_ttl) {
            return Err(WorkerOperationError::WorkerUnavailable);
        }

        Ok(self.worker.pending_assignment.take())
    }

    pub fn mark_running(&mut self, session_id: &Uuid) -> Result<(), SessionTransitionError> {
        let Some(session) = self.sessions.get_mut(session_id) else {
            return Err(SessionTransitionError::NotFound);
        };

        if self.active_session != Some(*session_id)
            || session.session.state != SessionState::Assigned
        {
            return Err(SessionTransitionError::InvalidState);
        }

        session.session.state = SessionState::Running;
        Ok(())
    }

    pub fn expire_stale_worker(&mut self, now: Instant, heartbeat_ttl: Duration) {
        if !self.worker.registered {
            return;
        }

        let Some(last_heartbeat) = self.worker.last_heartbeat else {
            return;
        };

        if now.duration_since(last_heartbeat) <= heartbeat_ttl {
            return;
        }

        self.worker.registered = false;
        self.worker.last_heartbeat = None;
        self.worker.pending_assignment = None;
        self.worker.state = WorkerState::Idle;

        if let Some(session_id) = self.active_session.take() {
            if let Some(record) = self.sessions.get_mut(&session_id) {
                record.session.state = SessionState::Ended;
                record.session.error_code = Some(WORKER_DISCONNECTED_ERROR.to_string());
                record.cancel_requested = false;
            }
        }
    }

    fn worker_is_live(&self, now: Instant, heartbeat_ttl: Duration) -> bool {
        self.worker
            .last_heartbeat
            .map(|last| now.duration_since(last) <= heartbeat_ttl)
            .unwrap_or(false)
    }

    fn worker_id_matches(&self, worker_id: &str) -> bool {
        self.worker.worker_id == worker_id
    }
}

#[cfg(test)]
mod tests {
    use tokio::time::{advance, Duration, Instant};

    use super::*;

    #[test]
    fn room_name_is_deterministic() {
        let session_id =
            Uuid::parse_str("4f3a4031-3b33-44c2-856a-d450980ec8cb").expect("valid uuid");
        assert_eq!(
            RuntimeState::room_name_for(session_id),
            "wm-4f3a4031-3b33-44c2-856a-d450980ec8cb"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn stale_worker_ends_active_session() {
        let mut state = RuntimeState::new("wm-worker-1".to_string());

        state
            .register_worker("wm-worker-1", Instant::now())
            .expect("worker should register");

        let session_id = Uuid::new_v4();
        state.create_assigned_session(session_id, "worker-token".to_string());

        advance(Duration::from_secs(20)).await;
        state.expire_stale_worker(Instant::now(), Duration::from_secs(15));

        let session = state
            .get_session(&session_id)
            .expect("session should still exist");

        assert_eq!(session.state, SessionState::Ended);
        assert_eq!(
            session.error_code,
            Some(WORKER_DISCONNECTED_ERROR.to_string())
        );
        assert!(!state.worker.registered);
        assert_eq!(state.worker.state, WorkerState::Idle);
    }
}
