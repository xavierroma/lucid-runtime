use std::{collections::HashMap, time::Duration};

use tokio::time::Instant;
use uuid::Uuid;

use crate::models::{Session, SessionState};

pub const STARTUP_TIMEOUT_ERROR: &str = "STARTUP_TIMEOUT";
pub const SESSION_TIMEOUT_ERROR: &str = "SESSION_TIMEOUT";
pub const CANCEL_TIMEOUT_ERROR: &str = "CANCEL_TIMEOUT";

#[derive(Clone, Debug)]
struct SessionRecord {
    session: Session,
    modal_function_call_id: String,
    created_at: Instant,
    running_at: Option<Instant>,
    cancel_requested: bool,
    cancel_requested_at: Option<Instant>,
}

#[derive(Clone, Debug)]
pub struct RuntimeState {
    sessions: HashMap<Uuid, SessionRecord>,
    active_session: Option<Uuid>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionTransitionError {
    NotFound,
    InvalidState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EndRequestError {
    NotFound,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndRequestResult {
    pub function_call_id: Option<String>,
}

impl RuntimeState {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            active_session: None,
        }
    }

    pub fn room_name_for(session_id: Uuid) -> String {
        format!("wm-{session_id}")
    }

    pub fn can_create_session(&self) -> bool {
        self.active_session.is_none()
    }

    pub fn create_session(
        &mut self,
        session_id: Uuid,
        modal_function_call_id: String,
        now: Instant,
    ) -> Session {
        let session = Session {
            session_id,
            room_name: Self::room_name_for(session_id),
            state: SessionState::Created,
            error_code: None,
        };
        self.sessions.insert(
            session_id,
            SessionRecord {
                session: session.clone(),
                modal_function_call_id,
                created_at: now,
                running_at: None,
                cancel_requested: false,
                cancel_requested_at: None,
            },
        );
        self.active_session = Some(session_id);
        session
    }

    pub fn get_session(&self, session_id: &Uuid) -> Option<Session> {
        self.sessions
            .get(session_id)
            .map(|record| record.session.clone())
    }

    pub fn request_end_session(
        &mut self,
        session_id: &Uuid,
        now: Instant,
    ) -> Result<EndRequestResult, EndRequestError> {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return Err(EndRequestError::NotFound);
        };
        if record.session.state == SessionState::Ended || record.cancel_requested {
            return Ok(EndRequestResult {
                function_call_id: None,
            });
        }

        record.cancel_requested = true;
        record.cancel_requested_at = Some(now);
        Ok(EndRequestResult {
            function_call_id: Some(record.modal_function_call_id.clone()),
        })
    }

    pub fn mark_running(
        &mut self,
        session_id: &Uuid,
        now: Instant,
    ) -> Result<(), SessionTransitionError> {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return Err(SessionTransitionError::NotFound);
        };
        if self.active_session != Some(*session_id) || record.session.state != SessionState::Created
        {
            return Err(SessionTransitionError::InvalidState);
        }
        record.session.state = SessionState::Running;
        record.running_at = Some(now);
        Ok(())
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
        record.cancel_requested_at = None;
        if self.active_session == Some(*session_id) {
            self.active_session = None;
        }
        true
    }

    pub fn reconcile_timeouts(
        &mut self,
        now: Instant,
        startup_timeout: Duration,
        max_duration: Duration,
        cancel_grace: Duration,
    ) {
        let Some(active_session_id) = self.active_session else {
            return;
        };

        let Some(record) = self.sessions.get(&active_session_id).cloned() else {
            self.active_session = None;
            return;
        };

        let timeout_error = if record.cancel_requested {
            record
                .cancel_requested_at
                .filter(|started| now.duration_since(*started) > cancel_grace)
                .map(|_| CANCEL_TIMEOUT_ERROR.to_string())
        } else if record.session.state == SessionState::Created {
            if now.duration_since(record.created_at) > startup_timeout {
                Some(STARTUP_TIMEOUT_ERROR.to_string())
            } else {
                None
            }
        } else if record.session.state == SessionState::Running {
            record
                .running_at
                .filter(|started| now.duration_since(*started) > max_duration)
                .map(|_| SESSION_TIMEOUT_ERROR.to_string())
        } else {
            None
        };

        if let Some(error_code) = timeout_error {
            let _ = self.end_session(&active_session_id, Some(error_code));
        }
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
    async fn startup_timeout_ends_created_session() {
        let mut state = RuntimeState::new();
        let session_id = Uuid::new_v4();
        state.create_session(session_id, "call-1".to_string(), Instant::now());

        advance(Duration::from_secs(121)).await;
        state.reconcile_timeouts(
            Instant::now(),
            Duration::from_secs(120),
            Duration::from_secs(3600),
            Duration::from_secs(30),
        );

        let session = state
            .get_session(&session_id)
            .expect("session should still exist");
        assert_eq!(session.state, SessionState::Ended);
        assert_eq!(session.error_code, Some(STARTUP_TIMEOUT_ERROR.to_string()));
    }
}
