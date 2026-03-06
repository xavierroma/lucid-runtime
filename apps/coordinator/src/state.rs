use std::{collections::HashMap, time::Duration};

use tokio::time::Instant;
use uuid::Uuid;

use crate::{
    modal_dispatch::ModalExecutionStatus,
    models::{Session, SessionEndReason, SessionState},
};

pub const STARTUP_TIMEOUT_ERROR: &str = "STARTUP_TIMEOUT";
pub const SESSION_TIMEOUT_ERROR: &str = "SESSION_TIMEOUT";
pub const CANCEL_TIMEOUT_ERROR: &str = "CANCEL_TIMEOUT";
pub const WORKER_HEARTBEAT_TIMEOUT_ERROR: &str = "WORKER_HEARTBEAT_TIMEOUT";
pub const MODAL_FAILURE_ERROR: &str = "MODAL_FAILURE";
pub const MODAL_INIT_FAILURE_ERROR: &str = "MODAL_INIT_FAILURE";
pub const MODAL_TERMINATED_ERROR: &str = "MODAL_TERMINATED";
pub const MODAL_TIMEOUT_ERROR: &str = "MODAL_TIMEOUT";
pub const MODAL_NOT_FOUND_ERROR: &str = "MODAL_NOT_FOUND";

#[derive(Clone, Debug)]
struct PendingTerminalDisposition {
    state: SessionState,
    end_reason: SessionEndReason,
    error_code: Option<String>,
}

impl PendingTerminalDisposition {
    fn ended(end_reason: SessionEndReason) -> Self {
        Self {
            state: SessionState::Ended,
            end_reason,
            error_code: None,
        }
    }

    fn failed(error_code: &str, end_reason: SessionEndReason) -> Self {
        Self {
            state: SessionState::Failed,
            end_reason,
            error_code: Some(error_code.to_string()),
        }
    }
}

#[derive(Clone, Debug)]
struct SessionRecord {
    session: Session,
    modal_function_call_id: String,
    created_at: Instant,
    running_at: Option<Instant>,
    last_heartbeat_at: Option<Instant>,
    cancel_requested_at: Option<Instant>,
    cancel_dispatched_at: Option<Instant>,
    force_cancel_dispatched_at: Option<Instant>,
    pending_terminal: Option<PendingTerminalDisposition>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveSessionSnapshot {
    pub session_id: Uuid,
    pub function_call_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconcileCommand {
    CancelModal {
        session_id: Uuid,
        function_call_id: String,
        force: bool,
    },
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
            state: SessionState::Starting,
            error_code: None,
            end_reason: None,
        };
        self.sessions.insert(
            session_id,
            SessionRecord {
                session: session.clone(),
                modal_function_call_id,
                created_at: now,
                running_at: None,
                last_heartbeat_at: None,
                cancel_requested_at: None,
                cancel_dispatched_at: None,
                force_cancel_dispatched_at: None,
                pending_terminal: None,
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

    pub fn active_session_snapshot(&self) -> Option<ActiveSessionSnapshot> {
        let session_id = self.active_session?;
        let record = self.sessions.get(&session_id)?;
        Some(ActiveSessionSnapshot {
            session_id,
            function_call_id: record.modal_function_call_id.clone(),
        })
    }

    pub fn request_end_session(
        &mut self,
        session_id: &Uuid,
        now: Instant,
    ) -> Result<EndRequestResult, EndRequestError> {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return Err(EndRequestError::NotFound);
        };
        if record.session.state.is_terminal() || record.session.state == SessionState::Canceling {
            return Ok(EndRequestResult {
                function_call_id: None,
            });
        }

        Self::begin_canceling(
            record,
            now,
            PendingTerminalDisposition::ended(SessionEndReason::ClientRequested),
        );
        Ok(EndRequestResult {
            function_call_id: Some(record.modal_function_call_id.clone()),
        })
    }

    pub fn mark_cancel_dispatched(&mut self, session_id: &Uuid, now: Instant, force: bool) -> bool {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return false;
        };
        if record.session.state != SessionState::Canceling {
            return false;
        }
        if force {
            record.force_cancel_dispatched_at = Some(now);
        } else {
            record.cancel_dispatched_at = Some(now);
        }
        true
    }

    pub fn mark_heartbeat(
        &mut self,
        session_id: &Uuid,
        now: Instant,
    ) -> Result<(), SessionTransitionError> {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return Err(SessionTransitionError::NotFound);
        };
        if record.session.state.is_terminal() {
            return Err(SessionTransitionError::InvalidState);
        }
        record.last_heartbeat_at = Some(now);
        Ok(())
    }

    pub fn mark_running(
        &mut self,
        session_id: &Uuid,
        now: Instant,
    ) -> Result<(), SessionTransitionError> {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return Err(SessionTransitionError::NotFound);
        };
        if self.active_session != Some(*session_id) {
            return Err(SessionTransitionError::InvalidState);
        }

        match record.session.state {
            SessionState::Starting => {
                record.session.state = SessionState::Running;
            }
            SessionState::Canceling => {}
            _ => return Err(SessionTransitionError::InvalidState),
        }

        if record.running_at.is_none() {
            record.running_at = Some(now);
        }
        record.last_heartbeat_at = Some(now);
        Ok(())
    }

    pub fn mark_ended(
        &mut self,
        session_id: &Uuid,
        error_code: Option<String>,
        end_reason: Option<SessionEndReason>,
    ) -> bool {
        let Some(record) = self.sessions.get(session_id) else {
            return false;
        };

        let pending = record.pending_terminal.clone();
        let (state, final_error_code, final_end_reason) = if let Some(error_code) = error_code {
            (
                SessionState::Failed,
                Some(error_code),
                Some(end_reason.unwrap_or(SessionEndReason::WorkerReportedError)),
            )
        } else if let Some(pending) = pending {
            (
                pending.state,
                pending.error_code,
                Some(end_reason.unwrap_or(pending.end_reason)),
            )
        } else {
            (
                SessionState::Ended,
                None,
                Some(end_reason.unwrap_or(SessionEndReason::NormalCompletion)),
            )
        };

        self.finish_session(session_id, state, final_error_code, final_end_reason)
    }

    pub fn reconcile_active_session(
        &mut self,
        session_id: &Uuid,
        function_call_id: &str,
        modal_status: Option<ModalExecutionStatus>,
        now: Instant,
        startup_timeout: Duration,
        max_duration: Duration,
        cancel_grace: Duration,
        worker_heartbeat_timeout: Duration,
    ) -> Vec<ReconcileCommand> {
        let Some(record) = self.sessions.get(session_id) else {
            return Vec::new();
        };
        if self.active_session != Some(*session_id)
            || record.modal_function_call_id != function_call_id
        {
            return Vec::new();
        }

        match modal_status {
            Some(ModalExecutionStatus::Success) => {
                let pending = record.pending_terminal.clone();
                let state = pending
                    .as_ref()
                    .map(|disposition| disposition.state.clone())
                    .unwrap_or(SessionState::Ended);
                let error_code = pending.and_then(|disposition| disposition.error_code);
                let end_reason = record
                    .pending_terminal
                    .as_ref()
                    .map(|disposition| disposition.end_reason.clone())
                    .unwrap_or(SessionEndReason::ModalSuccess);
                let _ = self.finish_session(session_id, state, error_code, Some(end_reason));
                return Vec::new();
            }
            Some(ModalExecutionStatus::Failure) => {
                let _ = self.finish_session(
                    session_id,
                    SessionState::Failed,
                    Some(MODAL_FAILURE_ERROR.to_string()),
                    Some(SessionEndReason::ModalFailure),
                );
                return Vec::new();
            }
            Some(ModalExecutionStatus::InitFailure) => {
                let _ = self.finish_session(
                    session_id,
                    SessionState::Failed,
                    Some(MODAL_INIT_FAILURE_ERROR.to_string()),
                    Some(SessionEndReason::ModalInitFailure),
                );
                return Vec::new();
            }
            Some(ModalExecutionStatus::Timeout) => {
                let _ = self.finish_session(
                    session_id,
                    SessionState::Failed,
                    Some(MODAL_TIMEOUT_ERROR.to_string()),
                    Some(SessionEndReason::ModalTimeout),
                );
                return Vec::new();
            }
            Some(ModalExecutionStatus::NotFound) => {
                let _ = self.finish_session(
                    session_id,
                    SessionState::Failed,
                    Some(MODAL_NOT_FOUND_ERROR.to_string()),
                    Some(SessionEndReason::ModalNotFound),
                );
                return Vec::new();
            }
            Some(ModalExecutionStatus::Terminated) => {
                if let Some(pending) = record.pending_terminal.clone() {
                    let _ = self.finish_session(
                        session_id,
                        pending.state,
                        pending.error_code,
                        Some(pending.end_reason),
                    );
                } else {
                    let _ = self.finish_session(
                        session_id,
                        SessionState::Failed,
                        Some(MODAL_TERMINATED_ERROR.to_string()),
                        Some(SessionEndReason::ModalTerminated),
                    );
                }
                return Vec::new();
            }
            Some(ModalExecutionStatus::Pending) | None => {}
        }

        let Some(record) = self.sessions.get_mut(session_id) else {
            return Vec::new();
        };

        if record.session.state == SessionState::Starting
            && now.duration_since(record.created_at) > startup_timeout
        {
            Self::begin_canceling(
                record,
                now,
                PendingTerminalDisposition::failed(
                    STARTUP_TIMEOUT_ERROR,
                    SessionEndReason::StartupTimeout,
                ),
            );
        }

        if record.session.state == SessionState::Running
            && record
                .running_at
                .filter(|started| now.duration_since(*started) > max_duration)
                .is_some()
        {
            Self::begin_canceling(
                record,
                now,
                PendingTerminalDisposition::failed(
                    SESSION_TIMEOUT_ERROR,
                    SessionEndReason::SessionTimeout,
                ),
            );
        }

        if record.session.state == SessionState::Running
            && record
                .last_heartbeat_at
                .filter(|heartbeat| now.duration_since(*heartbeat) > worker_heartbeat_timeout)
                .is_some()
        {
            Self::begin_canceling(
                record,
                now,
                PendingTerminalDisposition::failed(
                    WORKER_HEARTBEAT_TIMEOUT_ERROR,
                    SessionEndReason::WorkerHeartbeatTimeout,
                ),
            );
        }

        if record.session.state != SessionState::Canceling {
            return Vec::new();
        }

        if record.cancel_dispatched_at.is_none() {
            return vec![ReconcileCommand::CancelModal {
                session_id: *session_id,
                function_call_id: record.modal_function_call_id.clone(),
                force: false,
            }];
        }

        if record
            .cancel_requested_at
            .filter(|requested| now.duration_since(*requested) > cancel_grace)
            .is_some()
            && record.force_cancel_dispatched_at.is_none()
        {
            return vec![ReconcileCommand::CancelModal {
                session_id: *session_id,
                function_call_id: record.modal_function_call_id.clone(),
                force: true,
            }];
        }

        if record
            .force_cancel_dispatched_at
            .filter(|forced| now.duration_since(*forced) > cancel_grace)
            .is_some()
        {
            let _ = self.finish_session(
                session_id,
                SessionState::Failed,
                Some(CANCEL_TIMEOUT_ERROR.to_string()),
                Some(SessionEndReason::CancelTimeout),
            );
        }

        Vec::new()
    }

    fn begin_canceling(
        record: &mut SessionRecord,
        now: Instant,
        disposition: PendingTerminalDisposition,
    ) {
        if record.session.state == SessionState::Canceling || record.session.state.is_terminal() {
            return;
        }
        record.session.state = SessionState::Canceling;
        record.session.end_reason = Some(disposition.end_reason.clone());
        record.cancel_requested_at = Some(now);
        record.cancel_dispatched_at = None;
        record.force_cancel_dispatched_at = None;
        record.pending_terminal = Some(disposition);
    }

    fn finish_session(
        &mut self,
        session_id: &Uuid,
        state: SessionState,
        error_code: Option<String>,
        end_reason: Option<SessionEndReason>,
    ) -> bool {
        let Some(record) = self.sessions.get_mut(session_id) else {
            return false;
        };

        record.session.state = state;
        record.session.error_code = error_code;
        record.session.end_reason = end_reason;
        record.pending_terminal = None;
        record.cancel_requested_at = None;
        record.cancel_dispatched_at = None;
        record.force_cancel_dispatched_at = None;

        if self.active_session == Some(*session_id) {
            self.active_session = None;
        }
        true
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
    async fn startup_timeout_enters_canceling_and_uses_timeout_failure_on_termination() {
        let mut state = RuntimeState::new();
        let session_id = Uuid::new_v4();
        state.create_session(session_id, "call-1".to_string(), Instant::now());

        advance(Duration::from_secs(121)).await;
        let commands = state.reconcile_active_session(
            &session_id,
            "call-1",
            Some(ModalExecutionStatus::Pending),
            Instant::now(),
            Duration::from_secs(120),
            Duration::from_secs(3600),
            Duration::from_secs(30),
            Duration::from_secs(15),
        );

        assert_eq!(
            commands,
            vec![ReconcileCommand::CancelModal {
                session_id,
                function_call_id: "call-1".to_string(),
                force: false,
            }]
        );
        let session = state
            .get_session(&session_id)
            .expect("session should still exist");
        assert_eq!(session.state, SessionState::Canceling);
        assert_eq!(session.end_reason, Some(SessionEndReason::StartupTimeout));

        state.mark_cancel_dispatched(&session_id, Instant::now(), false);
        let commands = state.reconcile_active_session(
            &session_id,
            "call-1",
            Some(ModalExecutionStatus::Terminated),
            Instant::now(),
            Duration::from_secs(120),
            Duration::from_secs(3600),
            Duration::from_secs(30),
            Duration::from_secs(15),
        );
        assert!(commands.is_empty());

        let session = state
            .get_session(&session_id)
            .expect("session should still exist");
        assert_eq!(session.state, SessionState::Failed);
        assert_eq!(session.error_code, Some(STARTUP_TIMEOUT_ERROR.to_string()));
        assert_eq!(session.end_reason, Some(SessionEndReason::StartupTimeout));
    }
}
