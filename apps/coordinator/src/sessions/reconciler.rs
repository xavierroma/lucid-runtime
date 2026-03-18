use std::time::Duration;

use tokio::time::Instant;
use uuid::Uuid;

use crate::{
    modal_dispatch::ModalExecutionStatus,
    models::{SessionEndReason, SessionState},
    registry::ModelTimeouts,
    sessions::store::{PendingTerminal, SessionRecord, SessionUpdate},
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconcileCommand {
    CancelModal {
        session_id: Uuid,
        model_name: String,
        function_call_id: String,
        force: bool,
    },
}

/// Pure reconciliation logic.
///
/// Given the current state of a session and its latest Modal execution status,
/// returns an optional update to apply to the store and zero or more side-effect
/// commands to execute (e.g. dispatching a Modal cancel).
pub fn reconcile(
    record: &SessionRecord,
    modal_status: Option<ModalExecutionStatus>,
    now: Instant,
    timeouts: &ModelTimeouts,
) -> (Option<SessionUpdate>, Vec<ReconcileCommand>) {
    if record.state.is_terminal() {
        return (None, vec![]);
    }

    let startup_timeout = Duration::from_secs(timeouts.startup_timeout_secs);
    let max_duration = Duration::from_secs(timeouts.session_max_duration_secs);
    let cancel_grace = Duration::from_secs(timeouts.session_cancel_grace_secs);
    let heartbeat_timeout = Duration::from_secs(timeouts.worker_heartbeat_timeout_secs);

    // --- Handle terminal Modal statuses first ---

    match modal_status {
        Some(ModalExecutionStatus::Success) => {
            let (state, error_code, end_reason) =
                if let Some(ref pending) = record.pending_terminal {
                    (
                        pending.state.clone(),
                        pending.error_code.clone(),
                        pending.end_reason.clone(),
                    )
                } else {
                    (SessionState::Ended, None, SessionEndReason::ModalSuccess)
                };
            return (
                Some(SessionUpdate::finish(state, error_code, Some(end_reason))),
                vec![],
            );
        }
        Some(ModalExecutionStatus::Failure) => {
            return (
                Some(SessionUpdate::finish(
                    SessionState::Failed,
                    Some(MODAL_FAILURE_ERROR.to_string()),
                    Some(SessionEndReason::ModalFailure),
                )),
                vec![],
            );
        }
        Some(ModalExecutionStatus::InitFailure) => {
            return (
                Some(SessionUpdate::finish(
                    SessionState::Failed,
                    Some(MODAL_INIT_FAILURE_ERROR.to_string()),
                    Some(SessionEndReason::ModalInitFailure),
                )),
                vec![],
            );
        }
        Some(ModalExecutionStatus::Timeout) => {
            return (
                Some(SessionUpdate::finish(
                    SessionState::Failed,
                    Some(MODAL_TIMEOUT_ERROR.to_string()),
                    Some(SessionEndReason::ModalTimeout),
                )),
                vec![],
            );
        }
        Some(ModalExecutionStatus::NotFound) => {
            return (
                Some(SessionUpdate::finish(
                    SessionState::Failed,
                    Some(MODAL_NOT_FOUND_ERROR.to_string()),
                    Some(SessionEndReason::ModalNotFound),
                )),
                vec![],
            );
        }
        Some(ModalExecutionStatus::Terminated) => {
            let update = if let Some(ref pending) = record.pending_terminal {
                SessionUpdate::finish(
                    pending.state.clone(),
                    pending.error_code.clone(),
                    Some(pending.end_reason.clone()),
                )
            } else {
                SessionUpdate::finish(
                    SessionState::Failed,
                    Some(MODAL_TERMINATED_ERROR.to_string()),
                    Some(SessionEndReason::ModalTerminated),
                )
            };
            return (Some(update), vec![]);
        }
        Some(ModalExecutionStatus::Pending) | None => {}
    }

    // --- Timeout checks (may transition to Canceling) ---

    if record.state == SessionState::Starting
        && now.duration_since(record.created_at) > startup_timeout
    {
        return begin_cancel(
            record,
            now,
            PendingTerminal::failed(STARTUP_TIMEOUT_ERROR, SessionEndReason::StartupTimeout),
        );
    }

    if matches!(
        record.state,
        SessionState::Ready | SessionState::Running | SessionState::Paused
    ) && record
        .ready_at
        .filter(|ready| now.duration_since(*ready) > max_duration)
        .is_some()
    {
        return begin_cancel(
            record,
            now,
            PendingTerminal::failed(SESSION_TIMEOUT_ERROR, SessionEndReason::SessionTimeout),
        );
    }

    if matches!(
        record.state,
        SessionState::Ready | SessionState::Running | SessionState::Paused
    ) && record
        .last_heartbeat_at
        .filter(|hb| now.duration_since(*hb) > heartbeat_timeout)
        .is_some()
    {
        return begin_cancel(
            record,
            now,
            PendingTerminal::failed(
                WORKER_HEARTBEAT_TIMEOUT_ERROR,
                SessionEndReason::WorkerHeartbeatTimeout,
            ),
        );
    }

    // --- Canceling state: decide whether to (re-)dispatch cancel ---

    if record.state != SessionState::Canceling {
        return (None, vec![]);
    }

    if record.cancel_dispatched_at.is_none() {
        return (
            None,
            vec![ReconcileCommand::CancelModal {
                session_id: record.session_id,
                model_name: record.model_name.clone(),
                function_call_id: record.function_call_id.clone(),
                force: false,
            }],
        );
    }

    if record
        .cancel_requested_at
        .filter(|requested| now.duration_since(*requested) > cancel_grace)
        .is_some()
        && record.force_cancel_dispatched_at.is_none()
    {
        return (
            None,
            vec![ReconcileCommand::CancelModal {
                session_id: record.session_id,
                model_name: record.model_name.clone(),
                function_call_id: record.function_call_id.clone(),
                force: true,
            }],
        );
    }

    if record
        .force_cancel_dispatched_at
        .filter(|forced| now.duration_since(*forced) > cancel_grace)
        .is_some()
    {
        return (
            Some(SessionUpdate::finish(
                SessionState::Failed,
                Some(CANCEL_TIMEOUT_ERROR.to_string()),
                Some(SessionEndReason::CancelTimeout),
            )),
            vec![],
        );
    }

    (None, vec![])
}

fn begin_cancel(
    record: &SessionRecord,
    now: Instant,
    disposition: PendingTerminal,
) -> (Option<SessionUpdate>, Vec<ReconcileCommand>) {
    if record.state == SessionState::Canceling || record.state.is_terminal() {
        return (None, vec![]);
    }
    let update = SessionUpdate::begin_canceling(disposition, now);
    let command = ReconcileCommand::CancelModal {
        session_id: record.session_id,
        model_name: record.model_name.clone(),
        function_call_id: record.function_call_id.clone(),
        force: false,
    };
    (Some(update), vec![command])
}
