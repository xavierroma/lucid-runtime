use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tokio::time::Instant;
use uuid::Uuid;

use crate::{
    auth,
    models::{ErrorResponse, RuntimeEndedRequest, SessionEndReason, SessionState},
    sessions::store::{SessionUpdate, UpdateError},
    AppContext,
};

pub async fn mark_ready(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id): Path<Uuid>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    // No-op when the session is already being cancelled or is terminal.
    if let Some(session) = ctx.sessions.get(&session_id) {
        if session.state == SessionState::Canceling || session.state.is_terminal() {
            return StatusCode::OK.into_response();
        }
    }

    let now = Instant::now();
    match ctx.sessions.update(
        &session_id,
        SessionUpdate {
            state: Some(SessionState::Ready),
            ready_at: Some(now),
            last_heartbeat_at: Some(now),
            ..Default::default()
        },
    ) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(UpdateError::NotFound) => error_response(StatusCode::NOT_FOUND, "session not found"),
        Err(UpdateError::InvalidTransition) => {
            error_response(StatusCode::CONFLICT, "invalid session transition")
        }
    }
}

pub async fn mark_running(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id): Path<Uuid>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    if let Some(session) = ctx.sessions.get(&session_id) {
        if session.state == SessionState::Canceling || session.state.is_terminal() {
            return StatusCode::OK.into_response();
        }
    }

    let now = Instant::now();
    match ctx.sessions.update(
        &session_id,
        SessionUpdate {
            state: Some(SessionState::Running),
            running_at: Some(now),
            last_heartbeat_at: Some(now),
            ..Default::default()
        },
    ) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(UpdateError::NotFound) => error_response(StatusCode::NOT_FOUND, "session not found"),
        Err(UpdateError::InvalidTransition) => {
            error_response(StatusCode::CONFLICT, "invalid session transition")
        }
    }
}

pub async fn mark_paused(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id): Path<Uuid>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    if let Some(session) = ctx.sessions.get(&session_id) {
        if session.state == SessionState::Canceling || session.state.is_terminal() {
            return StatusCode::OK.into_response();
        }
    }

    let now = Instant::now();
    match ctx.sessions.update(
        &session_id,
        SessionUpdate {
            state: Some(SessionState::Paused),
            last_heartbeat_at: Some(now),
            ..Default::default()
        },
    ) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(UpdateError::NotFound) => error_response(StatusCode::NOT_FOUND, "session not found"),
        Err(UpdateError::InvalidTransition) => {
            error_response(StatusCode::CONFLICT, "invalid session transition")
        }
    }
}

pub async fn mark_heartbeat(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id): Path<Uuid>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    match ctx.sessions.update(
        &session_id,
        SessionUpdate {
            last_heartbeat_at: Some(Instant::now()),
            ..Default::default()
        },
    ) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(UpdateError::NotFound) => error_response(StatusCode::NOT_FOUND, "session not found"),
        Err(UpdateError::InvalidTransition) => {
            error_response(StatusCode::CONFLICT, "invalid session transition")
        }
    }
}

pub async fn mark_ended(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id): Path<Uuid>,
    Json(payload): Json<RuntimeEndedRequest>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let Some(record) = ctx.sessions.get_record(&session_id) else {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    };

    // Compute the terminal state, respecting any pending disposition set by a
    // cancel request.
    let (state, error_code, end_reason) = if let Some(error_code) = payload.error_code {
        (
            SessionState::Failed,
            Some(error_code),
            Some(payload.end_reason.unwrap_or(SessionEndReason::WorkerReportedError)),
        )
    } else if let Some(pending) = record.pending_terminal {
        (
            pending.state,
            pending.error_code,
            Some(payload.end_reason.unwrap_or(pending.end_reason)),
        )
    } else {
        (
            SessionState::Ended,
            None,
            Some(payload.end_reason.unwrap_or(SessionEndReason::NormalCompletion)),
        )
    };

    match ctx
        .sessions
        .update(&session_id, SessionUpdate::finish(state, error_code, end_reason))
    {
        Ok(_) | Err(UpdateError::InvalidTransition) => StatusCode::OK.into_response(),
        Err(UpdateError::NotFound) => error_response(StatusCode::NOT_FOUND, "session not found"),
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: message.to_string(),
        }),
    )
        .into_response()
}
