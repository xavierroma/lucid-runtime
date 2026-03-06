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
    models::{ErrorResponse, RuntimeEndedRequest},
    state::SessionTransitionError,
    AppContext,
};

pub async fn mark_running(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id): Path<Uuid>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let mut runtime = ctx.runtime.write().await;
    match runtime.mark_running(&session_id, Instant::now()) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(SessionTransitionError::NotFound) => {
            error_response(StatusCode::NOT_FOUND, "session not found")
        }
        Err(SessionTransitionError::InvalidState) => {
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

    let mut runtime = ctx.runtime.write().await;
    match runtime.mark_heartbeat(&session_id, Instant::now()) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(SessionTransitionError::NotFound) => {
            error_response(StatusCode::NOT_FOUND, "session not found")
        }
        Err(SessionTransitionError::InvalidState) => {
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

    let mut runtime = ctx.runtime.write().await;
    let ended = runtime.mark_ended(&session_id, payload.error_code, payload.end_reason);
    if !ended {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    }

    StatusCode::OK.into_response()
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
