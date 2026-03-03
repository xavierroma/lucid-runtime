use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tokio::time::Instant;
use uuid::Uuid;

use crate::{
    auth,
    models::{
        ErrorResponse, WorkerAssignmentQuery, WorkerAssignmentResponse, WorkerEndedRequest,
        WorkerHeartbeatRequest, WorkerRegisterRequest, WorkerStatusResponse, CONTROL_TOPIC,
        VIDEO_TRACK_NAME,
    },
    state::{SessionTransitionError, WorkerOperationError},
    AppContext,
};

pub async fn register_worker(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Json(payload): Json<WorkerRegisterRequest>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let mut runtime = ctx.runtime.write().await;
    let result = runtime.register_worker(&payload.worker_id, Instant::now());
    if let Err(WorkerOperationError::UnknownWorker) = result {
        return error_response(StatusCode::BAD_REQUEST, "unknown worker_id");
    }

    (StatusCode::OK, Json(WorkerStatusResponse { status: "ok" })).into_response()
}

pub async fn heartbeat_worker(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Json(payload): Json<WorkerHeartbeatRequest>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let mut runtime = ctx.runtime.write().await;
    let result = runtime.heartbeat_worker(&payload.worker_id, Instant::now());
    if let Err(WorkerOperationError::UnknownWorker) = result {
        return error_response(StatusCode::BAD_REQUEST, "unknown worker_id");
    }

    (StatusCode::OK, Json(WorkerStatusResponse { status: "ok" })).into_response()
}

pub async fn get_assignment(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Query(query): Query<WorkerAssignmentQuery>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let mut runtime = ctx.runtime.write().await;
    let result =
        runtime.take_assignment(&query.worker_id, Instant::now(), ctx.config.heartbeat_ttl);

    match result {
        Ok(Some(assignment)) => (
            StatusCode::OK,
            Json(WorkerAssignmentResponse {
                session_id: assignment.session_id,
                room_name: assignment.room_name,
                worker_access_token: assignment.worker_access_token,
                video_track_name: VIDEO_TRACK_NAME.to_string(),
                control_topic: CONTROL_TOPIC.to_string(),
            }),
        )
            .into_response(),
        Ok(None) => StatusCode::NO_CONTENT.into_response(),
        Err(WorkerOperationError::UnknownWorker) => {
            error_response(StatusCode::BAD_REQUEST, "unknown worker_id")
        }
        Err(WorkerOperationError::WorkerUnavailable) => {
            error_response(StatusCode::CONFLICT, "worker unavailable")
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

    let mut runtime = ctx.runtime.write().await;
    runtime.expire_stale_worker(Instant::now(), ctx.config.heartbeat_ttl);
    match runtime.mark_running(&session_id) {
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
    Json(payload): Json<WorkerEndedRequest>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.worker_internal_token) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let mut runtime = ctx.runtime.write().await;
    let ended = runtime.end_session(&session_id, payload.error_code);
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
