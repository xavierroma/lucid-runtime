use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tokio::time::Instant;
use uuid::Uuid;

use crate::{
    auth, livekit_tokens,
    models::{CreateSessionResponse, ErrorResponse},
    state::RuntimeState,
    AppContext,
};

pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

pub async fn create_session(State(ctx): State<AppContext>, headers: HeaderMap) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.api_key) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let now = Instant::now();
    let mut runtime = ctx.runtime.write().await;
    runtime.expire_stale_worker(now, ctx.config.heartbeat_ttl);

    if !runtime.can_create_session(now, ctx.config.heartbeat_ttl) {
        return error_response(StatusCode::CONFLICT, "worker unavailable or busy");
    }

    let session_id = Uuid::new_v4();
    let room_name = RuntimeState::room_name_for(session_id);

    let client_access_token = match livekit_tokens::mint_access_token(
        &ctx.config.livekit_api_key,
        &ctx.config.livekit_api_secret,
        &format!("client-{session_id}"),
        &room_name,
    ) {
        Ok(token) => token,
        Err(err) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to mint client token: {err}"),
            );
        }
    };

    let worker_access_token = match livekit_tokens::mint_access_token(
        &ctx.config.livekit_api_key,
        &ctx.config.livekit_api_secret,
        &ctx.config.worker_id,
        &room_name,
    ) {
        Ok(token) => token,
        Err(err) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to mint worker token: {err}"),
            );
        }
    };

    let session = runtime.create_assigned_session(session_id, worker_access_token.clone());

    (
        StatusCode::CREATED,
        Json(CreateSessionResponse {
            session,
            client_access_token: Some(client_access_token),
            worker_access_token: Some(worker_access_token),
        }),
    )
        .into_response()
}

pub async fn get_session(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.api_key) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let Ok(session_id) = Uuid::parse_str(&session_id) else {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    };

    let runtime = ctx.runtime.read().await;
    let Some(session) = runtime.get_session(&session_id) else {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    };

    (StatusCode::OK, Json(session)).into_response()
}

pub async fn end_session(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    Path(session_id_and_action): Path<String>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.api_key) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let Some(session_id) = parse_end_session_id(&session_id_and_action) else {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    };

    let mut runtime = ctx.runtime.write().await;
    let ended = runtime.request_end_session(&session_id);
    if !ended {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    }

    StatusCode::OK.into_response()
}

fn parse_end_session_id(raw: &str) -> Option<Uuid> {
    raw.strip_suffix(":end")
        .and_then(|session_id| Uuid::parse_str(session_id).ok())
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
