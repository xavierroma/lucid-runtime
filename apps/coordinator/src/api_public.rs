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
    modal_dispatch::LaunchSessionRequest,
    models::{CreateSessionResponse, ErrorResponse, CONTROL_TOPIC, VIDEO_TRACK_NAME},
    state::EndRequestError,
    AppContext,
};

pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

pub async fn create_session(State(ctx): State<AppContext>, headers: HeaderMap) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.api_key) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    {
        let mut runtime = ctx.runtime.write().await;
        runtime.reconcile_timeouts(
            Instant::now(),
            ctx.config.session_startup_timeout,
            ctx.config.session_max_duration,
            ctx.config.session_cancel_grace,
        );
        if !runtime.can_create_session() {
            return error_response(StatusCode::CONFLICT, "active session in progress");
        }
    }

    let session_id = Uuid::new_v4();
    let room_name = crate::state::RuntimeState::room_name_for(session_id);

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

    let function_call_id = match ctx
        .modal_dispatch
        .launch_session(LaunchSessionRequest {
            session_id,
            room_name: room_name.clone(),
            worker_id: ctx.config.worker_id.clone(),
            worker_access_token: worker_access_token.clone(),
            video_track_name: VIDEO_TRACK_NAME.to_string(),
            control_topic: CONTROL_TOPIC.to_string(),
            coordinator_base_url: ctx.config.callback_base_url.clone(),
            coordinator_internal_token: ctx.config.worker_internal_token.clone(),
        })
        .await
    {
        Ok(function_call_id) => function_call_id,
        Err(err) => {
            tracing::error!(error = %err, session_id = %session_id, "modal launch failed");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to dispatch session",
            );
        }
    };

    let session = {
        let mut runtime = ctx.runtime.write().await;
        runtime.reconcile_timeouts(
            Instant::now(),
            ctx.config.session_startup_timeout,
            ctx.config.session_max_duration,
            ctx.config.session_cancel_grace,
        );
        if !runtime.can_create_session() {
            if let Err(err) = ctx.modal_dispatch.cancel_session(&function_call_id).await {
                tracing::warn!(
                    error = %err,
                    function_call_id = %function_call_id,
                    "failed to cancel modal call after coordinator conflict"
                );
            }
            return error_response(StatusCode::CONFLICT, "active session in progress");
        }
        runtime.create_session(session_id, function_call_id, Instant::now())
    };

    (
        StatusCode::ACCEPTED,
        Json(CreateSessionResponse {
            session,
            client_access_token: Some(client_access_token),
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

    let end_result = {
        let mut runtime = ctx.runtime.write().await;
        runtime.reconcile_timeouts(
            Instant::now(),
            ctx.config.session_startup_timeout,
            ctx.config.session_max_duration,
            ctx.config.session_cancel_grace,
        );
        runtime.request_end_session(&session_id, Instant::now())
    };

    let end_result = match end_result {
        Ok(end_result) => end_result,
        Err(EndRequestError::NotFound) => {
            return error_response(StatusCode::NOT_FOUND, "session not found")
        }
    };

    if let Some(function_call_id) = end_result.function_call_id {
        if let Err(err) = ctx.modal_dispatch.cancel_session(&function_call_id).await {
            tracing::warn!(
                error = %err,
                function_call_id = %function_call_id,
                "failed to cancel modal function call"
            );
        }
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
