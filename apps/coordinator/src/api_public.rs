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
    models::{
        CreateSessionRequest, ErrorResponse, ModelsResponse, SessionResponse, SupportedModel,
        CONTROL_TOPIC,
    },
    sessions::store::{room_name_for, PendingTerminal, SessionUpdate},
    AppContext,
};
use crate::models::SessionEndReason;

pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

pub async fn get_models(State(ctx): State<AppContext>, headers: HeaderMap) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.api_key) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let models = ctx
        .workers
        .list()
        .into_iter()
        .map(|w| SupportedModel {
            id: w.model_id,
            display_name: w.display_name,
            description: None,
        })
        .collect();

    (StatusCode::OK, Json(ModelsResponse { models })).into_response()
}

pub async fn create_session(
    State(ctx): State<AppContext>,
    headers: HeaderMap,
    payload: Option<Json<CreateSessionRequest>>,
) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.api_key) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    let model_name = match payload
        .and_then(|payload| payload.0.model_name)
        .map(|model_name| model_name.trim().to_lowercase())
        .filter(|model_name| !model_name.is_empty())
    {
        Some(model_name) => model_name,
        None => return error_response(StatusCode::BAD_REQUEST, "model_name is required"),
    };

    let Some(worker) = ctx.workers.get(&model_name) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!("unsupported model_name: {model_name}"),
        );
    };

    let Some(capabilities) = ctx.capabilities_by_model.read().await.get(&model_name).cloned()
    else {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("capabilities not available for model: {model_name}"),
        );
    };

    let session_id = Uuid::new_v4();
    let room_name = room_name_for(session_id);

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
        &worker.worker_id,
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

    let Some(dispatcher) = ctx.dispatchers.get(&model_name).cloned() else {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("no dispatcher available for model: {model_name}"),
        );
    };

    let function_call_id = match dispatcher
        .launch_session(LaunchSessionRequest {
            session_id,
            room_name: room_name.clone(),
            worker_id: worker.worker_id,
            worker_access_token,
            control_topic: CONTROL_TOPIC.to_string(),
            coordinator_base_url: ctx.config.callback_base_url.clone(),
            coordinator_internal_token: ctx.config.worker_internal_token.clone(),
        })
        .await
    {
        Ok(function_call_id) => function_call_id,
        Err(err) => {
            tracing::error!(
                error = %err,
                session_id = %session_id,
                model_name = %model_name,
                "modal launch failed"
            );
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to dispatch session",
            );
        }
    };

    let session = ctx
        .sessions
        .create(session_id, model_name, room_name, function_call_id, Instant::now());

    (
        StatusCode::ACCEPTED,
        Json(SessionResponse {
            session,
            client_access_token: Some(client_access_token),
            capabilities,
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

    let Some(session) = ctx.sessions.get(&session_id) else {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    };

    if ctx.workers.get(&session.model_name).is_none() {
        tracing::error!(
            session_id = %session.session_id,
            model_name = %session.model_name,
            "session references unknown model"
        );
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session references unknown model",
        );
    }

    let Some(capabilities) = ctx
        .capabilities_by_model
        .read()
        .await
        .get(&session.model_name)
        .cloned()
    else {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("capabilities not available for model: {}", session.model_name),
        );
    };

    (
        StatusCode::OK,
        Json(SessionResponse {
            session,
            client_access_token: None,
            capabilities,
        }),
    )
        .into_response()
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

    let Some(record) = ctx.sessions.get_record(&session_id) else {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    };

    // If already terminal or canceling, this is a no-op.
    if record.state.is_terminal() || record.state == crate::models::SessionState::Canceling {
        return StatusCode::OK.into_response();
    }

    let function_call_id = record.function_call_id.clone();
    let model_name = record.model_name.clone();

    let update = SessionUpdate::begin_canceling(
        PendingTerminal::ended(SessionEndReason::ClientRequested),
        Instant::now(),
    );
    match ctx.sessions.update(&session_id, update) {
        Ok(_) => {}
        Err(_) => {
            // Race: session was already terminal/canceling. Treat as success.
            return StatusCode::OK.into_response();
        }
    }

    if let Some(dispatcher) = ctx.dispatchers.get(&model_name).cloned() {
        match dispatcher.cancel_session(&function_call_id, false).await {
            Ok(()) => {
                let _ = ctx.sessions.update(
                    &session_id,
                    SessionUpdate {
                        cancel_dispatched_at: Some(Some(Instant::now())),
                        ..Default::default()
                    },
                );
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    session_id = %session_id,
                    model_name = %model_name,
                    function_call_id = %function_call_id,
                    "failed to cancel modal function call"
                );
            }
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
