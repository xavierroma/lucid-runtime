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
    models::{CreateSessionRequest, ErrorResponse, ModelsResponse, SessionResponse, CONTROL_TOPIC},
    registry::ModelBackend,
    state::EndRequestError,
    AppContext,
};

pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

pub async fn get_models(State(ctx): State<AppContext>, headers: HeaderMap) -> Response {
    if !auth::is_bearer_authorized(&headers, &ctx.config.api_key) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }

    (
        StatusCode::OK,
        Json(ModelsResponse {
            models: ctx.config.model_registry.supported_models(),
        }),
    )
        .into_response()
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

    let Some(model) = ctx.config.model_registry.get(&model_name) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!("unsupported model_name: {model_name}"),
        );
    };

    let worker_id = match &model.backend {
        ModelBackend::Modal(backend) => backend.worker_id.clone(),
    };
    let capabilities = model.capabilities.clone();

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
        &worker_id,
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
        .modal_dispatch_for_model(&model_name)
        .launch_session(LaunchSessionRequest {
            session_id,
            room_name: room_name.clone(),
            worker_id,
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

    let session = {
        let mut runtime = ctx.runtime.write().await;
        runtime.create_session(session_id, model_name, function_call_id, Instant::now())
    };

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

    let runtime = ctx.runtime.read().await;
    let Some(session) = runtime.get_session(&session_id) else {
        return error_response(StatusCode::NOT_FOUND, "session not found");
    };
    drop(runtime);

    let Some(model) = ctx.config.model_registry.get(&session.model_name) else {
        tracing::error!(
            session_id = %session.session_id,
            model_name = %session.model_name,
            "session references unknown model"
        );
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session references unknown model",
        );
    };

    (
        StatusCode::OK,
        Json(SessionResponse {
            session,
            client_access_token: None,
            capabilities: model.capabilities.clone(),
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

    let end_result = {
        let mut runtime = ctx.runtime.write().await;
        runtime.request_end_session(&session_id, Instant::now())
    };

    let end_result = match end_result {
        Ok(end_result) => end_result,
        Err(EndRequestError::NotFound) => {
            return error_response(StatusCode::NOT_FOUND, "session not found")
        }
    };

    if let Some(function_call_id) = end_result.function_call_id {
        if let Err(err) = ctx
            .modal_dispatch_for_model(&end_result.model_name)
            .cancel_session(&function_call_id, false)
            .await
        {
            tracing::warn!(
                error = %err,
                session_id = %session_id,
                model_name = %end_result.model_name,
                function_call_id = %function_call_id,
                "failed to cancel modal function call"
            );
        } else {
            let mut runtime = ctx.runtime.write().await;
            let _ = runtime.mark_cancel_dispatched(&session_id, Instant::now(), false);
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
