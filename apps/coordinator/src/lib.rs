pub mod api_internal;
pub mod api_public;
pub mod auth;
pub mod config;
pub mod livekit_tokens;
pub mod modal_dispatch;
pub mod models;
pub mod state;

use std::sync::Arc;

use axum::Router;
use tokio::{
    sync::RwLock,
    time::{interval, Instant},
};

use config::Config;
use modal_dispatch::{HttpModalDispatchClient, ModalDispatch};
use state::{ReconcileCommand, RuntimeState};

#[derive(Clone)]
pub struct AppContext {
    pub config: Config,
    pub runtime: Arc<RwLock<RuntimeState>>,
    pub modal_dispatch: Arc<dyn ModalDispatch>,
}

impl AppContext {
    pub fn new(config: Config) -> Self {
        let runtime = RuntimeState::new();
        let modal_dispatch = HttpModalDispatchClient::new(
            config.modal_dispatch_base_url.clone(),
            config.modal_dispatch_token.clone(),
        );
        Self {
            config,
            runtime: Arc::new(RwLock::new(runtime)),
            modal_dispatch: Arc::new(modal_dispatch),
        }
    }

    pub fn with_modal_dispatch(config: Config, modal_dispatch: Arc<dyn ModalDispatch>) -> Self {
        let runtime = RuntimeState::new();
        Self {
            config,
            runtime: Arc::new(RwLock::new(runtime)),
            modal_dispatch,
        }
    }
}

pub fn build_router(ctx: AppContext) -> Router {
    Router::new()
        .route("/healthz", axum::routing::get(api_public::healthz))
        .route(
            "/v1/sessions",
            axum::routing::post(api_public::create_session),
        )
        .route(
            "/v1/sessions/:session_id",
            axum::routing::get(api_public::get_session).post(api_public::end_session),
        )
        .route(
            "/internal/v1/sessions/:session_id/running",
            axum::routing::post(api_internal::mark_running),
        )
        .route(
            "/internal/v1/sessions/:session_id/heartbeat",
            axum::routing::post(api_internal::mark_heartbeat),
        )
        .route(
            "/internal/v1/sessions/:session_id/ended",
            axum::routing::post(api_internal::mark_ended),
        )
        .with_state(ctx)
}

pub async fn reconcile_runtime(ctx: &AppContext) {
    let snapshot = {
        let runtime = ctx.runtime.read().await;
        runtime.active_session_snapshot()
    };

    let Some(snapshot) = snapshot else {
        return;
    };

    let modal_status = match ctx
        .modal_dispatch
        .get_session_status(&snapshot.function_call_id)
        .await
    {
        Ok(status) => Some(status),
        Err(err) => {
            tracing::warn!(
                error = %err,
                session_id = %snapshot.session_id,
                function_call_id = %snapshot.function_call_id,
                "failed to query modal status"
            );
            None
        }
    };

    let commands = {
        let mut runtime = ctx.runtime.write().await;
        runtime.reconcile_active_session(
            &snapshot.session_id,
            &snapshot.function_call_id,
            modal_status,
            Instant::now(),
            ctx.config.session_startup_timeout,
            ctx.config.session_max_duration,
            ctx.config.session_cancel_grace,
            ctx.config.worker_heartbeat_timeout,
        )
    };

    execute_reconcile_commands(ctx, commands).await;
}

pub async fn run_session_reconciler(ctx: AppContext) {
    let mut ticker = interval(std::time::Duration::from_secs(1));

    loop {
        ticker.tick().await;
        reconcile_runtime(&ctx).await;
    }
}

async fn execute_reconcile_commands(ctx: &AppContext, commands: Vec<ReconcileCommand>) {
    for command in commands {
        match command {
            ReconcileCommand::CancelModal {
                session_id,
                function_call_id,
                force,
            } => match ctx
                .modal_dispatch
                .cancel_session(&function_call_id, force)
                .await
            {
                Ok(()) => {
                    let mut runtime = ctx.runtime.write().await;
                    let _ = runtime.mark_cancel_dispatched(&session_id, Instant::now(), force);
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        session_id = %session_id,
                        function_call_id = %function_call_id,
                        force,
                        "failed to dispatch modal cancel during reconciliation"
                    );
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use axum::{
        body::{to_bytes, Body},
        http::{
            header::{AUTHORIZATION, CONTENT_TYPE},
            Method, Request, StatusCode,
        },
        Router,
    };
    use serde_json::{json, Value};
    use tower::util::ServiceExt;
    use uuid::Uuid;

    use crate::{
        build_router,
        config::Config,
        modal_dispatch::{
            LaunchSessionRequest, ModalDispatch, ModalDispatchError, ModalExecutionStatus,
        },
        models::{CreateSessionResponse, SessionState},
        AppContext,
    };

    struct FakeModalDispatch {
        launch_ids: Mutex<VecDeque<String>>,
        launched_sessions: Mutex<Vec<Uuid>>,
        canceled_calls: Mutex<Vec<String>>,
        fail_launch: Mutex<bool>,
        status: Mutex<ModalExecutionStatus>,
    }

    impl FakeModalDispatch {
        fn with_ids(ids: &[&str]) -> Self {
            Self {
                launch_ids: Mutex::new(ids.iter().map(|id| id.to_string()).collect()),
                launched_sessions: Mutex::new(Vec::new()),
                canceled_calls: Mutex::new(Vec::new()),
                fail_launch: Mutex::new(false),
                status: Mutex::new(ModalExecutionStatus::Pending),
            }
        }

        fn launched_count(&self) -> usize {
            self.launched_sessions
                .lock()
                .expect("launch sessions lock")
                .len()
        }

        fn canceled_count(&self) -> usize {
            self.canceled_calls.lock().expect("cancel calls lock").len()
        }
    }

    impl Default for FakeModalDispatch {
        fn default() -> Self {
            Self::with_ids(&[])
        }
    }

    #[async_trait]
    impl ModalDispatch for FakeModalDispatch {
        async fn launch_session(
            &self,
            payload: LaunchSessionRequest,
        ) -> Result<String, ModalDispatchError> {
            self.launched_sessions
                .lock()
                .expect("launch sessions lock")
                .push(payload.session_id);
            if *self.fail_launch.lock().expect("fail launch lock") {
                return Err(ModalDispatchError::Transport(
                    "forced launch failure".to_string(),
                ));
            }
            Ok(self
                .launch_ids
                .lock()
                .expect("launch ids lock")
                .pop_front()
                .unwrap_or_else(|| "call-default".to_string()))
        }

        async fn cancel_session(
            &self,
            function_call_id: &str,
            _force: bool,
        ) -> Result<(), ModalDispatchError> {
            self.canceled_calls
                .lock()
                .expect("cancel calls lock")
                .push(function_call_id.to_string());
            Ok(())
        }

        async fn get_session_status(
            &self,
            _function_call_id: &str,
        ) -> Result<ModalExecutionStatus, ModalDispatchError> {
            Ok(self.status.lock().expect("status lock").clone())
        }
    }

    fn test_app(fake_dispatch: Arc<FakeModalDispatch>) -> (Router, Config) {
        let config = Config::for_tests();
        let ctx = AppContext::with_modal_dispatch(config.clone(), fake_dispatch);
        (build_router(ctx), config)
    }

    async fn read_json(response: axum::response::Response) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        serde_json::from_slice(&body).expect("response body should be JSON")
    }

    async fn create_session(app: &Router, config: &Config) -> (StatusCode, CreateSessionResponse) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/sessions")
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        let payload = serde_json::from_slice::<CreateSessionResponse>(&body)
            .expect("create session body should decode");
        (status, payload)
    }

    #[tokio::test]
    async fn healthz_is_public() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, _) = test_app(fake_dispatch);

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/healthz")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn create_session_returns_accepted_and_conflicts_when_busy() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1", "call-2"]));
        let (app, config) = test_app(fake_dispatch.clone());

        let (status, payload) = create_session(&app, &config).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(payload.session.state, SessionState::Starting);
        assert!(payload.client_access_token.is_some());
        assert!(payload
            .session
            .room_name
            .starts_with(&format!("wm-{}", payload.session.session_id)));
        assert_eq!(fake_dispatch.launched_count(), 1);

        let second = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/sessions")
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(second.status(), StatusCode::CONFLICT);
        assert_eq!(fake_dispatch.launched_count(), 1);
    }

    #[tokio::test]
    async fn create_session_returns_server_error_on_launch_failure() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        *fake_dispatch.fail_launch.lock().expect("fail launch lock") = true;
        let (app, config) = test_app(fake_dispatch);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/sessions")
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn get_session_returns_found_and_not_found() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch);
        let (_, created) = create_session(&app, &config).await;

        let found = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/v1/sessions/{}", created.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(found.status(), StatusCode::OK);

        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/v1/sessions/{}", Uuid::new_v4()))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn end_session_is_idempotent_and_best_effort_cancels_modal_job() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch.clone());
        let (_, created) = create_session(&app, &config).await;

        let first_end = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/sessions/{}:end", created.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(first_end.status(), StatusCode::OK);
        assert_eq!(fake_dispatch.canceled_count(), 1);

        let get = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/v1/sessions/{}", created.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        let payload = read_json(get).await;
        assert_eq!(payload["state"], "CANCELING");
        assert_eq!(payload["end_reason"], "CLIENT_REQUESTED");

        let second_end = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/sessions/{}:end", created.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(second_end.status(), StatusCode::OK);
        assert_eq!(fake_dispatch.canceled_count(), 1);
    }

    #[tokio::test]
    async fn internal_callbacks_transition_to_running_and_ended() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch);
        let (_, created) = create_session(&app, &config).await;

        let running = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!(
                        "/internal/v1/sessions/{}/running",
                        created.session.session_id
                    ))
                    .header(
                        AUTHORIZATION,
                        format!("Bearer {}", config.worker_internal_token),
                    )
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(running.status(), StatusCode::OK);

        let ended = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!(
                        "/internal/v1/sessions/{}/ended",
                        created.session.session_id
                    ))
                    .header(
                        AUTHORIZATION,
                        format!("Bearer {}", config.worker_internal_token),
                    )
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({ "error_code": "MODEL_RUNTIME_ERROR" }).to_string(),
                    ))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(ended.status(), StatusCode::OK);

        let get = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/v1/sessions/{}", created.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        let payload = read_json(get).await;
        assert_eq!(payload["state"], "FAILED");
        assert_eq!(payload["error_code"], "MODEL_RUNTIME_ERROR");
        assert_eq!(payload["end_reason"], "WORKER_REPORTED_ERROR");
    }

    #[tokio::test]
    async fn internal_routes_require_auth() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, _) = test_app(fake_dispatch);
        let session_id = Uuid::new_v4();

        let running = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/internal/v1/sessions/{session_id}/running"))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(running.status(), StatusCode::UNAUTHORIZED);

        let heartbeat = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/internal/v1/sessions/{session_id}/heartbeat"))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(heartbeat.status(), StatusCode::UNAUTHORIZED);
    }
}
