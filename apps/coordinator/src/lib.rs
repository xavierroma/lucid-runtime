pub mod api_internal;
pub mod api_public;
pub mod auth;
pub mod capabilities;
pub mod config;
pub mod livekit_tokens;
pub mod modal_dispatch;
pub mod models;
pub mod state;

use std::sync::Arc;

use axum::Router;
use tokio::{
    sync::RwLock,
    task::JoinSet,
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
        .route("/sessions", axum::routing::post(api_public::create_session))
        .route(
            "/sessions/:session_id",
            axum::routing::get(api_public::get_session).post(api_public::end_session),
        )
        .route(
            "/internal/sessions/:session_id/ready",
            axum::routing::post(api_internal::mark_ready),
        )
        .route(
            "/internal/sessions/:session_id/running",
            axum::routing::post(api_internal::mark_running),
        )
        .route(
            "/internal/sessions/:session_id/heartbeat",
            axum::routing::post(api_internal::mark_heartbeat),
        )
        .route(
            "/internal/sessions/:session_id/ended",
            axum::routing::post(api_internal::mark_ended),
        )
        .with_state(ctx)
}

pub async fn reconcile_runtime(ctx: &AppContext) {
    let snapshots = {
        let runtime = ctx.runtime.read().await;
        runtime.non_terminal_session_snapshots()
    };

    if snapshots.is_empty() {
        return;
    }

    let mut status_queries = JoinSet::new();
    for snapshot in snapshots {
        let modal_dispatch = ctx.modal_dispatch.clone();
        status_queries.spawn(async move {
            let modal_status = match modal_dispatch
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
            (snapshot, modal_status)
        });
    }

    let now = Instant::now();
    let mut commands = Vec::new();
    while let Some(result) = status_queries.join_next().await {
        match result {
            Ok((snapshot, modal_status)) => {
                let mut runtime = ctx.runtime.write().await;
                commands.extend(runtime.reconcile_session(
                    &snapshot.session_id,
                    &snapshot.function_call_id,
                    modal_status,
                    now,
                    ctx.config.session_startup_timeout,
                    ctx.config.session_max_duration,
                    ctx.config.session_cancel_grace,
                    ctx.config.worker_heartbeat_timeout,
                ));
            }
            Err(err) => {
                tracing::warn!(error = %err, "modal status query task failed");
            }
        }
    }

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
        collections::{HashMap, VecDeque},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
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
    use tokio::{
        sync::{Barrier, Notify},
        time::{timeout, Instant},
    };
    use tower::util::ServiceExt;
    use uuid::Uuid;

    use crate::{
        build_router,
        config::Config,
        modal_dispatch::{
            LaunchSessionRequest, ModalDispatch, ModalDispatchError, ModalExecutionStatus,
        },
        models::{SessionEndReason, SessionResponse, SessionState},
        reconcile_runtime, AppContext,
    };

    struct StatusProbe {
        expected_started: usize,
        started: AtomicUsize,
        started_notify: Notify,
        barrier: Barrier,
    }

    impl StatusProbe {
        fn new(expected_started: usize) -> Self {
            Self {
                expected_started,
                started: AtomicUsize::new(0),
                started_notify: Notify::new(),
                barrier: Barrier::new(expected_started + 1),
            }
        }

        fn started(&self) -> usize {
            self.started.load(Ordering::SeqCst)
        }
    }

    struct FakeModalDispatch {
        launch_ids: Mutex<VecDeque<String>>,
        launched_sessions: Mutex<Vec<Uuid>>,
        canceled_calls: Mutex<Vec<String>>,
        fail_launch: Mutex<bool>,
        default_status: Mutex<ModalExecutionStatus>,
        status_by_call: Mutex<HashMap<String, ModalExecutionStatus>>,
        status_probe: Option<Arc<StatusProbe>>,
    }

    impl FakeModalDispatch {
        fn with_ids(ids: &[&str]) -> Self {
            Self {
                launch_ids: Mutex::new(ids.iter().map(|id| id.to_string()).collect()),
                launched_sessions: Mutex::new(Vec::new()),
                canceled_calls: Mutex::new(Vec::new()),
                fail_launch: Mutex::new(false),
                default_status: Mutex::new(ModalExecutionStatus::Pending),
                status_by_call: Mutex::new(HashMap::new()),
                status_probe: None,
            }
        }

        fn with_ids_and_status_probe(
            ids: &[&str],
            expected_started: usize,
        ) -> (Self, Arc<StatusProbe>) {
            let probe = Arc::new(StatusProbe::new(expected_started));
            (
                Self {
                    status_probe: Some(probe.clone()),
                    ..Self::with_ids(ids)
                },
                probe,
            )
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

        fn set_status_for(&self, function_call_id: &str, status: ModalExecutionStatus) {
            self.status_by_call
                .lock()
                .expect("status by call lock")
                .insert(function_call_id.to_string(), status);
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
            function_call_id: &str,
        ) -> Result<ModalExecutionStatus, ModalDispatchError> {
            if let Some(probe) = &self.status_probe {
                let started = probe.started.fetch_add(1, Ordering::SeqCst) + 1;
                if started >= probe.expected_started {
                    probe.started_notify.notify_waiters();
                }
                probe.barrier.wait().await;
            }

            if let Some(status) = self
                .status_by_call
                .lock()
                .expect("status by call lock")
                .get(function_call_id)
                .cloned()
            {
                return Ok(status);
            }

            Ok(self.default_status.lock().expect("status lock").clone())
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

    async fn create_session(
        app: &Router,
        config: &Config,
        model_name: Option<&str>,
    ) -> (StatusCode, SessionResponse) {
        let mut request = Request::builder()
            .method(Method::POST)
            .uri("/sessions")
            .header(AUTHORIZATION, format!("Bearer {}", config.api_key));
        let body = if let Some(model_name) = model_name {
            request = request.header(CONTENT_TYPE, "application/json");
            Body::from(json!({ "model_name": model_name }).to_string())
        } else {
            Body::empty()
        };
        let response = app
            .clone()
            .oneshot(request.body(body).expect("request should build"))
            .await
            .expect("request should succeed");

        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        let payload = serde_json::from_slice::<SessionResponse>(&body)
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
    async fn create_session_allows_multiple_inflight_sessions() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1", "call-2"]));
        let (app, config) = test_app(fake_dispatch.clone());

        let (first_status, first) = create_session(&app, &config, None).await;
        let (second_status, second) = create_session(&app, &config, None).await;

        assert_eq!(first_status, StatusCode::ACCEPTED);
        assert_eq!(second_status, StatusCode::ACCEPTED);
        assert_eq!(first.session.state, SessionState::Starting);
        assert_eq!(second.session.state, SessionState::Starting);
        assert!(first.client_access_token.is_some());
        assert!(second.client_access_token.is_some());
        assert!(first
            .session
            .room_name
            .starts_with(&format!("wm-{}", first.session.session_id)));
        assert_eq!(
            first.capabilities.manifest["model"]["name"],
            Value::String(config.model_name.clone())
        );
        assert_ne!(first.session.session_id, second.session.session_id);
        assert_eq!(fake_dispatch.launched_count(), 2);

        for session_id in [first.session.session_id, second.session.session_id] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::GET)
                        .uri(format!("/sessions/{session_id}"))
                        .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                        .body(Body::empty())
                        .expect("request should build"),
                )
                .await
                .expect("request should succeed");
            assert_eq!(response.status(), StatusCode::OK);
        }
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
                    .uri("/sessions")
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn create_session_rejects_mismatched_requested_model() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch.clone());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/sessions")
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(json!({ "model_name": "waypoint" }).to_string()))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(fake_dispatch.launched_count(), 0);
    }

    #[tokio::test]
    async fn get_session_returns_found_and_not_found() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch);
        let (_, created) = create_session(&app, &config, None).await;

        let found = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/sessions/{}", created.session.session_id))
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
                    .uri(format!("/sessions/{}", Uuid::new_v4()))
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
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1", "call-2"]));
        let (app, config) = test_app(fake_dispatch.clone());
        let (_, created) = create_session(&app, &config, None).await;
        let (_, other) = create_session(&app, &config, None).await;

        let first_end = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/sessions/{}:end", created.session.session_id))
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
                    .uri(format!("/sessions/{}", created.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        let payload = read_json(get).await;
        assert_eq!(payload["session"]["state"], "CANCELING");
        assert_eq!(payload["session"]["end_reason"], "CLIENT_REQUESTED");

        let other_get = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/sessions/{}", other.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        let other_payload = read_json(other_get).await;
        assert_eq!(other_payload["session"]["state"], "STARTING");

        let second_end = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/sessions/{}:end", created.session.session_id))
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
    async fn reconcile_runtime_handles_multiple_sessions_independently() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&[]));
        fake_dispatch.set_status_for("call-1", ModalExecutionStatus::Failure);
        fake_dispatch.set_status_for("call-2", ModalExecutionStatus::Pending);
        let config = Config::for_tests();
        let ctx = AppContext::with_modal_dispatch(config, fake_dispatch);
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();

        {
            let mut runtime = ctx.runtime.write().await;
            runtime.create_session(first, "call-1".to_string(), Instant::now());
            runtime.create_session(second, "call-2".to_string(), Instant::now());
        }

        reconcile_runtime(&ctx).await;

        let runtime = ctx.runtime.read().await;
        let first_session = runtime
            .get_session(&first)
            .expect("first session should exist");
        let second_session = runtime
            .get_session(&second)
            .expect("second session should exist");
        assert_eq!(first_session.state, SessionState::Failed);
        assert_eq!(first_session.error_code.as_deref(), Some("MODAL_FAILURE"));
        assert_eq!(
            first_session.end_reason,
            Some(SessionEndReason::ModalFailure)
        );
        assert_eq!(second_session.state, SessionState::Starting);
    }

    #[tokio::test]
    async fn reconcile_runtime_polls_modal_statuses_concurrently() {
        let (dispatch, probe) = FakeModalDispatch::with_ids_and_status_probe(&[], 2);
        let fake_dispatch = Arc::new(dispatch);
        let config = Config::for_tests();
        let ctx = AppContext::with_modal_dispatch(config, fake_dispatch);

        {
            let mut runtime = ctx.runtime.write().await;
            runtime.create_session(Uuid::new_v4(), "call-1".to_string(), Instant::now());
            runtime.create_session(Uuid::new_v4(), "call-2".to_string(), Instant::now());
        }

        let reconcile_task = tokio::spawn({
            let ctx = ctx.clone();
            async move { reconcile_runtime(&ctx).await }
        });

        if probe.started() < 2 {
            timeout(
                std::time::Duration::from_millis(200),
                probe.started_notify.notified(),
            )
            .await
            .expect("both status requests should start before release");
        }
        assert_eq!(probe.started(), 2);

        probe.barrier.wait().await;
        reconcile_task
            .await
            .expect("reconcile task should complete without panic");
    }

    #[tokio::test]
    async fn internal_callbacks_transition_to_ready_running_and_ended() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch);
        let (_, created) = create_session(&app, &config, None).await;

        let ready = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!(
                        "/internal/sessions/{}/ready",
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
        assert_eq!(ready.status(), StatusCode::OK);

        let running = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!(
                        "/internal/sessions/{}/running",
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
                        "/internal/sessions/{}/ended",
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
                    .uri(format!("/sessions/{}", created.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        let payload = read_json(get).await;
        assert_eq!(payload["session"]["state"], "FAILED");
        assert_eq!(payload["session"]["error_code"], "MODEL_RUNTIME_ERROR");
        assert_eq!(payload["session"]["end_reason"], "WORKER_REPORTED_ERROR");
    }

    #[tokio::test]
    async fn internal_routes_require_auth() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, _) = test_app(fake_dispatch);
        let session_id = Uuid::new_v4();

        let ready = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/internal/sessions/{session_id}/ready"))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(ready.status(), StatusCode::UNAUTHORIZED);

        let running = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/internal/sessions/{session_id}/running"))
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
                    .uri(format!("/internal/sessions/{session_id}/heartbeat"))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(heartbeat.status(), StatusCode::UNAUTHORIZED);
    }
}
