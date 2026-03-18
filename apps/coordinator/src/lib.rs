pub mod api_internal;
pub mod api_public;
pub mod auth;
pub mod capabilities;
pub mod config;
pub mod livekit_tokens;
pub mod modal_dispatch;
pub mod models;
pub mod registry;
pub mod sessions;
pub mod workers;

use std::{collections::HashMap, sync::Arc};

use axum::Router;
use tokio::{
    sync::RwLock,
    task::JoinSet,
    time::{interval, Instant},
};

use capabilities::build_capabilities;
use config::Config;
use modal_dispatch::{HttpModalDispatchClient, ModalDispatch};
use models::Capabilities;
use registry::ModelBackend;
use sessions::{
    reconciler::{reconcile, ReconcileCommand},
    store::{InMemorySessionStore, SessionStore, SessionUpdate},
};
use workers::store::{InMemoryWorkerStore, RegisteredWorker, WorkerStore};

#[derive(Clone)]
pub struct AppContext {
    pub config: Config,
    pub sessions: Arc<dyn SessionStore>,
    pub workers: Arc<dyn WorkerStore>,
    pub dispatchers: Arc<HashMap<String, Arc<dyn ModalDispatch>>>,
    pub capabilities_by_model: Arc<RwLock<HashMap<String, Capabilities>>>,
}

impl AppContext {
    pub fn with_modal_dispatchers(
        config: Config,
        dispatchers: HashMap<String, Arc<dyn ModalDispatch>>,
        capabilities: HashMap<String, Capabilities>,
    ) -> Self {
        let workers = build_worker_store(&config);
        Self {
            config,
            sessions: Arc::new(InMemorySessionStore::new()),
            workers,
            dispatchers: Arc::new(dispatchers),
            capabilities_by_model: Arc::new(RwLock::new(capabilities)),
        }
    }
}

fn build_worker_store(config: &Config) -> Arc<dyn WorkerStore> {
    let store = InMemoryWorkerStore::new();
    for model in config.model_registry.models() {
        let (dispatch_base_url, dispatch_token, worker_id) = match &model.backend {
            ModelBackend::Modal(backend) => (
                backend.dispatch_base_url.clone(),
                backend.dispatch_token.clone(),
                backend.worker_id.clone(),
            ),
        };
        store.save(RegisteredWorker {
            model_id: model.id.clone(),
            display_name: model.display_name.clone(),
            dispatch_base_url,
            dispatch_token,
            worker_id,
            timeouts: model.timeouts.clone(),
        });
    }
    Arc::new(store)
}

pub async fn fetch_capabilities(
    models: &[registry::RegisteredModel],
    dispatchers: &HashMap<String, Arc<dyn ModalDispatch>>,
) -> Result<HashMap<String, Capabilities>, String> {
    let mut map = HashMap::new();
    for model in models {
        let dispatch = dispatchers
            .get(&model.id)
            .expect("dispatcher must exist for every model");
        let manifest = dispatch
            .get_manifest()
            .await
            .map_err(|err| format!("failed to fetch manifest for model {}: {err}", model.id))?;
        let caps = build_capabilities(&manifest);
        map.insert(model.id.clone(), caps);
    }
    Ok(map)
}

pub fn build_dispatchers(config: &Config) -> HashMap<String, Arc<dyn ModalDispatch>> {
    config
        .model_registry
        .models()
        .iter()
        .map(|model| {
            let dispatch = match &model.backend {
                ModelBackend::Modal(backend) => Arc::new(HttpModalDispatchClient::new(
                    backend.dispatch_base_url.clone(),
                    backend.dispatch_token.clone(),
                )) as Arc<dyn ModalDispatch>,
            };
            (model.id.clone(), dispatch)
        })
        .collect()
}

pub fn build_router(ctx: AppContext) -> Router {
    Router::new()
        .route("/healthz", axum::routing::get(api_public::healthz))
        .route("/models", axum::routing::get(api_public::get_models))
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
            "/internal/sessions/:session_id/paused",
            axum::routing::post(api_internal::mark_paused),
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
    let records = ctx.sessions.list_non_terminal();

    if records.is_empty() {
        return;
    }

    let mut status_queries = JoinSet::new();
    for record in records {
        let dispatcher = ctx.dispatchers.get(&record.model_name).cloned();
        status_queries.spawn(async move {
            let modal_status = match dispatcher {
                Some(d) => match d.get_session_status(&record.function_call_id).await {
                    Ok(status) => Some(status),
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            session_id = %record.session_id,
                            function_call_id = %record.function_call_id,
                            "failed to query modal status"
                        );
                        None
                    }
                },
                None => {
                    tracing::warn!(
                        model_name = %record.model_name,
                        "no dispatcher found for session model during reconciliation"
                    );
                    None
                }
            };
            (record, modal_status)
        });
    }

    let now = Instant::now();
    let mut commands = Vec::new();
    while let Some(result) = status_queries.join_next().await {
        match result {
            Ok((record, modal_status)) => {
                let timeouts = ctx
                    .workers
                    .get(&record.model_name)
                    .map(|w| w.timeouts)
                    .unwrap_or_else(|| registry::ModelTimeouts {
                        startup_timeout_secs: 120,
                        session_max_duration_secs: 3600,
                        session_cancel_grace_secs: 30,
                        worker_heartbeat_timeout_secs: 15,
                    });
                let (update, cmds) = reconcile(&record, modal_status, now, &timeouts);
                if let Some(update) = update {
                    let _ = ctx.sessions.update(&record.session_id, update);
                }
                commands.extend(cmds);
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
                model_name,
                function_call_id,
                force,
            } => {
                let Some(dispatcher) = ctx.dispatchers.get(&model_name).cloned() else {
                    tracing::warn!(
                        model_name = %model_name,
                        "no dispatcher for cancel command"
                    );
                    continue;
                };
                match dispatcher.cancel_session(&function_call_id, force).await {
                    Ok(()) => {
                        let update = if force {
                            SessionUpdate {
                                force_cancel_dispatched_at: Some(Some(Instant::now())),
                                ..Default::default()
                            }
                        } else {
                            SessionUpdate {
                                cancel_dispatched_at: Some(Some(Instant::now())),
                                ..Default::default()
                            }
                        };
                        let _ = ctx.sessions.update(&session_id, update);
                    }
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            session_id = %session_id,
                            model_name = %model_name,
                            function_call_id = %function_call_id,
                            force,
                            "failed to dispatch modal cancel during reconciliation"
                        );
                    }
                }
            }
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
        models::{ModelsResponse, SessionEndReason, SessionResponse, SessionState},
        reconcile_runtime,
        sessions::store::{room_name_for, SessionUpdate},
        AppContext,
    };

    const DEFAULT_TEST_MODEL: &str = "yume";
    const HELIOS_TEST_MODEL: &str = "helios";

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

        async fn get_manifest(&self) -> Result<serde_json::Value, ModalDispatchError> {
            Err(ModalDispatchError::Transport(
                "get_manifest not implemented in FakeModalDispatch".to_string(),
            ))
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

    fn dispatchers_for_registry(
        config: &Config,
        default_dispatch: Arc<FakeModalDispatch>,
        overrides: &[(&str, Arc<FakeModalDispatch>)],
    ) -> HashMap<String, Arc<dyn ModalDispatch>> {
        config
            .model_registry
            .models()
            .iter()
            .map(|model| {
                let dispatch = overrides
                    .iter()
                    .find(|(id, _)| *id == model.id)
                    .map(|(_, dispatch)| dispatch.clone())
                    .unwrap_or_else(|| default_dispatch.clone());
                (model.id.clone(), dispatch as Arc<dyn ModalDispatch>)
            })
            .collect()
    }

    fn test_app(fake_dispatch: Arc<FakeModalDispatch>) -> (Router, Config) {
        let config = Config::for_tests();
        let capabilities = test_capabilities_for_registry(&config);
        let ctx = AppContext::with_modal_dispatchers(
            config.clone(),
            dispatchers_for_registry(&config, fake_dispatch, &[]),
            capabilities,
        );
        (build_router(ctx), config)
    }

    fn test_app_with_overrides(
        default_dispatch: Arc<FakeModalDispatch>,
        overrides: &[(&str, Arc<FakeModalDispatch>)],
    ) -> (Router, Config) {
        let config = Config::for_tests();
        let capabilities = test_capabilities_for_registry(&config);
        let ctx = AppContext::with_modal_dispatchers(
            config.clone(),
            dispatchers_for_registry(&config, default_dispatch, overrides),
            capabilities,
        );
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
        model_name: &str,
    ) -> (StatusCode, SessionResponse) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/sessions")
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(json!({ "model_name": model_name }).to_string()))
                    .expect("request should build"),
            )
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

    async fn get_models(app: &Router, config: &Config) -> (StatusCode, ModelsResponse) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/models")
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
        let payload =
            serde_json::from_slice::<ModelsResponse>(&body).expect("get models body should decode");
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
    async fn get_models_returns_registry_order() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch);

        let (status, payload) = get_models(&app, &config).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            payload
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["yume", "waypoint", "helios"]
        );
        assert_eq!(payload.models[2].display_name, "Helios (Distilled)");
    }

    #[tokio::test]
    async fn create_session_allows_multiple_inflight_sessions() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1", "call-2"]));
        let (app, config) = test_app(fake_dispatch.clone());

        let (first_status, first) = create_session(&app, &config, DEFAULT_TEST_MODEL).await;
        let (second_status, second) = create_session(&app, &config, DEFAULT_TEST_MODEL).await;

        assert_eq!(first_status, StatusCode::ACCEPTED);
        assert_eq!(second_status, StatusCode::ACCEPTED);
        assert_eq!(first.session.state, SessionState::Starting);
        assert_eq!(second.session.state, SessionState::Starting);
        assert_eq!(first.session.model_name, DEFAULT_TEST_MODEL);
        assert_eq!(second.session.model_name, DEFAULT_TEST_MODEL);
        assert!(first.client_access_token.is_some());
        assert!(second.client_access_token.is_some());
        assert!(first
            .session
            .room_name
            .starts_with(&format!("wm-{}", first.session.session_id)));
        assert_eq!(
            first.capabilities.manifest["model"]["name"],
            Value::String(DEFAULT_TEST_MODEL.to_string())
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
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({ "model_name": DEFAULT_TEST_MODEL }).to_string(),
                    ))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn create_session_requires_model_name() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch.clone());

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

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(fake_dispatch.launched_count(), 0);
    }

    #[tokio::test]
    async fn create_session_rejects_unknown_model() {
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
                    .body(Body::from(json!({ "model_name": "missing" }).to_string()))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(fake_dispatch.launched_count(), 0);
    }

    #[tokio::test]
    async fn create_session_routes_models_to_their_configured_dispatchers() {
        let default_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-yume"]));
        let helios_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-helios"]));
        let (app, config) = test_app_with_overrides(
            default_dispatch.clone(),
            &[(HELIOS_TEST_MODEL, helios_dispatch.clone())],
        );

        let (_, yume) = create_session(&app, &config, DEFAULT_TEST_MODEL).await;
        let (_, helios) = create_session(&app, &config, HELIOS_TEST_MODEL).await;

        assert_eq!(default_dispatch.launched_count(), 1);
        assert_eq!(helios_dispatch.launched_count(), 1);
        assert_eq!(yume.session.model_name, DEFAULT_TEST_MODEL);
        assert_eq!(helios.session.model_name, HELIOS_TEST_MODEL);
        assert_eq!(
            yume.capabilities.manifest["model"]["name"],
            Value::String(DEFAULT_TEST_MODEL.to_string())
        );
        assert_eq!(
            helios.capabilities.manifest["model"]["name"],
            Value::String(HELIOS_TEST_MODEL.to_string())
        );
    }

    #[tokio::test]
    async fn get_session_returns_found_and_not_found() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch);
        let (_, created) = create_session(&app, &config, DEFAULT_TEST_MODEL).await;

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
        let (_, created) = create_session(&app, &config, DEFAULT_TEST_MODEL).await;
        let (_, other) = create_session(&app, &config, DEFAULT_TEST_MODEL).await;

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
    async fn end_session_uses_the_session_model_dispatcher() {
        let default_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-yume"]));
        let helios_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-helios"]));
        let (app, config) = test_app_with_overrides(
            default_dispatch.clone(),
            &[(HELIOS_TEST_MODEL, helios_dispatch.clone())],
        );
        let (_, helios) = create_session(&app, &config, HELIOS_TEST_MODEL).await;

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/sessions/{}:end", helios.session.session_id))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(default_dispatch.canceled_count(), 0);
        assert_eq!(helios_dispatch.canceled_count(), 1);
    }

    #[tokio::test]
    async fn reconcile_runtime_handles_multiple_sessions_independently() {
        let default_dispatch = Arc::new(FakeModalDispatch::with_ids(&[]));
        let helios_dispatch = Arc::new(FakeModalDispatch::with_ids(&[]));
        default_dispatch.set_status_for("call-1", ModalExecutionStatus::Failure);
        helios_dispatch.set_status_for("call-2", ModalExecutionStatus::Pending);
        let config = Config::for_tests();
        let capabilities = test_capabilities_for_registry(&config);
        let ctx = AppContext::with_modal_dispatchers(
            config.clone(),
            dispatchers_for_registry(
                &config,
                default_dispatch,
                &[(HELIOS_TEST_MODEL, helios_dispatch)],
            ),
            capabilities,
        );
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();

        ctx.sessions.create(
            first,
            DEFAULT_TEST_MODEL.to_string(),
            room_name_for(first),
            "call-1".to_string(),
            Instant::now(),
        );
        ctx.sessions.create(
            second,
            HELIOS_TEST_MODEL.to_string(),
            room_name_for(second),
            "call-2".to_string(),
            Instant::now(),
        );

        reconcile_runtime(&ctx).await;

        let first_session = ctx
            .sessions
            .get(&first)
            .expect("first session should exist");
        let second_session = ctx
            .sessions
            .get(&second)
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
        let capabilities = test_capabilities_for_registry(&config);
        let ctx = AppContext::with_modal_dispatchers(
            config.clone(),
            dispatchers_for_registry(&config, fake_dispatch, &[]),
            capabilities,
        );

        ctx.sessions.create(
            Uuid::new_v4(),
            DEFAULT_TEST_MODEL.to_string(),
            "room-1".to_string(),
            "call-1".to_string(),
            Instant::now(),
        );
        ctx.sessions.create(
            Uuid::new_v4(),
            HELIOS_TEST_MODEL.to_string(),
            "room-2".to_string(),
            "call-2".to_string(),
            Instant::now(),
        );

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
    async fn internal_callbacks_transition_to_ready_paused_running_and_ended() {
        let fake_dispatch = Arc::new(FakeModalDispatch::with_ids(&["call-1"]));
        let (app, config) = test_app(fake_dispatch);
        let (_, created) = create_session(&app, &config, DEFAULT_TEST_MODEL).await;

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

        let paused = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!(
                        "/internal/sessions/{}/paused",
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
        assert_eq!(paused.status(), StatusCode::OK);

        let get_paused = app
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
        let paused_payload = read_json(get_paused).await;
        assert_eq!(paused_payload["session"]["state"], "PAUSED");

        let resumed = app
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
        assert_eq!(resumed.status(), StatusCode::OK);

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

        let paused = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/internal/sessions/{session_id}/paused"))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(paused.status(), StatusCode::UNAUTHORIZED);

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

    #[allow(dead_code)]
    fn _uses_session_update() {
        // Suppress "unused import" warnings for SessionUpdate imported at the top.
        let _ = SessionUpdate::default();
    }
}
