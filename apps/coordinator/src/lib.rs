pub mod api_internal;
pub mod api_public;
pub mod auth;
pub mod config;
pub mod livekit_tokens;
pub mod models;
pub mod state;

use std::sync::Arc;

use axum::Router;
use tokio::{
    sync::RwLock,
    time::{interval, Instant},
};

use config::Config;
use state::RuntimeState;

#[derive(Clone)]
pub struct AppContext {
    pub config: Config,
    pub runtime: Arc<RwLock<RuntimeState>>,
}

impl AppContext {
    pub fn new(config: Config) -> Self {
        let runtime = RuntimeState::new(config.worker_id.clone());
        Self {
            config,
            runtime: Arc::new(RwLock::new(runtime)),
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
            "/internal/v1/worker/register",
            axum::routing::post(api_internal::register_worker),
        )
        .route(
            "/internal/v1/worker/heartbeat",
            axum::routing::post(api_internal::heartbeat_worker),
        )
        .route(
            "/internal/v1/worker/assignment",
            axum::routing::get(api_internal::get_assignment),
        )
        .route(
            "/internal/v1/sessions/:session_id/running",
            axum::routing::post(api_internal::mark_running),
        )
        .route(
            "/internal/v1/sessions/:session_id/ended",
            axum::routing::post(api_internal::mark_ended),
        )
        .with_state(ctx)
}

pub async fn run_heartbeat_monitor(ctx: AppContext) {
    let mut ticker = interval(std::time::Duration::from_secs(1));

    loop {
        ticker.tick().await;
        let mut runtime = ctx.runtime.write().await;
        runtime.expire_stale_worker(Instant::now(), ctx.config.heartbeat_ttl);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

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
        models::{CreateSessionResponse, SessionState},
        AppContext,
    };

    fn test_app() -> (Router, Config) {
        let config = Config::for_tests();
        let ctx = AppContext::new(config.clone());
        (build_router(ctx), config)
    }

    async fn read_json(response: axum::response::Response) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        serde_json::from_slice(&body).expect("response body should be JSON")
    }

    async fn register_worker(app: &Router, config: &Config) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/internal/v1/worker/register")
                    .header(
                        AUTHORIZATION,
                        format!("Bearer {}", config.worker_internal_token),
                    )
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({ "worker_id": config.worker_id }).to_string(),
                    ))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::OK);
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
        let (app, _) = test_app();

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
    async fn public_endpoints_require_auth() {
        let (app, _) = test_app();
        let session_id = Uuid::new_v4();

        let create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/sessions")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(create.status(), StatusCode::UNAUTHORIZED);

        let get = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/v1/sessions/{session_id}"))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(get.status(), StatusCode::UNAUTHORIZED);

        let end = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/sessions/{session_id}:end"))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(end.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn create_session_success_and_busy_conflict() {
        let (app, config) = test_app();
        register_worker(&app, &config).await;

        let (status, payload) = create_session(&app, &config).await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(payload.session.state, SessionState::Assigned);
        assert!(payload.client_access_token.is_some());
        assert!(payload.worker_access_token.is_some());
        assert!(payload
            .session
            .room_name
            .starts_with(&format!("wm-{}", payload.session.session_id)));

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
    }

    #[tokio::test]
    async fn get_session_returns_found_and_not_found() {
        let (app, config) = test_app();
        register_worker(&app, &config).await;
        let (_, create_payload) = create_session(&app, &config).await;

        let found = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!(
                        "/v1/sessions/{}",
                        create_payload.session.session_id
                    ))
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
    async fn end_session_is_idempotent_for_existing_session() {
        let (app, config) = test_app();
        register_worker(&app, &config).await;
        let (_, create_payload) = create_session(&app, &config).await;

        let end_uri = format!("/v1/sessions/{}:end", create_payload.session.session_id);

        let first_end = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&end_uri)
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(first_end.status(), StatusCode::OK);

        let second_end = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&end_uri)
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(second_end.status(), StatusCode::OK);

        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/sessions/{}:end", Uuid::new_v4()))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn internal_endpoints_require_auth() {
        let (app, _) = test_app();

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/internal/v1/worker/register")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"worker_id":"wm-worker-1"}"#))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn assignment_lifecycle_and_running_transition() {
        let (app, config) = test_app();
        register_worker(&app, &config).await;

        let no_assignment = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!(
                        "/internal/v1/worker/assignment?worker_id={}",
                        config.worker_id
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
        assert_eq!(no_assignment.status(), StatusCode::NO_CONTENT);

        let (_, create_payload) = create_session(&app, &config).await;

        let assignment = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!(
                        "/internal/v1/worker/assignment?worker_id={}",
                        config.worker_id
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
        assert_eq!(assignment.status(), StatusCode::OK);

        let assignment_payload = read_json(assignment).await;
        assert_eq!(
            assignment_payload["session_id"],
            create_payload.session.session_id.to_string()
        );

        let second_poll = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!(
                        "/internal/v1/worker/assignment?worker_id={}",
                        config.worker_id
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
        assert_eq!(second_poll.status(), StatusCode::NO_CONTENT);

        let running = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!(
                        "/internal/v1/sessions/{}/running",
                        create_payload.session.session_id
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

        let get = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!(
                        "/v1/sessions/{}",
                        create_payload.session.session_id
                    ))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");
        assert_eq!(get.status(), StatusCode::OK);

        let get_payload = read_json(get).await;
        assert_eq!(get_payload["state"], "RUNNING");
    }

    #[tokio::test]
    async fn internal_ended_releases_worker_for_new_session() {
        let (app, config) = test_app();
        register_worker(&app, &config).await;
        let (_, create_payload) = create_session(&app, &config).await;

        let ended = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!(
                        "/internal/v1/sessions/{}/ended",
                        create_payload.session.session_id
                    ))
                    .header(
                        AUTHORIZATION,
                        format!("Bearer {}", config.worker_internal_token),
                    )
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"error_code": "MODEL_RUNTIME_ERROR"}).to_string(),
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
                    .uri(format!(
                        "/v1/sessions/{}",
                        create_payload.session.session_id
                    ))
                    .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        let payload = read_json(get).await;
        assert_eq!(payload["state"], "ENDED");
        assert_eq!(payload["error_code"], "MODEL_RUNTIME_ERROR");

        tokio::time::sleep(Duration::from_millis(5)).await;

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

        assert_eq!(second.status(), StatusCode::CREATED);
    }
}
