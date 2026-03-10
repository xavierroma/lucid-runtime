use std::{env, net::SocketAddr, str::FromStr, time::Duration};

use thiserror::Error;

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub api_key: String,
    pub worker_internal_token: String,
    pub livekit_api_key: String,
    pub livekit_api_secret: String,
    pub model_name: String,
    pub worker_id: String,
    pub modal_dispatch_base_url: String,
    pub modal_dispatch_token: String,
    pub callback_base_url: String,
    pub session_startup_timeout: Duration,
    pub session_max_duration: Duration,
    pub session_cancel_grace: Duration,
    pub worker_heartbeat_timeout: Duration,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing required environment variable: {0}")]
    MissingVar(&'static str),
    #[error("invalid COORDINATOR_BIND_ADDR: {0}")]
    InvalidBindAddr(String),
    #[error("invalid SESSION_STARTUP_TIMEOUT_SECS: {0}")]
    InvalidSessionStartupTimeout(String),
    #[error("invalid SESSION_MAX_DURATION_SECS: {0}")]
    InvalidSessionMaxDuration(String),
    #[error("invalid SESSION_CANCEL_GRACE_SECS: {0}")]
    InvalidSessionCancelGrace(String),
    #[error("invalid WORKER_HEARTBEAT_TIMEOUT_SECS: {0}")]
    InvalidWorkerHeartbeatTimeout(String),
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let bind_addr = env::var("COORDINATOR_BIND_ADDR")
            .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
            .parse()
            .map_err(|err: std::net::AddrParseError| {
                ConfigError::InvalidBindAddr(err.to_string())
            })?;

        Ok(Self {
            bind_addr,
            api_key: required("API_KEY")?,
            worker_internal_token: required("WORKER_INTERNAL_TOKEN")?,
            livekit_api_key: required("LIVEKIT_API_KEY")?,
            livekit_api_secret: required("LIVEKIT_API_SECRET")?,
            model_name: env::var("WM_MODEL_NAME")
                .unwrap_or_else(|_| "yume".to_string())
                .trim()
                .to_lowercase(),
            worker_id: env::var("WORKER_ID").unwrap_or_else(|_| "wm-worker-1".to_string()),
            modal_dispatch_base_url: required("MODAL_DISPATCH_BASE_URL")?
                .trim_end_matches('/')
                .to_string(),
            modal_dispatch_token: required("MODAL_DISPATCH_TOKEN")?,
            callback_base_url: required("COORDINATOR_CALLBACK_BASE_URL")?
                .trim_end_matches('/')
                .to_string(),
            session_startup_timeout: duration_var(
                "SESSION_STARTUP_TIMEOUT_SECS",
                120,
                ConfigError::InvalidSessionStartupTimeout,
            )?,
            session_max_duration: duration_var(
                "SESSION_MAX_DURATION_SECS",
                3600,
                ConfigError::InvalidSessionMaxDuration,
            )?,
            session_cancel_grace: duration_var(
                "SESSION_CANCEL_GRACE_SECS",
                30,
                ConfigError::InvalidSessionCancelGrace,
            )?,
            worker_heartbeat_timeout: duration_var(
                "WORKER_HEARTBEAT_TIMEOUT_SECS",
                15,
                ConfigError::InvalidWorkerHeartbeatTimeout,
            )?,
        })
    }

    pub fn for_tests() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".parse().expect("valid socket address"),
            api_key: "test-public-key".to_string(),
            worker_internal_token: "test-internal-key".to_string(),
            livekit_api_key: "test-livekit-api-key".to_string(),
            livekit_api_secret: "test-livekit-secret".to_string(),
            model_name: "yume".to_string(),
            worker_id: "wm-worker-1".to_string(),
            modal_dispatch_base_url: "http://modal-dispatch.test".to_string(),
            modal_dispatch_token: "test-modal-token".to_string(),
            callback_base_url: "http://coordinator.test".to_string(),
            session_startup_timeout: Duration::from_secs(120),
            session_max_duration: Duration::from_secs(3600),
            session_cancel_grace: Duration::from_secs(30),
            worker_heartbeat_timeout: Duration::from_secs(15),
        }
    }
}

fn required(name: &'static str) -> Result<String, ConfigError> {
    env::var(name).map_err(|_| ConfigError::MissingVar(name))
}

fn duration_var<F>(
    name: &'static str,
    default_secs: u64,
    invalid: F,
) -> Result<Duration, ConfigError>
where
    F: FnOnce(String) -> ConfigError + Copy,
{
    match env::var(name) {
        Ok(raw) => {
            let secs = u64::from_str(&raw).map_err(|err| invalid(err.to_string()))?;
            Ok(Duration::from_secs(secs))
        }
        Err(_) => Ok(Duration::from_secs(default_secs)),
    }
}
