use std::{env, net::SocketAddr, str::FromStr, time::Duration};

use thiserror::Error;

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub api_key: String,
    pub worker_internal_token: String,
    pub livekit_api_key: String,
    pub livekit_api_secret: String,
    pub worker_id: String,
    pub heartbeat_ttl: Duration,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing required environment variable: {0}")]
    MissingVar(&'static str),
    #[error("invalid COORDINATOR_BIND_ADDR: {0}")]
    InvalidBindAddr(String),
    #[error("invalid HEARTBEAT_TTL_SECS: {0}")]
    InvalidHeartbeatTtl(String),
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let bind_addr = env::var("COORDINATOR_BIND_ADDR")
            .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
            .parse()
            .map_err(|err: std::net::AddrParseError| {
                ConfigError::InvalidBindAddr(err.to_string())
            })?;

        let heartbeat_ttl = match env::var("HEARTBEAT_TTL_SECS") {
            Ok(raw) => {
                let secs = u64::from_str(&raw)
                    .map_err(|err| ConfigError::InvalidHeartbeatTtl(err.to_string()))?;
                Duration::from_secs(secs)
            }
            Err(_) => Duration::from_secs(15),
        };

        Ok(Self {
            bind_addr,
            api_key: required("API_KEY")?,
            worker_internal_token: required("WORKER_INTERNAL_TOKEN")?,
            livekit_api_key: required("LIVEKIT_API_KEY")?,
            livekit_api_secret: required("LIVEKIT_API_SECRET")?,
            worker_id: env::var("WORKER_ID").unwrap_or_else(|_| "wm-worker-1".to_string()),
            heartbeat_ttl,
        })
    }

    pub fn for_tests() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".parse().expect("valid socket address"),
            api_key: "test-public-key".to_string(),
            worker_internal_token: "test-internal-key".to_string(),
            livekit_api_key: "test-livekit-api-key".to_string(),
            livekit_api_secret: "test-livekit-secret".to_string(),
            worker_id: "wm-worker-1".to_string(),
            heartbeat_ttl: Duration::from_secs(15),
        }
    }
}

fn required(name: &'static str) -> Result<String, ConfigError> {
    env::var(name).map_err(|_| ConfigError::MissingVar(name))
}
