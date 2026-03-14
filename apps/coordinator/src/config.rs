use std::{env, net::SocketAddr};

use thiserror::Error;

use crate::registry::ModelRegistry;

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub api_key: String,
    pub worker_internal_token: String,
    pub livekit_api_key: String,
    pub livekit_api_secret: String,
    pub callback_base_url: String,
    pub model_registry: ModelRegistry,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing required environment variable: {0}")]
    MissingVar(&'static str),
    #[error("invalid COORDINATOR_BIND_ADDR: {0}")]
    InvalidBindAddr(String),
    #[error("invalid COORDINATOR_MODELS_FILE: {0}")]
    InvalidModelsFile(String),
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
            callback_base_url: required("COORDINATOR_CALLBACK_BASE_URL")?
                .trim_end_matches('/')
                .to_string(),
            model_registry: ModelRegistry::from_path(&std::path::PathBuf::from(required(
                "COORDINATOR_MODELS_FILE",
            )?))
            .map_err(|err| ConfigError::InvalidModelsFile(err.to_string()))?,
        })
    }

    pub fn for_tests() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".parse().expect("valid socket address"),
            api_key: "test-public-key".to_string(),
            worker_internal_token: "test-internal-key".to_string(),
            livekit_api_key: "test-livekit-api-key".to_string(),
            livekit_api_secret: "test-livekit-secret".to_string(),
            callback_base_url: "http://coordinator.test".to_string(),
            model_registry: ModelRegistry::for_tests(),
        }
    }
}

fn required(name: &'static str) -> Result<String, ConfigError> {
    env::var(name).map_err(|_| ConfigError::MissingVar(name))
}
