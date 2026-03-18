use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const LAUNCH_ENDPOINT: &str = "/launch";
pub const CANCEL_ENDPOINT: &str = "/cancel";
pub const STATUS_ENDPOINT: &str = "/status";
pub const MANIFEST_ENDPOINT: &str = "/manifest";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModalExecutionStatus {
    Pending,
    Success,
    Failure,
    InitFailure,
    Terminated,
    Timeout,
    NotFound,
}

#[derive(Clone, Debug, Serialize)]
pub struct LaunchSessionRequest {
    pub session_id: Uuid,
    pub room_name: String,
    pub worker_id: String,
    pub worker_access_token: String,
    pub control_topic: String,
    pub coordinator_base_url: String,
    pub coordinator_internal_token: String,
}

#[derive(Clone, Debug, Serialize)]
struct CancelSessionRequest<'a> {
    function_call_id: &'a str,
    force: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct LaunchSessionResponse {
    function_call_id: String,
}

#[derive(Clone, Debug, Deserialize)]
struct StatusResponse {
    status: ModalExecutionStatus,
}

#[derive(Clone, Debug, Deserialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Error)]
pub enum ModalDispatchError {
    #[error("modal request failed: {0}")]
    Transport(String),
    #[error("modal endpoint returned {status}: {message}")]
    Upstream { status: u16, message: String },
    #[error("modal response decode failed: {0}")]
    Decode(String),
}

#[async_trait]
pub trait ModalDispatch: Send + Sync {
    async fn launch_session(
        &self,
        payload: LaunchSessionRequest,
    ) -> Result<String, ModalDispatchError>;
    async fn cancel_session(
        &self,
        function_call_id: &str,
        force: bool,
    ) -> Result<(), ModalDispatchError>;
    async fn get_session_status(
        &self,
        function_call_id: &str,
    ) -> Result<ModalExecutionStatus, ModalDispatchError>;
    async fn get_manifest(&self) -> Result<serde_json::Value, ModalDispatchError>;
}

#[derive(Clone)]
pub struct HttpModalDispatchClient {
    base_url: String,
    token: String,
    client: reqwest::Client,
}

impl HttpModalDispatchClient {
    pub fn new(base_url: String, token: String) -> Self {
        Self {
            base_url,
            token,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ModalDispatch for HttpModalDispatchClient {
    async fn launch_session(
        &self,
        payload: LaunchSessionRequest,
    ) -> Result<String, ModalDispatchError> {
        let url = format!("{}{}", self.base_url, LAUNCH_ENDPOINT);
        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.token)
            .json(&payload)
            .send()
            .await
            .map_err(|err| ModalDispatchError::Transport(err.to_string()))?;
        if response.status().is_success() {
            let parsed = response
                .json::<LaunchSessionResponse>()
                .await
                .map_err(|err| ModalDispatchError::Decode(err.to_string()))?;
            return Ok(parsed.function_call_id);
        }
        Err(parse_error(response).await)
    }

    async fn cancel_session(
        &self,
        function_call_id: &str,
        force: bool,
    ) -> Result<(), ModalDispatchError> {
        let url = format!("{}{}", self.base_url, CANCEL_ENDPOINT);
        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.token)
            .json(&CancelSessionRequest {
                function_call_id,
                force,
            })
            .send()
            .await
            .map_err(|err| ModalDispatchError::Transport(err.to_string()))?;
        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            return Ok(());
        }
        Err(parse_error(response).await)
    }

    async fn get_session_status(
        &self,
        function_call_id: &str,
    ) -> Result<ModalExecutionStatus, ModalDispatchError> {
        let url = format!(
            "{}/{function_call_id}",
            format!("{}{}", self.base_url, STATUS_ENDPOINT)
        );
        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|err| ModalDispatchError::Transport(err.to_string()))?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(ModalExecutionStatus::NotFound);
        }
        if response.status().is_success() {
            let parsed = response
                .json::<StatusResponse>()
                .await
                .map_err(|err| ModalDispatchError::Decode(err.to_string()))?;
            return Ok(parsed.status);
        }
        Err(parse_error(response).await)
    }

    async fn get_manifest(&self) -> Result<serde_json::Value, ModalDispatchError> {
        let url = format!("{}{}", self.base_url, MANIFEST_ENDPOINT);
        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|err| ModalDispatchError::Transport(err.to_string()))?;
        if response.status().is_success() {
            return response
                .json::<serde_json::Value>()
                .await
                .map_err(|err| ModalDispatchError::Decode(err.to_string()));
        }
        Err(parse_error(response).await)
    }
}

async fn parse_error(response: reqwest::Response) -> ModalDispatchError {
    let status = response.status().as_u16();
    let body = response
        .json::<ErrorResponse>()
        .await
        .map(|decoded| decoded.error)
        .unwrap_or_else(|_| "upstream request failed".to_string());
    ModalDispatchError::Upstream {
        status,
        message: body,
    }
}
