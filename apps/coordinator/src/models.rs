use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub const CONTROL_TOPIC: &str = "wm.control";
pub const STATUS_TOPIC: &str = "wm.status";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SessionState {
    Starting,
    Ready,
    Running,
    Paused,
    Canceling,
    Ended,
    Failed,
}

impl SessionState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Ended | Self::Failed)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SessionEndReason {
    NormalCompletion,
    ClientRequested,
    ControlRequested,
    WorkerReportedError,
    StartupTimeout,
    SessionTimeout,
    CancelTimeout,
    WorkerHeartbeatTimeout,
    ModalSuccess,
    ModalFailure,
    ModalInitFailure,
    ModalTerminated,
    ModalTimeout,
    ModalNotFound,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Session {
    pub session_id: Uuid,
    pub room_name: String,
    pub model_name: String,
    pub state: SessionState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_reason: Option<SessionEndReason>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputBinding {
    pub name: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub control_topic: String,
    pub status_topic: String,
    pub manifest: Value,
    pub output_bindings: Vec<OutputBinding>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionResponse {
    pub session: Session,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_access_token: Option<String>,
    pub capabilities: Capabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedModel {
    pub id: String,
    pub display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsResponse {
    pub models: Vec<SupportedModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct RuntimeEndedRequest {
    #[serde(default)]
    pub error_code: Option<String>,
    #[serde(default)]
    pub end_reason: Option<SessionEndReason>,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct CreateSessionRequest {
    #[serde(default)]
    pub model_name: Option<String>,
}
