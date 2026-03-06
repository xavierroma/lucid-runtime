use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const VIDEO_TRACK_NAME: &str = "main_video";
pub const CONTROL_TOPIC: &str = "wm.control.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SessionState {
    Starting,
    Running,
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
    pub state: SessionState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_reason: Option<SessionEndReason>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session: Session,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_access_token: Option<String>,
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
