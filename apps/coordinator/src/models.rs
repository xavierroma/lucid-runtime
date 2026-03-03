use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const VIDEO_TRACK_NAME: &str = "main_video";
pub const CONTROL_TOPIC: &str = "wm.control.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SessionState {
    Created,
    Assigned,
    Running,
    Ended,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Session {
    pub session_id: Uuid,
    pub room_name: String,
    pub state: SessionState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session: Session,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_access_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_access_token: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WorkerRegisterRequest {
    pub worker_id: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WorkerHeartbeatRequest {
    pub worker_id: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WorkerAssignmentQuery {
    pub worker_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkerAssignmentResponse {
    pub session_id: Uuid,
    pub room_name: String,
    pub worker_access_token: String,
    pub video_track_name: String,
    pub control_topic: String,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct WorkerEndedRequest {
    #[serde(default)]
    pub error_code: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkerStatusResponse {
    pub status: &'static str,
}
