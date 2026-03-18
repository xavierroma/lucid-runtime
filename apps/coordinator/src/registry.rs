use std::{collections::HashSet, fs, path::Path};

use serde::Deserialize;
use thiserror::Error;

use crate::models::SupportedModel;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelTimeouts {
    pub startup_timeout_secs: u64,
    pub session_max_duration_secs: u64,
    pub session_cancel_grace_secs: u64,
    pub worker_heartbeat_timeout_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModalBackend {
    pub dispatch_base_url: String,
    pub dispatch_token: String,
    pub worker_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelBackend {
    Modal(ModalBackend),
}

#[derive(Clone, Debug)]
pub struct RegisteredModel {
    pub id: String,
    pub display_name: String,
    pub backend: ModelBackend,
    pub timeouts: ModelTimeouts,
}

#[derive(Clone, Debug)]
pub struct ModelRegistry {
    models: Vec<RegisteredModel>,
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("failed reading models file {path}: {message}")]
    ReadModelsFile { path: String, message: String },
    #[error("invalid models JSON {path}: {message}")]
    ParseModelsFile { path: String, message: String },
    #[error("models list must contain at least one entry")]
    EmptyModels,
    #[error("duplicate model id: {0}")]
    DuplicateModelId(String),
    #[error("model id cannot be empty")]
    EmptyModelId,
    #[error("display_name cannot be empty for model {0}")]
    EmptyDisplayName(String),
}

#[derive(Debug, Deserialize)]
struct ModelsFile {
    models: Vec<ModelFileEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelFileEntry {
    id: String,
    display_name: String,
    backend: BackendFileEntry,
    timeouts: TimeoutsFileEntry,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BackendFileEntry {
    Modal {
        dispatch_base_url: String,
        dispatch_token: String,
        worker_id: String,
    },
}

#[derive(Debug, Deserialize)]
struct TimeoutsFileEntry {
    startup_timeout_secs: u64,
    session_max_duration_secs: u64,
    session_cancel_grace_secs: u64,
    worker_heartbeat_timeout_secs: u64,
}

impl ModelRegistry {
    pub fn from_path(path: &Path) -> Result<Self, RegistryError> {
        let raw = fs::read_to_string(path).map_err(|err| RegistryError::ReadModelsFile {
            path: path.display().to_string(),
            message: err.to_string(),
        })?;
        let parsed = serde_json::from_str::<ModelsFile>(&raw).map_err(|err| {
            RegistryError::ParseModelsFile {
                path: path.display().to_string(),
                message: err.to_string(),
            }
        })?;
        Self::from_file_entries(parsed, path)
    }

    fn from_file_entries(parsed: ModelsFile, _source_path: &Path) -> Result<Self, RegistryError> {
        if parsed.models.is_empty() {
            return Err(RegistryError::EmptyModels);
        }

        let mut seen_ids = HashSet::new();
        let mut models = Vec::with_capacity(parsed.models.len());

        for entry in parsed.models {
            let id = entry.id.trim().to_lowercase();
            if id.is_empty() {
                return Err(RegistryError::EmptyModelId);
            }
            if !seen_ids.insert(id.clone()) {
                return Err(RegistryError::DuplicateModelId(id));
            }

            let display_name = entry.display_name.trim().to_string();
            if display_name.is_empty() {
                return Err(RegistryError::EmptyDisplayName(id));
            }

            let backend = match entry.backend {
                BackendFileEntry::Modal {
                    dispatch_base_url,
                    dispatch_token,
                    worker_id,
                } => ModelBackend::Modal(ModalBackend {
                    dispatch_base_url: dispatch_base_url.trim_end_matches('/').to_string(),
                    dispatch_token,
                    worker_id,
                }),
            };

            models.push(RegisteredModel {
                id,
                display_name,
                backend,
                timeouts: ModelTimeouts {
                    startup_timeout_secs: entry.timeouts.startup_timeout_secs,
                    session_max_duration_secs: entry.timeouts.session_max_duration_secs,
                    session_cancel_grace_secs: entry.timeouts.session_cancel_grace_secs,
                    worker_heartbeat_timeout_secs: entry.timeouts.worker_heartbeat_timeout_secs,
                },
            });
        }

        Ok(Self { models })
    }

    pub fn get(&self, id: &str) -> Option<&RegisteredModel> {
        self.models.iter().find(|model| model.id == id)
    }

    pub fn models(&self) -> &[RegisteredModel] {
        &self.models
    }

    pub fn supported_models(&self) -> Vec<SupportedModel> {
        self.models
            .iter()
            .map(|model| SupportedModel {
                id: model.id.clone(),
                display_name: model.display_name.clone(),
                description: None,
            })
            .collect()
    }

    pub fn for_tests() -> Self {
        Self {
            models: vec![
                RegisteredModel {
                    id: "yume".to_string(),
                    display_name: "Yume".to_string(),
                    backend: ModelBackend::Modal(ModalBackend {
                        dispatch_base_url: "https://yume.modal.test".to_string(),
                        dispatch_token: "token-yume".to_string(),
                        worker_id: "wm-yume".to_string(),
                    }),
                    timeouts: ModelTimeouts {
                        startup_timeout_secs: 120,
                        session_max_duration_secs: 3600,
                        session_cancel_grace_secs: 30,
                        worker_heartbeat_timeout_secs: 15,
                    },
                },
                RegisteredModel {
                    id: "waypoint".to_string(),
                    display_name: "Waypoint".to_string(),
                    backend: ModelBackend::Modal(ModalBackend {
                        dispatch_base_url: "https://waypoint.modal.test".to_string(),
                        dispatch_token: "token-waypoint".to_string(),
                        worker_id: "wm-waypoint".to_string(),
                    }),
                    timeouts: ModelTimeouts {
                        startup_timeout_secs: 900,
                        session_max_duration_secs: 3600,
                        session_cancel_grace_secs: 30,
                        worker_heartbeat_timeout_secs: 15,
                    },
                },
                RegisteredModel {
                    id: "helios".to_string(),
                    display_name: "Helios (Distilled)".to_string(),
                    backend: ModelBackend::Modal(ModalBackend {
                        dispatch_base_url: "https://helios.modal.test".to_string(),
                        dispatch_token: "token-helios".to_string(),
                        worker_id: "wm-helios".to_string(),
                    }),
                    timeouts: ModelTimeouts {
                        startup_timeout_secs: 900,
                        session_max_duration_secs: 3600,
                        session_cancel_grace_secs: 30,
                        worker_heartbeat_timeout_secs: 15,
                    },
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{ModelRegistry, RegistryError};

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!("lucid-coordinator-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("temp dir should create");
        path
    }

    fn write_file(dir: &PathBuf, relative: &str, contents: &str) -> PathBuf {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent dir should create");
        }
        fs::write(&path, contents).expect("file should write");
        path
    }

    #[test]
    fn valid_registry_loads() {
        let dir = temp_dir();
        let registry_path = write_file(
            &dir,
            "coordinator.models.json",
            r#"{
  "models": [
    {
      "id": "yume",
      "display_name": "Yume",
      "backend": {
        "kind": "modal",
        "dispatch_base_url": "https://yume.modal.run",
        "dispatch_token": "token-yume",
        "worker_id": "wm-yume"
      },
      "timeouts": {
        "startup_timeout_secs": 120,
        "session_max_duration_secs": 3600,
        "session_cancel_grace_secs": 30,
        "worker_heartbeat_timeout_secs": 15
      }
    }
  ]
}"#,
        );

        let registry = ModelRegistry::from_path(&registry_path).expect("registry should load");
        assert_eq!(registry.models().len(), 1);
        assert_eq!(registry.models()[0].id, "yume");
        assert_eq!(registry.supported_models()[0].display_name, "Yume");
    }

    #[test]
    fn duplicate_model_ids_fail() {
        let dir = temp_dir();
        let registry_path = write_file(
            &dir,
            "coordinator.models.json",
            r#"{
  "models": [
    {
      "id": "dup",
      "display_name": "First",
      "backend": {"kind": "modal", "dispatch_base_url": "https://a", "dispatch_token": "a", "worker_id": "a"},
      "timeouts": {"startup_timeout_secs": 1, "session_max_duration_secs": 2, "session_cancel_grace_secs": 3, "worker_heartbeat_timeout_secs": 4}
    },
    {
      "id": "dup",
      "display_name": "Second",
      "backend": {"kind": "modal", "dispatch_base_url": "https://b", "dispatch_token": "b", "worker_id": "b"},
      "timeouts": {"startup_timeout_secs": 1, "session_max_duration_secs": 2, "session_cancel_grace_secs": 3, "worker_heartbeat_timeout_secs": 4}
    }
  ]
}"#,
        );

        let err = ModelRegistry::from_path(&registry_path).expect_err("registry should fail");
        assert!(matches!(err, RegistryError::DuplicateModelId(id) if id == "dup"));
    }

    #[test]
    fn unsupported_backend_kind_fails() {
        let dir = temp_dir();
        let registry_path = write_file(
            &dir,
            "coordinator.models.json",
            r#"{
  "models": [
    {
      "id": "yume",
      "display_name": "Yume",
      "backend": {"kind": "ec2"},
      "timeouts": {"startup_timeout_secs": 1, "session_max_duration_secs": 2, "session_cancel_grace_secs": 3, "worker_heartbeat_timeout_secs": 4}
    }
  ]
}"#,
        );

        let err = ModelRegistry::from_path(&registry_path).expect_err("registry should fail");
        assert!(matches!(err, RegistryError::ParseModelsFile { .. }));
    }
}
