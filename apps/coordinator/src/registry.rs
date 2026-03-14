use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::{
    capabilities,
    models::{Capabilities, SupportedModel},
};

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
    pub description: Option<String>,
    pub capabilities: Capabilities,
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
    #[error("{0}")]
    Manifest(String),
}

#[derive(Debug, Deserialize)]
struct ModelsFile {
    models: Vec<ModelFileEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelFileEntry {
    id: String,
    display_name: String,
    manifest_path: String,
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

    fn from_file_entries(parsed: ModelsFile, source_path: &Path) -> Result<Self, RegistryError> {
        if parsed.models.is_empty() {
            return Err(RegistryError::EmptyModels);
        }

        let mut seen_ids = HashSet::new();
        let mut models = Vec::with_capacity(parsed.models.len());
        let base_dir = source_path.parent().unwrap_or_else(|| Path::new("."));

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

            let manifest_path = resolve_path(base_dir, &entry.manifest_path);
            let manifest = capabilities::load_manifest_from_path(&manifest_path)
                .map_err(RegistryError::Manifest)?;
            let capabilities = capabilities::build_capabilities(&manifest);
            let description = manifest
                .get("model")
                .and_then(|model| model.get("description"))
                .and_then(Value::as_str)
                .map(|value| value.to_string());

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
                description,
                capabilities,
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
                description: model.description.clone(),
            })
            .collect()
    }

    pub fn for_tests() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let manifests_dir = root.join("packages/contracts/generated");
        let dir = std::env::temp_dir().join(format!(
            "lucid-coordinator-registry-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).expect("test registry dir should create");
        let source = dir.join("coordinator.models.json");
        let contents = format!(
            r#"{{
  "models": [
    {{
      "id": "yume",
      "display_name": "Yume",
      "manifest_path": "{}",
      "backend": {{
        "kind": "modal",
        "dispatch_base_url": "https://yume.modal.test",
        "dispatch_token": "token-yume",
        "worker_id": "wm-yume"
      }},
      "timeouts": {{
        "startup_timeout_secs": 120,
        "session_max_duration_secs": 3600,
        "session_cancel_grace_secs": 30,
        "worker_heartbeat_timeout_secs": 15
      }}
    }},
    {{
      "id": "waypoint",
      "display_name": "Waypoint",
      "manifest_path": "{}",
      "backend": {{
        "kind": "modal",
        "dispatch_base_url": "https://waypoint.modal.test",
        "dispatch_token": "token-waypoint",
        "worker_id": "wm-waypoint"
      }},
      "timeouts": {{
        "startup_timeout_secs": 900,
        "session_max_duration_secs": 3600,
        "session_cancel_grace_secs": 30,
        "worker_heartbeat_timeout_secs": 15
      }}
    }},
    {{
      "id": "helios",
      "display_name": "Helios (Distilled)",
      "manifest_path": "{}",
      "backend": {{
        "kind": "modal",
        "dispatch_base_url": "https://helios.modal.test",
        "dispatch_token": "token-helios",
        "worker_id": "wm-helios"
      }},
      "timeouts": {{
        "startup_timeout_secs": 900,
        "session_max_duration_secs": 3600,
        "session_cancel_grace_secs": 30,
        "worker_heartbeat_timeout_secs": 15
      }}
    }}
  ]
}}"#,
            manifests_dir.join("lucid_manifest.json").display(),
            manifests_dir.join("lucid_manifest.waypoint.json").display(),
            manifests_dir.join("lucid_manifest.helios.json").display(),
        );
        fs::write(&source, contents).expect("test models file should write");
        Self::from_path(&source).expect("test models registry should load")
    }
}

fn resolve_path(base_dir: &Path, raw_path: &str) -> PathBuf {
    let candidate = PathBuf::from(raw_path.trim());
    if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
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

    fn manifest_json(model_name: &str) -> String {
        format!(
            r#"{{
  "model": {{
    "name": "{model_name}",
    "description": "desc"
  }},
  "inputs": [],
  "outputs": [
    {{
      "name": "main_video",
      "kind": "video"
    }}
  ]
}}"#
        )
    }

    #[test]
    fn valid_registry_loads() {
        let dir = temp_dir();
        write_file(&dir, "contracts/yume.json", &manifest_json("yume"));
        let registry_path = write_file(
            &dir,
            "coordinator.models.json",
            r#"{
  "models": [
    {
      "id": "yume",
      "display_name": "Yume",
      "manifest_path": "contracts/yume.json",
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
        write_file(&dir, "contracts/a.json", &manifest_json("a"));
        write_file(&dir, "contracts/b.json", &manifest_json("b"));
        let registry_path = write_file(
            &dir,
            "coordinator.models.json",
            r#"{
  "models": [
    {
      "id": "dup",
      "display_name": "First",
      "manifest_path": "contracts/a.json",
      "backend": {"kind": "modal", "dispatch_base_url": "https://a", "dispatch_token": "a", "worker_id": "a"},
      "timeouts": {"startup_timeout_secs": 1, "session_max_duration_secs": 2, "session_cancel_grace_secs": 3, "worker_heartbeat_timeout_secs": 4}
    },
    {
      "id": "dup",
      "display_name": "Second",
      "manifest_path": "contracts/b.json",
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
    fn missing_manifest_path_fails() {
        let dir = temp_dir();
        let registry_path = write_file(
            &dir,
            "coordinator.models.json",
            r#"{
  "models": [
    {
      "id": "yume",
      "display_name": "Yume",
      "manifest_path": "contracts/missing.json",
      "backend": {"kind": "modal", "dispatch_base_url": "https://yume", "dispatch_token": "token", "worker_id": "worker"},
      "timeouts": {"startup_timeout_secs": 1, "session_max_duration_secs": 2, "session_cancel_grace_secs": 3, "worker_heartbeat_timeout_secs": 4}
    }
  ]
}"#,
        );

        let err = ModelRegistry::from_path(&registry_path).expect_err("registry should fail");
        assert!(
            matches!(err, RegistryError::Manifest(message) if message.contains("failed reading manifest"))
        );
    }

    #[test]
    fn unsupported_backend_kind_fails() {
        let dir = temp_dir();
        write_file(&dir, "contracts/yume.json", &manifest_json("yume"));
        let registry_path = write_file(
            &dir,
            "coordinator.models.json",
            r#"{
  "models": [
    {
      "id": "yume",
      "display_name": "Yume",
      "manifest_path": "contracts/yume.json",
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
