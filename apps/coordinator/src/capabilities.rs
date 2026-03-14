use std::{fs, path::Path};

use serde_json::Value;

use crate::models::{Capabilities, OutputBinding, CONTROL_TOPIC, STATUS_TOPIC};

pub fn load_manifest_from_path(path: &Path) -> Result<Value, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading manifest {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("invalid manifest JSON {}: {err}", path.display()))
}

pub fn build_capabilities(manifest: &Value) -> Capabilities {
    let output_bindings = manifest
        .get("outputs")
        .and_then(Value::as_array)
        .map(|outputs| {
            outputs
                .iter()
                .filter_map(|output| {
                    let name = output.get("name")?.as_str()?.to_string();
                    let kind = output.get("kind")?.as_str()?.to_string();
                    let mut binding = OutputBinding {
                        name,
                        kind: kind.clone(),
                        track_name: None,
                        topic: None,
                    };
                    if kind == "video" || kind == "audio" {
                        binding.track_name = Some(binding.name.clone());
                    } else {
                        binding.topic = Some(format!("wm.output.{}", binding.name));
                    }
                    Some(binding)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Capabilities {
        control_topic: CONTROL_TOPIC.to_string(),
        status_topic: STATUS_TOPIC.to_string(),
        manifest: manifest.clone(),
        output_bindings,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::build_capabilities;

    #[test]
    fn capabilities_include_track_binding_for_video_outputs() {
        let capabilities = build_capabilities(&json!({
            "model": {"name": "helios"},
            "outputs": [
                {"name": "main_video", "kind": "video"}
            ]
        }));

        assert_eq!(
            capabilities.manifest["model"]["name"],
            Value::String("helios".to_string())
        );
        assert_eq!(capabilities.output_bindings.len(), 1);
        assert_eq!(capabilities.output_bindings[0].name, "main_video");
        assert_eq!(
            capabilities.output_bindings[0].track_name.as_deref(),
            Some("main_video")
        );
    }
}
