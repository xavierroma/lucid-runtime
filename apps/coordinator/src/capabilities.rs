use std::sync::OnceLock;

use serde_json::Value;

use crate::models::{Capabilities, OutputBinding, CONTROL_TOPIC, STATUS_TOPIC};

static YUME_MANIFEST: OnceLock<Value> = OnceLock::new();
static WAYPOINT_MANIFEST: OnceLock<Value> = OnceLock::new();

fn parse_manifest(path: &str) -> Value {
    serde_json::from_str(path).expect("embedded lucid manifest should be valid JSON")
}

pub fn manifest(model_name: &str) -> Value {
    match model_name {
        "waypoint" => WAYPOINT_MANIFEST
            .get_or_init(|| {
                parse_manifest(include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../packages/contracts/generated/lucid_manifest.waypoint.json"
                )))
            })
            .clone(),
        "yume" => YUME_MANIFEST
            .get_or_init(|| {
                parse_manifest(include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../packages/contracts/generated/lucid_manifest.json"
                )))
            })
            .clone(),
        other => {
            tracing::warn!(
                model_name = other,
                "unknown coordinator model; falling back to yume"
            );
            YUME_MANIFEST
                .get_or_init(|| {
                    parse_manifest(include_str!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../packages/contracts/generated/lucid_manifest.json"
                    )))
                })
                .clone()
        }
    }
}

pub fn build_capabilities(model_name: &str) -> Capabilities {
    let manifest = manifest(model_name);
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
        manifest,
        output_bindings,
    }
}
