export type SessionState =
  | "STARTING"
  | "READY"
  | "RUNNING"
  | "PAUSED"
  | "CANCELING"
  | "ENDED"
  | "FAILED"

export interface SessionRecord {
  session_id: string
  room_name: string
  model_name: string
  state: SessionState
  error_code?: string | null
  end_reason?: string | null
}

export interface OutputBinding {
  name: string
  kind: string
  track_name?: string | null
  topic?: string | null
}

export interface HoldInputBinding {
  kind: "hold"
  keys: string[]
  mouse_buttons: number[]
}

export interface PressInputBinding {
  kind: "press"
  keys: string[]
  mouse_buttons: number[]
}

export interface AxisInputBinding {
  kind: "axis"
  positive_keys: string[]
  negative_keys: string[]
}

export interface PointerInputBinding {
  kind: "pointer"
  pointer_lock: boolean
}

export interface WheelInputBinding {
  kind: "wheel"
  step: number
}

export type InputBinding =
  | HoldInputBinding
  | PressInputBinding
  | AxisInputBinding
  | PointerInputBinding
  | WheelInputBinding

export type JsonSchema = Record<string, unknown>

export interface ManifestInput {
  name: string
  description?: string | null
  args_schema: JsonSchema
  binding?: InputBinding
}

export interface ManifestOutput {
  name: string
  kind: string
  width?: number
  height?: number
  fps?: number
  pixel_format?: string
  sample_rate_hz?: number
  channels?: number
  max_bytes?: number
}

export interface LucidManifest {
  model: {
    name: string
    description?: string | null
  }
  inputs: ManifestInput[]
  outputs: ManifestOutput[]
}

export interface Capabilities {
  control_topic: string
  status_topic: string
  manifest: LucidManifest
  output_bindings: OutputBinding[]
}

export interface SessionResponse {
  session: SessionRecord
  client_access_token?: string | null
  capabilities: Capabilities
}

export interface SupportedModel {
  id: string
  display_name: string
  description?: string | null
}

export interface ModelsResponse {
  models: SupportedModel[]
}

export interface UploadFieldConfig {
  kind: string
  mime_types: string[]
  max_bytes: number
  target_width?: number
  target_height?: number
}

export interface UploadInputSpec {
  name: string
  argName: string
  upload: UploadFieldConfig
}

export function isTerminalSessionState(state: SessionState) {
  return state === "ENDED" || state === "FAILED"
}
