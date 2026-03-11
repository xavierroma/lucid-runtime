import { demoEnv } from "@/lib/env"

export type SessionState =
  | "STARTING"
  | "READY"
  | "RUNNING"
  | "CANCELING"
  | "ENDED"
  | "FAILED"

export interface SessionRecord {
  session_id: string
  room_name: string
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

export interface ManifestAction {
  name: string
  description?: string | null
  mode: string
  args_schema: Record<string, unknown>
}

export interface ManifestOutput {
  name: string
  kind: string
  width?: number
  height?: number
  fps?: number
  pixel_format?: string
}

export interface LucidManifest {
  model: {
    name: string
    description?: string | null
  }
  actions: ManifestAction[]
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

interface ErrorResponse {
  error?: string
}

function buildUrl(path: string) {
  const base = demoEnv.coordinatorBaseUrl.replace(/\/$/, "")
  return `${base}${path}`
}

async function parseError(response: Response) {
  try {
    const payload = (await response.json()) as ErrorResponse
    if (payload.error) {
      return payload.error
    }
  } catch {
    // Ignore JSON parse failures and fall back to status text.
  }

  return response.statusText || `request failed with ${response.status}`
}

async function request<T>(path: string, init?: RequestInit) {
  const headers = new Headers(init?.headers)

  if (demoEnv.coordinatorApiKey) {
    headers.set("Authorization", `Bearer ${demoEnv.coordinatorApiKey}`)
  }

  const response = await fetch(buildUrl(path), {
    ...init,
    headers,
  })

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export async function createSession(modelName?: string) {
  return request<SessionResponse>("/sessions", {
    method: "POST",
    headers: modelName
      ? {
          "Content-Type": "application/json",
        }
      : undefined,
    body: modelName ? JSON.stringify({ model_name: modelName }) : undefined,
  })
}

export async function getSession(sessionId: string) {
  return request<SessionResponse>(`/sessions/${sessionId}`)
}

export async function endSession(sessionId: string) {
  return request<void>(`/sessions/${sessionId}:end`, {
    method: "POST",
  })
}

export function isTerminalSessionState(state: SessionState) {
  return state === "ENDED" || state === "FAILED"
}
