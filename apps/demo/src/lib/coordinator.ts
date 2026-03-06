import { demoEnv } from "@/lib/env"

export type SessionState = "CREATED" | "RUNNING" | "ENDED"

export interface SessionRecord {
  session_id: string
  room_name: string
  state: SessionState
  error_code?: string | null
}

export interface CreateSessionResponse {
  session: SessionRecord
  client_access_token?: string | null
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

export async function createSession() {
  return request<CreateSessionResponse>("/v1/sessions", {
    method: "POST",
  })
}

export async function getSession(sessionId: string) {
  return request<SessionRecord>(`/v1/sessions/${sessionId}`)
}

export async function endSession(sessionId: string) {
  return request<void>(`/v1/sessions/${sessionId}:end`, {
    method: "POST",
  })
}
