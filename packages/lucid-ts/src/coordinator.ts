import type { ModelsResponse, SessionResponse } from "./types.js"

interface ErrorResponse {
  error?: string
}

export interface CoordinatorClientOptions {
  baseUrl: string
  apiKey?: string
  fetch?: typeof globalThis.fetch
  headers?: HeadersInit
}

export class CoordinatorClient {
  readonly #baseUrl: string
  readonly #apiKey: string | null
  readonly #fetchImpl: typeof globalThis.fetch
  readonly #headers: HeadersInit | undefined

  constructor(options: CoordinatorClientOptions) {
    this.#baseUrl = options.baseUrl.replace(/\/$/, "")
    this.#apiKey = options.apiKey?.trim() || null
    this.#fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis)
    this.#headers = options.headers
  }

  getModels() {
    return this.#request<ModelsResponse>("/models")
  }

  createSession(modelName: string) {
    return this.#request<SessionResponse>("/sessions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model_name: modelName }),
    })
  }

  getSession(sessionId: string) {
    return this.#request<SessionResponse>(`/sessions/${sessionId}`)
  }

  endSession(sessionId: string) {
    return this.#request<void>(`/sessions/${sessionId}:end`, {
      method: "POST",
    })
  }

  async #request<T>(path: string, init?: RequestInit) {
    const headers = new Headers(this.#headers)
    const overrideHeaders = new Headers(init?.headers)
    overrideHeaders.forEach((value, key) => headers.set(key, value))
    if (this.#apiKey) {
      headers.set("Authorization", `Bearer ${this.#apiKey}`)
    }

    const response = await this.#fetchImpl(`${this.#baseUrl}${path}`, {
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
}

async function parseError(response: Response) {
  try {
    const payload = (await response.json()) as ErrorResponse
    if (payload.error) {
      return payload.error
    }
  } catch {
    // Fall back to the response status when the body is not valid JSON.
  }

  return response.statusText || `request failed with ${response.status}`
}
