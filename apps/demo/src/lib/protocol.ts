const encoder = new TextEncoder()
const decoder = new TextDecoder()

export type ControlMessageType = "set_prompt" | "end" | "ping"
export type StatusMessageType =
  | "started"
  | "busy"
  | "frame_metrics"
  | "error"
  | "ended"
  | "pong"

export interface ControlEnvelope<TPayload extends Record<string, unknown>> {
  v: "v1"
  type: ControlMessageType
  seq: number
  ts_ms: number
  session_id: string | null
  payload: TPayload
}

export interface StatusEnvelope {
  v: string
  type: StatusMessageType
  seq: number
  ts_ms: number
  session_id: string | null
  payload: Record<string, unknown>
}

export function encodePromptMessage(args: {
  prompt: string
  seq: number
  sessionId: string
}) {
  const envelope: ControlEnvelope<{ prompt: string }> = {
    v: "v1",
    type: "set_prompt",
    seq: args.seq,
    ts_ms: Date.now(),
    session_id: args.sessionId,
    payload: {
      prompt: args.prompt,
    },
  }

  return encoder.encode(JSON.stringify(envelope))
}

export function decodeStatusMessage(payload: Uint8Array) {
  try {
    const parsed = JSON.parse(decoder.decode(payload)) as Partial<StatusEnvelope>

    if (
      parsed.v !== "v1" ||
      typeof parsed.type !== "string" ||
      typeof parsed.seq !== "number" ||
      typeof parsed.ts_ms !== "number" ||
      typeof parsed.payload !== "object" ||
      parsed.payload === null
    ) {
      return null
    }

    return {
      v: parsed.v,
      type: parsed.type as StatusMessageType,
      seq: parsed.seq,
      ts_ms: parsed.ts_ms,
      session_id: typeof parsed.session_id === "string" ? parsed.session_id : null,
      payload: parsed.payload,
    } satisfies StatusEnvelope
  } catch {
    return null
  }
}

export function formatTopicPayload(payload: Uint8Array) {
  try {
    return decoder.decode(payload)
  } catch {
    return "<binary payload>"
  }
}
