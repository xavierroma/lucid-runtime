export const DEFAULT_CONTROL_TOPIC = "wm.control"
export const DEFAULT_STATUS_TOPIC = "wm.status"
export const DEFAULT_INPUT_FILE_TOPIC = "wm.input.file"

export type ControlMessageType = "end" | "pause" | "resume"

export interface ActionEnvelope<
  TName extends string = string,
  TArgs extends Record<string, unknown> = Record<string, unknown>,
> {
  type: "action"
  seq: number
  ts_ms: number
  session_id: string | null
  payload: {
    name: TName
    args: TArgs
  }
}

export interface ControlEnvelope {
  type: ControlMessageType
  seq: number
  ts_ms: number
  session_id: string | null
  payload: Record<string, never>
}

export interface PingEnvelope {
  type: "ping"
  seq: number
  ts_ms: number
  session_id: string | null
  payload: {
    client_ts_ms?: number
  }
}

export type LucidControlEnvelope = ActionEnvelope | ControlEnvelope | PingEnvelope

export interface StatusEnvelope<
  TType extends string = string,
  TPayload extends Record<string, unknown> = Record<string, unknown>,
> {
  type: TType
  seq: number
  ts_ms: number
  session_id: string | null
  payload: TPayload
}

const encoder = new TextEncoder()
const decoder = new TextDecoder()

export function buildOutputTopic(outputName: string) {
  return `wm.output.${outputName}`
}

export function createInputEnvelope<
  TName extends string,
  TArgs extends Record<string, unknown>,
>(args: {
  name: TName
  args: TArgs
  seq: number
  sessionId: string | null
  tsMs?: number
}): ActionEnvelope<TName, TArgs> {
  return {
    type: "action",
    seq: args.seq,
    ts_ms: args.tsMs ?? Date.now(),
    session_id: args.sessionId,
    payload: {
      name: args.name,
      args: args.args,
    },
  }
}

export function encodeInputMessage<
  TName extends string,
  TArgs extends Record<string, unknown>,
>(args: {
  name: TName
  args: TArgs
  seq: number
  sessionId: string | null
  tsMs?: number
}) {
  return encoder.encode(JSON.stringify(createInputEnvelope(args)))
}

export function createControlEnvelope(args: {
  type: ControlMessageType
  seq: number
  sessionId: string | null
  tsMs?: number
}): ControlEnvelope {
  return {
    type: args.type,
    seq: args.seq,
    ts_ms: args.tsMs ?? Date.now(),
    session_id: args.sessionId,
    payload: {},
  }
}

export function encodeControlMessage(args: {
  type: ControlMessageType
  seq: number
  sessionId: string | null
  tsMs?: number
}) {
  return encoder.encode(JSON.stringify(createControlEnvelope(args)))
}

export function createPingEnvelope(args: {
  seq: number
  sessionId: string | null
  clientTsMs?: number
  tsMs?: number
}): PingEnvelope {
  const payload: PingEnvelope["payload"] = {}
  if (args.clientTsMs !== undefined) {
    payload.client_ts_ms = args.clientTsMs
  } else {
    payload.client_ts_ms = Date.now()
  }
  return {
    type: "ping",
    seq: args.seq,
    ts_ms: args.tsMs ?? Date.now(),
    session_id: args.sessionId,
    payload,
  }
}

export function encodePingMessage(args: {
  seq: number
  sessionId: string | null
  clientTsMs?: number
  tsMs?: number
}) {
  return encoder.encode(JSON.stringify(createPingEnvelope(args)))
}

export function decodeControlMessage(raw: Uint8Array | string) {
  const payload = parseEnvelope(raw)
  if (!payload || typeof payload.type !== "string") {
    return null
  }
  if (!("payload" in payload) || typeof payload.payload !== "object" || payload.payload === null) {
    return null
  }
  return payload as unknown as LucidControlEnvelope
}

export function decodeStatusMessage<
  TType extends string = string,
  TPayload extends Record<string, unknown> = Record<string, unknown>,
>(raw: Uint8Array | string) {
  const payload = parseEnvelope(raw)
  if (!payload || typeof payload.type !== "string") {
    return null
  }
  if (!("payload" in payload) || typeof payload.payload !== "object" || payload.payload === null) {
    return null
  }
  return payload as unknown as StatusEnvelope<TType, TPayload>
}

function parseEnvelope(raw: Uint8Array | string) {
  try {
    const decoded = JSON.parse(
      typeof raw === "string" ? raw : decoder.decode(raw),
    ) as Record<string, unknown>
    if (
      !decoded ||
      typeof decoded !== "object" ||
      typeof decoded.seq !== "number" ||
      typeof decoded.ts_ms !== "number"
    ) {
      return null
    }
    const sessionId = decoded.session_id
    if (sessionId !== null && sessionId !== undefined && typeof sessionId !== "string") {
      return null
    }
    return decoded
  } catch {
    return null
  }
}
