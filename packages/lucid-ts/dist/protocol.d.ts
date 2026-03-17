export declare const DEFAULT_CONTROL_TOPIC = "wm.control";
export declare const DEFAULT_STATUS_TOPIC = "wm.status";
export declare const DEFAULT_INPUT_FILE_TOPIC = "wm.input.file";
export type ControlMessageType = "end" | "pause" | "resume";
export interface ActionEnvelope<TName extends string = string, TArgs extends Record<string, unknown> = Record<string, unknown>> {
    type: "action";
    seq: number;
    ts_ms: number;
    session_id: string | null;
    payload: {
        name: TName;
        args: TArgs;
    };
}
export interface ControlEnvelope {
    type: ControlMessageType;
    seq: number;
    ts_ms: number;
    session_id: string | null;
    payload: Record<string, never>;
}
export interface PingEnvelope {
    type: "ping";
    seq: number;
    ts_ms: number;
    session_id: string | null;
    payload: {
        client_ts_ms?: number;
    };
}
export type LucidControlEnvelope = ActionEnvelope | ControlEnvelope | PingEnvelope;
export interface StatusEnvelope<TType extends string = string, TPayload extends Record<string, unknown> = Record<string, unknown>> {
    type: TType;
    seq: number;
    ts_ms: number;
    session_id: string | null;
    payload: TPayload;
}
export declare function buildOutputTopic(outputName: string): string;
export declare function createInputEnvelope<TName extends string, TArgs extends Record<string, unknown>>(args: {
    name: TName;
    args: TArgs;
    seq: number;
    sessionId: string | null;
    tsMs?: number;
}): ActionEnvelope<TName, TArgs>;
export declare function encodeInputMessage<TName extends string, TArgs extends Record<string, unknown>>(args: {
    name: TName;
    args: TArgs;
    seq: number;
    sessionId: string | null;
    tsMs?: number;
}): Uint8Array<ArrayBuffer>;
export declare function createControlEnvelope(args: {
    type: ControlMessageType;
    seq: number;
    sessionId: string | null;
    tsMs?: number;
}): ControlEnvelope;
export declare function encodeControlMessage(args: {
    type: ControlMessageType;
    seq: number;
    sessionId: string | null;
    tsMs?: number;
}): Uint8Array<ArrayBuffer>;
export declare function createPingEnvelope(args: {
    seq: number;
    sessionId: string | null;
    clientTsMs?: number;
    tsMs?: number;
}): PingEnvelope;
export declare function encodePingMessage(args: {
    seq: number;
    sessionId: string | null;
    clientTsMs?: number;
    tsMs?: number;
}): Uint8Array<ArrayBuffer>;
export declare function decodeControlMessage(raw: Uint8Array | string): LucidControlEnvelope | null;
export declare function decodeStatusMessage<TType extends string = string, TPayload extends Record<string, unknown> = Record<string, unknown>>(raw: Uint8Array | string): StatusEnvelope<TType, TPayload> | null;
