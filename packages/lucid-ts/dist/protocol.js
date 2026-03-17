export const DEFAULT_CONTROL_TOPIC = "wm.control";
export const DEFAULT_STATUS_TOPIC = "wm.status";
export const DEFAULT_INPUT_FILE_TOPIC = "wm.input.file";
const encoder = new TextEncoder();
const decoder = new TextDecoder();
export function buildOutputTopic(outputName) {
    return `wm.output.${outputName}`;
}
export function createInputEnvelope(args) {
    return {
        type: "action",
        seq: args.seq,
        ts_ms: args.tsMs ?? Date.now(),
        session_id: args.sessionId,
        payload: {
            name: args.name,
            args: args.args,
        },
    };
}
export function encodeInputMessage(args) {
    return encoder.encode(JSON.stringify(createInputEnvelope(args)));
}
export function createControlEnvelope(args) {
    return {
        type: args.type,
        seq: args.seq,
        ts_ms: args.tsMs ?? Date.now(),
        session_id: args.sessionId,
        payload: {},
    };
}
export function encodeControlMessage(args) {
    return encoder.encode(JSON.stringify(createControlEnvelope(args)));
}
export function createPingEnvelope(args) {
    const payload = {};
    if (args.clientTsMs !== undefined) {
        payload.client_ts_ms = args.clientTsMs;
    }
    else {
        payload.client_ts_ms = Date.now();
    }
    return {
        type: "ping",
        seq: args.seq,
        ts_ms: args.tsMs ?? Date.now(),
        session_id: args.sessionId,
        payload,
    };
}
export function encodePingMessage(args) {
    return encoder.encode(JSON.stringify(createPingEnvelope(args)));
}
export function decodeControlMessage(raw) {
    const payload = parseEnvelope(raw);
    if (!payload || typeof payload.type !== "string") {
        return null;
    }
    if (!("payload" in payload) || typeof payload.payload !== "object" || payload.payload === null) {
        return null;
    }
    return payload;
}
export function decodeStatusMessage(raw) {
    const payload = parseEnvelope(raw);
    if (!payload || typeof payload.type !== "string") {
        return null;
    }
    if (!("payload" in payload) || typeof payload.payload !== "object" || payload.payload === null) {
        return null;
    }
    return payload;
}
function parseEnvelope(raw) {
    try {
        const decoded = JSON.parse(typeof raw === "string" ? raw : decoder.decode(raw));
        if (!decoded ||
            typeof decoded !== "object" ||
            typeof decoded.seq !== "number" ||
            typeof decoded.ts_ms !== "number") {
            return null;
        }
        const sessionId = decoded.session_id;
        if (sessionId !== null && sessionId !== undefined && typeof sessionId !== "string") {
            return null;
        }
        return decoded;
    }
    catch {
        return null;
    }
}
