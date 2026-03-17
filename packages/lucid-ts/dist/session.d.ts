import { type StatusEnvelope } from "./protocol.js";
import type { Capabilities, UploadInputSpec } from "./types.js";
export interface DataSendOptions {
    reliable?: boolean;
    topic?: string;
}
export interface BinaryUploadLike {
    size: number;
    type?: string;
    arrayBuffer(): Promise<ArrayBuffer>;
}
export interface NamedBinaryUploadLike extends BinaryUploadLike {
    name?: string;
}
export interface UploadRequest {
    name: string;
    size: number;
    mimeType: string;
    topic?: string;
    attributes?: Record<string, string>;
}
export interface UploadWriter {
    uploadId: string;
    write(chunk: Uint8Array): Promise<void>;
    close(): Promise<void>;
}
export interface LucidDataTransport {
    send(payload: Uint8Array, options?: DataSendOptions): Promise<void>;
}
export interface LucidUploadTransport extends LucidDataTransport {
    openUpload(request: UploadRequest): Promise<UploadWriter>;
}
export interface LiveKitByteStreamWriterLike {
    info?: {
        id?: string | null;
        stream_id?: string | null;
    };
    write(chunk: Uint8Array): Promise<void>;
    close(): Promise<void>;
}
export interface LocalParticipantLike {
    publishData(payload: Uint8Array, options?: DataSendOptions): Promise<void>;
    streamBytes?(request: {
        name: string;
        totalSize: number;
        mimeType: string;
        topic: string;
        attributes?: Record<string, string>;
    }): Promise<LiveKitByteStreamWriterLike>;
}
export interface LucidSessionClientOptions {
    sessionId: string;
    transport: LucidDataTransport | LucidUploadTransport;
    capabilities?: Pick<Capabilities, "control_topic" | "status_topic"> | null;
    controlTopic?: string;
    statusTopic?: string;
    inputFileTopic?: string;
    chunkSize?: number;
    now?: () => number;
}
export interface SendInputOptions {
    reliable?: boolean;
}
export interface UploadOptions {
    topic?: string;
    timeoutMs?: number;
    chunkSize?: number;
    mimeType?: string;
    attributes?: Record<string, string>;
}
export interface UploadResult {
    uploadId: string;
    sha256: string;
}
export declare function createLiveKitTransport(participant: LocalParticipantLike): LucidUploadTransport;
export declare class LucidSessionClient<TInputMap extends Record<string, Record<string, unknown>> = Record<string, Record<string, unknown>>> {
    #private;
    constructor(options: LucidSessionClientOptions);
    get sessionId(): string;
    get controlTopic(): string;
    get statusTopic(): string;
    nextSequence(): number;
    sendInput<TName extends Extract<keyof TInputMap, string>>(name: TName, args: TInputMap[TName], options?: SendInputOptions): Promise<void>;
    pause(): Promise<void>;
    resume(): Promise<void>;
    end(): Promise<void>;
    ping(clientTsMs?: number): Promise<void>;
    handleStatusMessage(raw: Uint8Array | string): StatusEnvelope<string, Record<string, unknown>> | null;
    upload(file: NamedBinaryUploadLike, options?: UploadOptions & {
        inputName?: string;
        argName?: string;
        name?: string;
    }): Promise<UploadResult>;
    waitForUploadReady(uploadId: string, timeoutMs?: number): Promise<void>;
    sendUploadedInput<TName extends Extract<keyof TInputMap, string>>(args: {
        inputName: TName;
        argName: string;
        file: NamedBinaryUploadLike;
        reliable?: boolean;
        timeoutMs?: number;
        mimeType?: string;
        chunkSize?: number;
        attributes?: Record<string, string>;
        name?: string;
    }): Promise<UploadResult>;
    sendManifestUpload<TName extends Extract<keyof TInputMap, string>>(input: Pick<UploadInputSpec, "name" | "argName"> & {
        name: TName;
    }, file: NamedBinaryUploadLike, options?: {
        reliable?: boolean;
        timeoutMs?: number;
        mimeType?: string;
        chunkSize?: number;
        attributes?: Record<string, string>;
        name?: string;
    }): Promise<UploadResult>;
    dispose(error?: Error): void;
}
