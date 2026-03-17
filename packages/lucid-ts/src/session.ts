import {
  DEFAULT_CONTROL_TOPIC,
  DEFAULT_INPUT_FILE_TOPIC,
  DEFAULT_STATUS_TOPIC,
  decodeStatusMessage,
  encodeControlMessage,
  encodeInputMessage,
  encodePingMessage,
  type StatusEnvelope,
} from "./protocol.js"
import type { Capabilities, UploadInputSpec } from "./types.js"

const DEFAULT_UPLOAD_TIMEOUT_MS = 15_000
const DEFAULT_UPLOAD_CHUNK_SIZE = 64 * 1024

export interface DataSendOptions {
  reliable?: boolean
  topic?: string
}

export interface BinaryUploadLike {
  size: number
  type?: string
  arrayBuffer(): Promise<ArrayBuffer>
}

export interface NamedBinaryUploadLike extends BinaryUploadLike {
  name?: string
}

export interface UploadRequest {
  name: string
  size: number
  mimeType: string
  topic?: string
  attributes?: Record<string, string>
}

export interface UploadWriter {
  uploadId: string
  write(chunk: Uint8Array): Promise<void>
  close(): Promise<void>
}

export interface LucidDataTransport {
  send(payload: Uint8Array, options?: DataSendOptions): Promise<void>
}

export interface LucidUploadTransport extends LucidDataTransport {
  openUpload(request: UploadRequest): Promise<UploadWriter>
}

export interface LiveKitByteStreamWriterLike {
  info?: {
    id?: string | null
    stream_id?: string | null
  }
  write(chunk: Uint8Array): Promise<void>
  close(): Promise<void>
}

export interface LocalParticipantLike {
  publishData(payload: Uint8Array, options?: DataSendOptions): Promise<void>
  streamBytes?(request: {
    name: string
    totalSize: number
    mimeType: string
    topic: string
    attributes?: Record<string, string>
  }): Promise<LiveKitByteStreamWriterLike>
}

export interface LucidSessionClientOptions {
  sessionId: string
  transport: LucidDataTransport | LucidUploadTransport
  capabilities?: Pick<Capabilities, "control_topic" | "status_topic"> | null
  controlTopic?: string
  statusTopic?: string
  inputFileTopic?: string
  chunkSize?: number
  now?: () => number
}

export interface SendInputOptions {
  reliable?: boolean
}

export interface UploadOptions {
  topic?: string
  timeoutMs?: number
  chunkSize?: number
  mimeType?: string
  attributes?: Record<string, string>
}

export interface UploadResult {
  uploadId: string
  sha256: string
}

export function createLiveKitTransport(
  participant: LocalParticipantLike,
): LucidUploadTransport {
  return {
    send(payload, options) {
      return participant.publishData(payload, options)
    },
    async openUpload(request) {
      if (!participant.streamBytes) {
        throw new Error("transport does not support byte-stream uploads")
      }
      const uploadRequest: {
        name: string
        totalSize: number
        mimeType: string
        topic: string
        attributes?: Record<string, string>
      } = {
        name: request.name,
        totalSize: request.size,
        mimeType: request.mimeType,
        topic: request.topic ?? DEFAULT_INPUT_FILE_TOPIC,
      }
      if (request.attributes) {
        uploadRequest.attributes = request.attributes
      }
      const writer = await participant.streamBytes(uploadRequest)
      const uploadId =
        writer.info?.id?.trim() || writer.info?.stream_id?.trim() || null
      if (!uploadId) {
        throw new Error("upload transport did not return an upload id")
      }
      return {
        uploadId,
        write(chunk) {
          return writer.write(chunk)
        },
        close() {
          return writer.close()
        },
      }
    },
  }
}

export class LucidSessionClient<
  TInputMap extends Record<string, Record<string, unknown>> = Record<
    string,
    Record<string, unknown>
  >,
> {
  readonly #sessionId: string
  readonly #transport: LucidDataTransport | LucidUploadTransport
  readonly #controlTopic: string
  readonly #statusTopic: string
  readonly #inputFileTopic: string
  readonly #chunkSize: number
  readonly #now: () => number
  #sequence = 0
  #completedUploads = new Set<string>()
  #failedUploads = new Map<string, string>()
  #pendingUploads = new Map<
    string,
    {
      resolve: () => void
      reject: (error: Error) => void
      timer: ReturnType<typeof setTimeout>
    }
  >()

  constructor(options: LucidSessionClientOptions) {
    this.#sessionId = options.sessionId
    this.#transport = options.transport
    this.#controlTopic =
      options.controlTopic ??
      options.capabilities?.control_topic ??
      DEFAULT_CONTROL_TOPIC
    this.#statusTopic =
      options.statusTopic ?? options.capabilities?.status_topic ?? DEFAULT_STATUS_TOPIC
    this.#inputFileTopic = options.inputFileTopic ?? DEFAULT_INPUT_FILE_TOPIC
    this.#chunkSize = Math.max(1, options.chunkSize ?? DEFAULT_UPLOAD_CHUNK_SIZE)
    this.#now = options.now ?? Date.now
  }

  get sessionId() {
    return this.#sessionId
  }

  get controlTopic() {
    return this.#controlTopic
  }

  get statusTopic() {
    return this.#statusTopic
  }

  nextSequence() {
    this.#sequence += 1
    return this.#sequence
  }

  sendInput<TName extends Extract<keyof TInputMap, string>>(
    name: TName,
    args: TInputMap[TName],
    options?: SendInputOptions,
  ) {
    return this.#transport.send(
      encodeInputMessage({
        name,
        args,
        seq: this.nextSequence(),
        sessionId: this.#sessionId,
        tsMs: this.#now(),
      }),
      {
        reliable: options?.reliable ?? true,
        topic: this.#controlTopic,
      },
    )
  }

  pause() {
    return this.#sendControl("pause")
  }

  resume() {
    return this.#sendControl("resume")
  }

  end() {
    return this.#sendControl("end")
  }

  ping(clientTsMs?: number) {
    const pingArgs: {
      seq: number
      sessionId: string
      tsMs: number
      clientTsMs?: number
    } = {
      seq: this.nextSequence(),
      sessionId: this.#sessionId,
      tsMs: this.#now(),
    }
    if (clientTsMs !== undefined) {
      pingArgs.clientTsMs = clientTsMs
    }
    return this.#transport.send(
      encodePingMessage(pingArgs),
      {
        reliable: true,
        topic: this.#controlTopic,
      },
    )
  }

  handleStatusMessage(raw: Uint8Array | string) {
    const status = decodeStatusMessage(raw)
    if (!status || status.session_id !== this.#sessionId) {
      return null
    }
    this.#settleUploadStatus(status)
    return status
  }

  async upload(
    file: NamedBinaryUploadLike,
    options?: UploadOptions & {
      inputName?: string
      argName?: string
      name?: string
    },
  ): Promise<UploadResult> {
    const transport = requireUploadTransport(this.#transport)
    const buffer = new Uint8Array(await file.arrayBuffer())
    const uploadIdAttributes: Record<string, string> = {
      session_id: this.#sessionId,
      ...options?.attributes,
    }
    if (options?.inputName) {
      uploadIdAttributes.input_name = options.inputName
    }
    if (options?.argName) {
      uploadIdAttributes.arg_name = options.argName
    }
    const sha256 = await sha256Hex(buffer)
    const uploadRequest: UploadRequest = {
      name: options?.name ?? file.name ?? "upload.bin",
      size: file.size,
      mimeType: options?.mimeType ?? file.type ?? "application/octet-stream",
      topic: options?.topic ?? this.#inputFileTopic,
      attributes: {
        ...uploadIdAttributes,
        sha256,
      },
    }
    const writer = await transport.openUpload(uploadRequest)
    const chunkSize = options?.chunkSize ?? this.#chunkSize
    for (let offset = 0; offset < buffer.byteLength; offset += chunkSize) {
      const end = Math.min(buffer.byteLength, offset + chunkSize)
      await writer.write(buffer.subarray(offset, end))
    }
    await writer.close()
    return { uploadId: writer.uploadId, sha256 }
  }

  waitForUploadReady(uploadId: string, timeoutMs = DEFAULT_UPLOAD_TIMEOUT_MS) {
    if (this.#completedUploads.delete(uploadId)) {
      this.#failedUploads.delete(uploadId)
      return Promise.resolve()
    }
    const failed = this.#failedUploads.get(uploadId)
    if (failed) {
      this.#failedUploads.delete(uploadId)
      return Promise.reject(new Error(failed))
    }
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#pendingUploads.delete(uploadId)
        reject(new Error("upload timed out"))
      }, timeoutMs)
      this.#pendingUploads.set(uploadId, { resolve, reject, timer })
    })
  }

  async sendUploadedInput<TName extends Extract<keyof TInputMap, string>>(
    args: {
      inputName: TName
      argName: string
      file: NamedBinaryUploadLike
      reliable?: boolean
      timeoutMs?: number
      mimeType?: string
      chunkSize?: number
      attributes?: Record<string, string>
      name?: string
    },
  ) {
    const uploadOptions: UploadOptions & {
      inputName?: string
      argName?: string
      name?: string
    } = {
      inputName: args.inputName,
      argName: args.argName,
    }
    if (args.timeoutMs !== undefined) {
      uploadOptions.timeoutMs = args.timeoutMs
    }
    if (args.mimeType !== undefined) {
      uploadOptions.mimeType = args.mimeType
    }
    if (args.chunkSize !== undefined) {
      uploadOptions.chunkSize = args.chunkSize
    }
    if (args.attributes !== undefined) {
      uploadOptions.attributes = args.attributes
    }
    if (args.name !== undefined) {
      uploadOptions.name = args.name
    }
    const upload = await this.upload(args.file, uploadOptions)
    await this.waitForUploadReady(upload.uploadId, args.timeoutMs)
    await this.sendInput(
      args.inputName,
      { [args.argName]: upload.uploadId } as TInputMap[TName],
      { reliable: args.reliable ?? true },
    )
    return upload
  }

  sendManifestUpload<TName extends Extract<keyof TInputMap, string>>(
    input: Pick<UploadInputSpec, "name" | "argName"> & { name: TName },
    file: NamedBinaryUploadLike,
    options?: {
      reliable?: boolean
      timeoutMs?: number
      mimeType?: string
      chunkSize?: number
      attributes?: Record<string, string>
      name?: string
    },
  ) {
    return this.sendUploadedInput({
      inputName: input.name,
      argName: input.argName,
      file,
      ...options,
    })
  }

  dispose(error: Error = new Error("session client disposed")) {
    for (const pending of this.#pendingUploads.values()) {
      clearTimeout(pending.timer)
      pending.reject(error)
    }
    this.#pendingUploads.clear()
    this.#completedUploads.clear()
    this.#failedUploads.clear()
  }

  #sendControl(type: "pause" | "resume" | "end") {
    return this.#transport.send(
      encodeControlMessage({
        type,
        seq: this.nextSequence(),
        sessionId: this.#sessionId,
        tsMs: this.#now(),
      }),
      {
        reliable: true,
        topic: this.#controlTopic,
      },
    )
  }

  #settleUploadStatus(status: StatusEnvelope) {
    const uploadId =
      typeof status.payload.upload_id === "string" ? status.payload.upload_id : null
    if (!uploadId) {
      return
    }
    const pending = this.#pendingUploads.get(uploadId)
    if (status.type === "upload_ready") {
      if (pending) {
        clearTimeout(pending.timer)
        this.#pendingUploads.delete(uploadId)
        pending.resolve()
        return
      }
      this.#completedUploads.add(uploadId)
      this.#failedUploads.delete(uploadId)
      return
    }
    if (status.type !== "upload_error") {
      return
    }
    const errorCode =
      typeof status.payload.error_code === "string"
        ? status.payload.error_code
        : "UPLOAD_FAILED"
    if (pending) {
      clearTimeout(pending.timer)
      this.#pendingUploads.delete(uploadId)
      pending.reject(new Error(errorCode))
      return
    }
    this.#failedUploads.set(uploadId, errorCode)
  }
}

function requireUploadTransport(
  transport: LucidDataTransport | LucidUploadTransport,
): LucidUploadTransport {
  if ("openUpload" in transport) {
    return transport
  }
  throw new Error("transport does not support byte-stream uploads")
}

async function sha256Hex(bytes: Uint8Array) {
  const subtle = globalThis.crypto?.subtle
  if (!subtle) {
    throw new Error("crypto.subtle is required for Lucid uploads")
  }
  const normalized = Uint8Array.from(bytes)
  const digest = new Uint8Array(
    await subtle.digest("SHA-256", normalized.buffer as ArrayBuffer),
  )
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("")
}
