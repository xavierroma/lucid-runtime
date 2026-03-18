import {
  useEffect,
  useMemo,
  useRef,
  type ReactNode,
  type RefObject,
} from "react"
import {
  LiveKitRoom,
  RoomAudioRenderer,
  VideoTrack,
  useConnectionState,
  useDataChannel,
  useRoomContext,
  useTracks,
} from "@livekit/components-react"
import { ConnectionState, Track } from "livekit-client"
import type { TrackReference } from "@livekit/components-react"

import { demoEnv } from "@/lib/env"
import {
  createLiveKitTransport,
  findInputsByBinding,
  findInitialFrameInput,
  findPromptInput,
  LucidSessionClient,
  type AxisInputBinding,
  type Capabilities,
  type HoldInputBinding,
  type ManifestInput,
  type SessionRecord,
  type UploadFieldConfig,
} from "@/lib/lucid"
import { InputOverlay } from "@/components/input-overlay"

interface ConsoleRoomProps {
  session: SessionRecord | null
  token: string | null
  capabilities: Capabilities | null
  inputFile: File | null
  promptValue: string | null
  transportControlSignal: TransportControlSignal | null
  interactionTargetRef: RefObject<HTMLElement | null>
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onActionError: (message: string | null) => void
  onRoomError: (message: string | null) => void
}

export type TransportControlSignalType = "pause" | "resume"

export interface TransportControlSignal {
  id: number
  type: TransportControlSignalType
}

interface PointerAccumulator {
  dx: number
  dy: number
}

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false
  }
  return (
    target.tagName === "TEXTAREA" ||
    target.tagName === "INPUT" ||
    target.isContentEditable
  )
}


function isHoldActive(
  binding: HoldInputBinding,
  pressedKeys: Set<string>,
  pressedMouseButtons: Set<number>,
) {
  return (
    binding.keys.some((key) => pressedKeys.has(key)) ||
    binding.mouse_buttons.some((button) => pressedMouseButtons.has(button))
  )
}

function computeAxisValue(binding: AxisInputBinding, pressedKeys: Set<string>) {
  const positive = binding.positive_keys.some((key) => pressedKeys.has(key)) ? 1 : 0
  const negative = binding.negative_keys.some((key) => pressedKeys.has(key)) ? 1 : 0
  return positive - negative
}

function fileIdentity(file: File | null) {
  if (!file) {
    return null
  }
  return [file.name, file.size, file.lastModified, file.type].join(":")
}

function resolveUploadMimeType(file: File, upload: UploadFieldConfig) {
  if (file.type && upload.mime_types.includes(file.type)) {
    return file.type
  }
  return upload.mime_types[0] ?? file.type ?? "application/octet-stream"
}

async function prepareUploadFile(file: File, upload: UploadFieldConfig) {
  if (
    upload.kind !== "image" ||
    !upload.target_width ||
    !upload.target_height
  ) {
    return file
  }

  const bitmap = await createImageBitmap(file)
  try {
    const canvas = document.createElement("canvas")
    canvas.width = upload.target_width
    canvas.height = upload.target_height
    const context = canvas.getContext("2d")
    if (!context) {
      return file
    }
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    const mimeType = resolveUploadMimeType(file, upload)
    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(
        (value) => resolve(value),
        mimeType,
        mimeType === "image/jpeg" || mimeType === "image/webp" ? 0.92 : undefined,
      )
    })
    if (!blob) {
      return file
    }
    return new File([blob], file.name, {
      type: mimeType,
      lastModified: file.lastModified,
    })
  } finally {
    bitmap.close()
  }
}


export function ConsoleRoom({
  session,
  token,
  capabilities,
  inputFile,
  promptValue,
  transportControlSignal,
  interactionTargetRef,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onActionError,
  onRoomError,
}: ConsoleRoomProps) {
  useEffect(() => {
    return () => {
      onConnectionChange(false)
      onTrackReadyChange(false)
    }
  }, [onConnectionChange, onTrackReadyChange])

  if (
    !demoEnv.livekitUrl ||
    !session ||
    !token ||
    !capabilities ||
    session.state === "ENDED" ||
    session.state === "FAILED"
  ) {
    return <>{fallback}</>
  }

  return (
    <LiveKitRoom
      key={session.session_id}
      audio={false}
      connect
      serverUrl={demoEnv.livekitUrl}
      token={token}
      video={false}
      className="contents"
      onConnected={() => onRoomError(null)}
      onDisconnected={() => onRoomError(null)}
      onError={(error) => onRoomError(error.message)}
    >
      <RoomAudioRenderer />
      <ConsoleRoomContent
        session={session}
        capabilities={capabilities}
        inputFile={inputFile}
        promptValue={promptValue}
        transportControlSignal={transportControlSignal}
        interactionTargetRef={interactionTargetRef}
        fallback={fallback}
        onConnectionChange={onConnectionChange}
        onTrackReadyChange={onTrackReadyChange}
        onActionError={onActionError}
      />
    </LiveKitRoom>
  )
}

interface ConsoleRoomContentProps {
  session: SessionRecord
  capabilities: Capabilities
  inputFile: File | null
  promptValue: string | null
  transportControlSignal: TransportControlSignal | null
  interactionTargetRef: RefObject<HTMLElement | null>
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onActionError: (message: string | null) => void
}

function ConsoleRoomContent({
  session,
  capabilities,
  inputFile,
  promptValue,
  transportControlSignal,
  interactionTargetRef,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onActionError,
}: ConsoleRoomContentProps) {
  const room = useRoomContext()
  const connectionState = useConnectionState(room)
  const cameraTracks = useTracks([Track.Source.Camera]) as TrackReference[]
  const lucidSession = useMemo(
    () =>
      new LucidSessionClient({
        sessionId: session.session_id,
        capabilities: {
          control_topic: capabilities.control_topic,
          status_topic: capabilities.status_topic,
        },
        transport: createLiveKitTransport(room.localParticipant),
      }),
    [
      capabilities.control_topic,
      capabilities.status_topic,
      room.localParticipant,
      session.session_id,
    ],
  )
  useDataChannel(capabilities.status_topic, (message) => {
    lucidSession.handleStatusMessage(message.payload)
  })
  const lastPromptRef = useRef<string | null>(null)
  const lastUploadedFileKeyRef = useRef<string | null>(null)
  const resumeRequestedRef = useRef(false)
  const lastTransportControlSignalIdRef = useRef<number | null>(null)
  const pressedKeysRef = useRef(new Set<string>())
  const pressedMouseButtonsRef = useRef(new Set<number>())
  const holdStateRef = useRef(new Map<string, boolean>())
  const axisValueRef = useRef(new Map<string, number>())
  const pointerAccumulatorRef = useRef(new Map<string, PointerAccumulator>())
  const pointerFrameRef = useRef<number | null>(null)
  const wheelAccumulatorRef = useRef(new Map<string, number>())
  const wheelFrameRef = useRef<number | null>(null)

  const videoBinding = useMemo(
    () => capabilities.output_bindings.find((binding) => binding.kind === "video"),
    [capabilities.output_bindings],
  )
  const promptInput = useMemo(
    () => findPromptInput(capabilities.manifest.inputs),
    [capabilities.manifest.inputs],
  )
  const initialFrameInput = useMemo(
    () => findInitialFrameInput(capabilities.manifest.inputs),
    [capabilities.manifest.inputs],
  )
  const holdInputs = useMemo(
    () => findInputsByBinding(capabilities.manifest.inputs, "hold"),
    [capabilities.manifest.inputs],
  )
  const pressInputs = useMemo(
    () => findInputsByBinding(capabilities.manifest.inputs, "press"),
    [capabilities.manifest.inputs],
  )
  const axisInputs = useMemo(
    () => findInputsByBinding(capabilities.manifest.inputs, "axis"),
    [capabilities.manifest.inputs],
  )
  const pointerInputs = useMemo(
    () => findInputsByBinding(capabilities.manifest.inputs, "pointer"),
    [capabilities.manifest.inputs],
  )
  const wheelInputs = useMemo(
    () => findInputsByBinding(capabilities.manifest.inputs, "wheel"),
    [capabilities.manifest.inputs],
  )
  const sessionCanResume = session.state === "READY"
  const sessionAcceptsLivePromptUpdates =
    session.state === "RUNNING" || session.state === "PAUSED"
  const sessionAcceptsPersistentInteractionMessages =
    session.state === "RUNNING" || session.state === "PAUSED"
  const sessionAcceptsTransientInteractionMessages = session.state === "RUNNING"

  const remoteTrack = useMemo(() => {
    const preferredTrack = cameraTracks.find(
      (trackRef) =>
        trackRef.publication.trackName === videoBinding?.track_name &&
        trackRef.participant.identity !== room.localParticipant.identity,
    )

    if (preferredTrack) {
      return preferredTrack
    }

    return cameraTracks.find(
      (trackRef) => trackRef.participant.identity !== room.localParticipant.identity,
    )
  }, [cameraTracks, room.localParticipant.identity, videoBinding?.track_name])

  const sendInput = async (
    name: string,
    args: Record<string, unknown>,
    reliable: boolean,
  ) => {
    await lucidSession.sendInput(name, args, { reliable })
  }

  const sendControl = async (type: TransportControlSignalType) => {
    if (type === "pause") {
      await lucidSession.pause()
      return
    }
    await lucidSession.resume()
  }

  const publishSelectedInputFile = async (file: File) => {
    if (!initialFrameInput) {
      return
    }
    const prepared = await prepareUploadFile(file, initialFrameInput.upload)
    if (prepared.size > initialFrameInput.upload.max_bytes) {
      throw new Error(
        `file exceeds ${initialFrameInput.upload.max_bytes} bytes after preprocessing`,
      )
    }
    await lucidSession.sendManifestUpload(initialFrameInput, prepared, {
      name: prepared.name,
      mimeType: prepared.type || "application/octet-stream",
    })
    lastUploadedFileKeyRef.current = fileIdentity(file)
  }

  useEffect(() => {
    onConnectionChange(connectionState === ConnectionState.Connected)
  }, [connectionState, onConnectionChange])

  useEffect(() => {
    return () => {
      lucidSession.dispose(new Error("room closed"))
    }
  }, [lucidSession])

  useEffect(() => {
    onTrackReadyChange(Boolean(remoteTrack))
  }, [onTrackReadyChange, remoteTrack])

  useEffect(() => {
    lastPromptRef.current = null
    lastUploadedFileKeyRef.current = null
    resumeRequestedRef.current = false
    lastTransportControlSignalIdRef.current = null
    pressedKeysRef.current.clear()
    pressedMouseButtonsRef.current.clear()
    holdStateRef.current.clear()
    axisValueRef.current.clear()
    pointerAccumulatorRef.current.clear()
    wheelAccumulatorRef.current.clear()
    if (pointerFrameRef.current !== null) {
      cancelAnimationFrame(pointerFrameRef.current)
      pointerFrameRef.current = null
    }
    if (wheelFrameRef.current !== null) {
      cancelAnimationFrame(wheelFrameRef.current)
      wheelFrameRef.current = null
    }
  }, [session.session_id])

  useEffect(() => {
    const normalizedPrompt = promptValue?.trim() ?? ""
    const currentFileKey = fileIdentity(inputFile)
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (!sessionCanResume || resumeRequestedRef.current) {
      return
    }

    let cancelled = false

    const publishStartupSequence = async () => {
      try {
        onActionError(null)
        if (
          promptInput &&
          normalizedPrompt &&
          lastPromptRef.current !== normalizedPrompt
        ) {
          await sendInput(promptInput.name, { prompt: normalizedPrompt }, true)
          if (cancelled) {
            return
          }
          lastPromptRef.current = normalizedPrompt
        }
        if (
          inputFile &&
          initialFrameInput &&
          currentFileKey &&
          lastUploadedFileKeyRef.current !== currentFileKey
        ) {
          await publishSelectedInputFile(inputFile)
          if (cancelled) {
            return
          }
        }
        await sendControl("resume")
        if (!cancelled) {
          resumeRequestedRef.current = true
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error
              ? error.message
              : "failed to publish startup controls",
          )
        }
      }
    }

    void publishStartupSequence()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    onActionError,
    promptInput,
    promptValue,
    inputFile,
    initialFrameInput,
    sessionCanResume,
    session.session_id,
  ])

  useEffect(() => {
    if (!transportControlSignal) {
      return
    }
    if (lastTransportControlSignalIdRef.current === transportControlSignal.id) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (transportControlSignal.type === "pause" && session.state !== "RUNNING") {
      return
    }
    if (transportControlSignal.type === "resume" && session.state !== "PAUSED") {
      return
    }

    let cancelled = false

    const publishTransportControl = async () => {
      try {
        onActionError(null)
        await sendControl(transportControlSignal.type)
        if (!cancelled) {
          lastTransportControlSignalIdRef.current = transportControlSignal.id
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error
              ? error.message
              : `failed to publish ${transportControlSignal.type}`,
          )
        }
      }
    }

    void publishTransportControl()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    onActionError,
    session.state,
    transportControlSignal,
    session.session_id,
  ])

  useEffect(() => {
    const normalizedPrompt = promptValue?.trim() ?? ""
    if (!promptInput || !normalizedPrompt) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (!sessionAcceptsLivePromptUpdates) {
      return
    }
    if (lastPromptRef.current === normalizedPrompt) {
      return
    }

    let cancelled = false

    const publishPrompt = async () => {
      try {
        onActionError(null)
        await sendInput(promptInput.name, { prompt: normalizedPrompt }, true)
        if (!cancelled) {
          lastPromptRef.current = normalizedPrompt
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish prompt",
          )
        }
      }
    }

    void publishPrompt()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    onActionError,
    promptInput,
    promptValue,
    session.session_id,
    sessionAcceptsLivePromptUpdates,
  ])

  useEffect(() => {
    const currentFileKey = fileIdentity(inputFile)
    if (!inputFile || !initialFrameInput || !currentFileKey) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (!(session.state === "RUNNING" || session.state === "PAUSED")) {
      return
    }
    if (lastUploadedFileKeyRef.current === currentFileKey) {
      return
    }

    let cancelled = false

    const publishInputFile = async () => {
      try {
        onActionError(null)
        await publishSelectedInputFile(inputFile)
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error
              ? error.message
              : "failed to publish uploaded input file",
          )
        }
      }
    }

    void publishInputFile()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    inputFile,
    onActionError,
    session.session_id,
    session.state,
    initialFrameInput,
  ])

  useEffect(() => {
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (
      !sessionAcceptsPersistentInteractionMessages &&
      !sessionAcceptsTransientInteractionMessages
    ) {
      return
    }

    const target = interactionTargetRef.current
    const hasPersistentMouseBindings = holdInputs.some(
      (input) => input.binding.mouse_buttons.length > 0,
    )
    const hasTransientMouseBindings =
      pointerInputs.length > 0 ||
      wheelInputs.length > 0 ||
      pressInputs.some((input) => input.binding.mouse_buttons.length > 0)
    const hasMouseBindings =
      (sessionAcceptsPersistentInteractionMessages && hasPersistentMouseBindings) ||
      (sessionAcceptsTransientInteractionMessages && hasTransientMouseBindings)
    const hasPersistentKeyboardBindings =
      holdInputs.some((input) => input.binding.keys.length > 0) ||
      axisInputs.length > 0
    const hasTransientKeyboardBindings = pressInputs.some(
      (input) => input.binding.keys.length > 0,
    )
    const hasKeyboardBindings =
      (sessionAcceptsPersistentInteractionMessages && hasPersistentKeyboardBindings) ||
      (sessionAcceptsTransientInteractionMessages && hasTransientKeyboardBindings)

    const flushPointerInputs = () => {
      pointerFrameRef.current = null
      const pending = Array.from(pointerAccumulatorRef.current.entries())
      pointerAccumulatorRef.current.clear()
      if (!pending.length) {
        return
      }
      void (async () => {
        try {
          onActionError(null)
          await Promise.all(
            pending.map(([name, delta]) =>
              sendInput(name, { dx: delta.dx, dy: delta.dy }, false),
            ),
          )
        } catch (error) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish pointer input",
          )
        }
      })()
    }

    const queuePointerDelta = (name: string, dx: number, dy: number) => {
      const current = pointerAccumulatorRef.current.get(name) ?? { dx: 0, dy: 0 }
      current.dx += dx
      current.dy += dy
      pointerAccumulatorRef.current.set(name, current)
      if (pointerFrameRef.current === null) {
        pointerFrameRef.current = requestAnimationFrame(flushPointerInputs)
      }
    }

    const flushWheelInputs = () => {
      wheelFrameRef.current = null
      const pending = Array.from(wheelAccumulatorRef.current.entries())
      wheelAccumulatorRef.current.clear()
      if (!pending.length) {
        return
      }
      void (async () => {
        try {
          onActionError(null)
          await Promise.all(
            pending.map(([name, delta]) => sendInput(name, { delta }, false)),
          )
        } catch (error) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish wheel input",
          )
        }
      })()
    }

    const queueWheelDelta = (name: string, delta: number) => {
      wheelAccumulatorRef.current.set(
        name,
        (wheelAccumulatorRef.current.get(name) ?? 0) + delta,
      )
      if (wheelFrameRef.current === null) {
        wheelFrameRef.current = requestAnimationFrame(flushWheelInputs)
      }
    }

    const publishHoldState = (input: ManifestInput & { binding: HoldInputBinding }) => {
      const pressed = isHoldActive(
        input.binding,
        pressedKeysRef.current,
        pressedMouseButtonsRef.current,
      )
      if (holdStateRef.current.get(input.name) === pressed) {
        return
      }
      holdStateRef.current.set(input.name, pressed)
      void (async () => {
        try {
          onActionError(null)
          await sendInput(input.name, { pressed }, true)
        } catch (error) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish hold input",
          )
        }
      })()
    }

    const publishAxisState = (input: ManifestInput & { binding: AxisInputBinding }) => {
      const value = computeAxisValue(input.binding, pressedKeysRef.current)
      if (axisValueRef.current.get(input.name) === value) {
        return
      }
      axisValueRef.current.set(input.name, value)
      void (async () => {
        try {
          onActionError(null)
          await sendInput(input.name, { value }, true)
        } catch (error) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish axis input",
          )
        }
      })()
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (!hasKeyboardBindings || isEditableTarget(event.target)) {
        return
      }
      if (pressedKeysRef.current.has(event.code)) {
        return
      }

      let consumed = false
      pressedKeysRef.current.add(event.code)

      if (sessionAcceptsPersistentInteractionMessages) {
        for (const input of holdInputs) {
          if (input.binding.keys.includes(event.code)) {
            consumed = true
            publishHoldState(input)
          }
        }

        for (const input of axisInputs) {
          if (
            input.binding.positive_keys.includes(event.code) ||
            input.binding.negative_keys.includes(event.code)
          ) {
            consumed = true
            publishAxisState(input)
          }
        }
      }

      if (sessionAcceptsTransientInteractionMessages) {
        for (const input of pressInputs) {
          if (!input.binding.keys.includes(event.code)) {
            continue
          }
          consumed = true
          void (async () => {
            try {
              onActionError(null)
              await sendInput(input.name, {}, true)
            } catch (error) {
              onActionError(
                error instanceof Error ? error.message : "failed to publish press input",
              )
            }
          })()
        }
      }

      if (consumed) {
        event.preventDefault()
      }
    }

    const handleKeyUp = (event: KeyboardEvent) => {
      if (!hasKeyboardBindings) {
        return
      }
      if (!pressedKeysRef.current.delete(event.code)) {
        return
      }

      let consumed = false
      if (sessionAcceptsPersistentInteractionMessages) {
        for (const input of holdInputs) {
          if (input.binding.keys.includes(event.code)) {
            consumed = true
            publishHoldState(input)
          }
        }

        for (const input of axisInputs) {
          if (
            input.binding.positive_keys.includes(event.code) ||
            input.binding.negative_keys.includes(event.code)
          ) {
            consumed = true
            publishAxisState(input)
          }
        }
      }

      if (consumed) {
        event.preventDefault()
      }
    }

    const handleMouseDown = (event: MouseEvent) => {
      if (!hasMouseBindings || !target || !target.contains(event.target as Node | null)) {
        return
      }

      let consumed = false
      if (!pressedMouseButtonsRef.current.has(event.button)) {
        pressedMouseButtonsRef.current.add(event.button)
      }

      if (sessionAcceptsPersistentInteractionMessages) {
        for (const input of holdInputs) {
          if (input.binding.mouse_buttons.includes(event.button)) {
            consumed = true
            publishHoldState(input)
          }
        }
      }

      if (sessionAcceptsTransientInteractionMessages) {
        for (const input of pressInputs) {
          if (!input.binding.mouse_buttons.includes(event.button)) {
            continue
          }
          consumed = true
          void (async () => {
            try {
              onActionError(null)
              await sendInput(input.name, {}, true)
            } catch (error) {
              onActionError(
                error instanceof Error ? error.message : "failed to publish press input",
              )
            }
          })()
        }

        const pointerLockInput = pointerInputs.find((input) => input.binding.pointer_lock)
        if (pointerLockInput && document.pointerLockElement !== target) {
          consumed = true
          void target.requestPointerLock()
        }
      }

      if (consumed) {
        event.preventDefault()
      }
    }

    const handleMouseUp = (event: MouseEvent) => {
      if (!hasMouseBindings) {
        return
      }
      if (!pressedMouseButtonsRef.current.delete(event.button)) {
        return
      }
      if (sessionAcceptsPersistentInteractionMessages) {
        for (const input of holdInputs) {
          if (input.binding.mouse_buttons.includes(event.button)) {
            publishHoldState(input)
          }
        }
      }
    }

    const handleMouseMove = (event: MouseEvent) => {
      if (
        !sessionAcceptsTransientInteractionMessages ||
        !pointerInputs.length ||
        !target
      ) {
        return
      }
      for (const input of pointerInputs) {
        const isActive = input.binding.pointer_lock
          ? document.pointerLockElement === target
          : target.contains(event.target as Node | null)
        if (!isActive) {
          continue
        }
        if (event.movementX === 0 && event.movementY === 0) {
          continue
        }
        queuePointerDelta(input.name, event.movementX, event.movementY)
      }
    }

    const handleWheel = (event: WheelEvent) => {
      if (
        !sessionAcceptsTransientInteractionMessages ||
        !wheelInputs.length ||
        !target ||
        !target.contains(event.target as Node | null)
      ) {
        return
      }
      if (event.deltaY === 0) {
        return
      }
      event.preventDefault()
      for (const input of wheelInputs) {
        queueWheelDelta(
          input.name,
          event.deltaY < 0 ? input.binding.step : -input.binding.step,
        )
      }
    }

    const handleContextMenu = (event: MouseEvent) => {
      if (!target || !target.contains(event.target as Node | null)) {
        return
      }
      if (hasMouseBindings) {
        event.preventDefault()
      }
    }

    if (hasKeyboardBindings) {
      window.addEventListener("keydown", handleKeyDown)
      window.addEventListener("keyup", handleKeyUp)
    }
    if (hasMouseBindings) {
      document.addEventListener("mousedown", handleMouseDown)
      document.addEventListener("mouseup", handleMouseUp)
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("wheel", handleWheel, { passive: false })
      document.addEventListener("contextmenu", handleContextMenu)
    }

    return () => {
      if (pointerFrameRef.current !== null) {
        cancelAnimationFrame(pointerFrameRef.current)
        pointerFrameRef.current = null
      }
      if (wheelFrameRef.current !== null) {
        cancelAnimationFrame(wheelFrameRef.current)
        wheelFrameRef.current = null
      }
      pointerAccumulatorRef.current.clear()
      wheelAccumulatorRef.current.clear()

      if (hasKeyboardBindings) {
        window.removeEventListener("keydown", handleKeyDown)
        window.removeEventListener("keyup", handleKeyUp)
      }
      if (hasMouseBindings) {
        document.removeEventListener("mousedown", handleMouseDown)
        document.removeEventListener("mouseup", handleMouseUp)
        document.removeEventListener("mousemove", handleMouseMove)
        document.removeEventListener("wheel", handleWheel)
        document.removeEventListener("contextmenu", handleContextMenu)
      }
    }
  }, [
    axisInputs,
    connectionState,
    holdInputs,
    interactionTargetRef,
    onActionError,
    pointerInputs,
    pressInputs,
    session.session_id,
    sessionAcceptsPersistentInteractionMessages,
    sessionAcceptsTransientInteractionMessages,
    wheelInputs,
  ])

  const overlay = (
    <InputOverlay
      inputs={capabilities.manifest.inputs}
      sessionState={session.state}
    />
  )

  if (!remoteTrack) {
    return (
      <>
        {fallback}
        {overlay}
      </>
    )
  }

  return (
    <>
      <VideoTrack trackRef={remoteTrack} className="console-video" />
      {overlay}
    </>
  )
}
