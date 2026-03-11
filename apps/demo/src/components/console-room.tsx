import { useEffect, useMemo, useRef, type ReactNode } from "react"
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
import type { Capabilities, SessionRecord } from "@/lib/coordinator"

export interface QueuedLaunch {
  nonce: number
  prompt: string | null
}

export interface QueuedMouseMove {
  nonce: number
  dx: number
  dy: number
}

export interface QueuedScroll {
  nonce: number
  amount: number
}

interface ConsoleRoomProps {
  session: SessionRecord | null
  token: string | null
  capabilities: Capabilities | null
  queuedLaunch: QueuedLaunch | null
  pressedButtonIds: number[]
  queuedMouseMove: QueuedMouseMove | null
  queuedScroll: QueuedScroll | null
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onLaunchSent: () => void
  onActionError: (message: string | null) => void
  onRoomError: (message: string | null) => void
}

interface ActionEnvelope {
  type: "action"
  seq: number
  ts_ms: number
  session_id: string | null
  payload: {
    name: string
    args: Record<string, unknown>
  }
}

const encoder = new TextEncoder()

function encodeActionMessage(args: {
  name: string
  args: Record<string, unknown>
  seq: number
  sessionId: string
}) {
  const envelope: ActionEnvelope = {
    type: "action",
    seq: args.seq,
    ts_ms: Date.now(),
    session_id: args.sessionId,
    payload: {
      name: args.name,
      args: args.args,
    },
  }
  return encoder.encode(JSON.stringify(envelope))
}

export function ConsoleRoom({
  session,
  token,
  capabilities,
  queuedLaunch,
  pressedButtonIds,
  queuedMouseMove,
  queuedScroll,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onLaunchSent,
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
        queuedLaunch={queuedLaunch}
        pressedButtonIds={pressedButtonIds}
        queuedMouseMove={queuedMouseMove}
        queuedScroll={queuedScroll}
        fallback={fallback}
        onConnectionChange={onConnectionChange}
        onTrackReadyChange={onTrackReadyChange}
        onLaunchSent={onLaunchSent}
        onActionError={onActionError}
      />
    </LiveKitRoom>
  )
}

interface ConsoleRoomContentProps {
  session: SessionRecord
  capabilities: Capabilities
  queuedLaunch: QueuedLaunch | null
  pressedButtonIds: number[]
  queuedMouseMove: QueuedMouseMove | null
  queuedScroll: QueuedScroll | null
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onLaunchSent: () => void
  onActionError: (message: string | null) => void
}

function ConsoleRoomContent({
  session,
  capabilities,
  queuedLaunch,
  pressedButtonIds,
  queuedMouseMove,
  queuedScroll,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onLaunchSent,
  onActionError,
}: ConsoleRoomContentProps) {
  const room = useRoomContext()
  const connectionState = useConnectionState(room)
  const cameraTracks = useTracks([Track.Source.Camera]) as TrackReference[]
  const { send } = useDataChannel(capabilities.control_topic)
  const actionSeqRef = useRef(0)
  const lastLaunchNonceRef = useRef<number | null>(null)
  const lastButtonsSignatureRef = useRef<string | null>(null)
  const lastMouseMoveNonceRef = useRef<number | null>(null)
  const lastScrollNonceRef = useRef<number | null>(null)

  const videoBinding = useMemo(
    () => capabilities.output_bindings.find((binding) => binding.kind === "video"),
    [capabilities.output_bindings],
  )
  const supportsButtonState = useMemo(
    () => capabilities.manifest.actions.some((action) => action.name === "set_buttons"),
    [capabilities.manifest.actions],
  )
  const supportsMouseMove = useMemo(
    () => capabilities.manifest.actions.some((action) => action.name === "mouse_move"),
    [capabilities.manifest.actions],
  )
  const supportsScroll = useMemo(
    () => capabilities.manifest.actions.some((action) => action.name === "scroll"),
    [capabilities.manifest.actions],
  )
  const supportsPromptState = useMemo(
    () => capabilities.manifest.actions.some((action) => action.name === "set_prompt"),
    [capabilities.manifest.actions],
  )
  const sessionAcceptsControlMessages =
    session.state === "READY" || session.state === "RUNNING"

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

  useEffect(() => {
    onConnectionChange(connectionState === ConnectionState.Connected)
  }, [connectionState, onConnectionChange])

  useEffect(() => {
    onTrackReadyChange(Boolean(remoteTrack))
  }, [onTrackReadyChange, remoteTrack])

  useEffect(() => {
    actionSeqRef.current = 0
    lastLaunchNonceRef.current = null
    lastButtonsSignatureRef.current = null
    lastMouseMoveNonceRef.current = null
    lastScrollNonceRef.current = null
  }, [session.session_id])

  useEffect(() => {
    if (!queuedLaunch) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (session.state !== "READY") {
      return
    }
    if (lastLaunchNonceRef.current === queuedLaunch.nonce) {
      return
    }

    let cancelled = false

    const publishLaunch = async () => {
      try {
        onActionError(null)

        if (queuedLaunch.prompt && supportsPromptState) {
          await send(
            encodeActionMessage({
              name: "set_prompt",
              args: { prompt: queuedLaunch.prompt },
              seq: actionSeqRef.current++,
              sessionId: session.session_id,
            }),
            { reliable: true },
          )
        }

        await send(
          encodeActionMessage({
            name: "lucid.runtime.start",
            args: {},
            seq: actionSeqRef.current++,
            sessionId: session.session_id,
          }),
          { reliable: true },
        )
        if (!cancelled) {
          lastLaunchNonceRef.current = queuedLaunch.nonce
          onLaunchSent()
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish launch",
          )
        }
      }
    }

    void publishLaunch()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    onActionError,
    onLaunchSent,
    queuedLaunch,
    send,
    session.state,
    session.session_id,
    supportsPromptState,
  ])

  useEffect(() => {
    if (!supportsButtonState) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (!sessionAcceptsControlMessages) {
      return
    }

    const signature = JSON.stringify(pressedButtonIds)
    if (lastButtonsSignatureRef.current === signature) {
      return
    }

    let cancelled = false

    const publishButtons = async () => {
      try {
        onActionError(null)
        await send(
          encodeActionMessage({
            name: "set_buttons",
            args: {
              buttons: pressedButtonIds,
            },
            seq: actionSeqRef.current++,
            sessionId: session.session_id,
          }),
          { reliable: true },
        )
        if (!cancelled) {
          lastButtonsSignatureRef.current = signature
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish button state",
          )
        }
      }
    }

    void publishButtons()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    onActionError,
    pressedButtonIds,
    send,
    session.state,
    session.session_id,
    sessionAcceptsControlMessages,
    supportsButtonState,
  ])

  useEffect(() => {
    if (!queuedMouseMove) {
      return
    }
    if (!supportsMouseMove) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (!sessionAcceptsControlMessages) {
      return
    }
    if (lastMouseMoveNonceRef.current === queuedMouseMove.nonce) {
      return
    }

    let cancelled = false

    const publishMouseMove = async () => {
      try {
        onActionError(null)
        await send(
          encodeActionMessage({
            name: "mouse_move",
            args: {
              dx: queuedMouseMove.dx,
              dy: queuedMouseMove.dy,
            },
            seq: actionSeqRef.current++,
            sessionId: session.session_id,
          }),
          { reliable: false },
        )
        if (!cancelled) {
          lastMouseMoveNonceRef.current = queuedMouseMove.nonce
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish mouse movement",
          )
        }
      }
    }

    void publishMouseMove()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    onActionError,
    queuedMouseMove,
    send,
    session.state,
    session.session_id,
    sessionAcceptsControlMessages,
    supportsMouseMove,
  ])

  useEffect(() => {
    if (!queuedScroll) {
      return
    }
    if (!supportsScroll) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (!sessionAcceptsControlMessages) {
      return
    }
    if (lastScrollNonceRef.current === queuedScroll.nonce) {
      return
    }

    let cancelled = false

    const publishScroll = async () => {
      try {
        onActionError(null)
        await send(
          encodeActionMessage({
            name: "scroll",
            args: {
              amount: queuedScroll.amount,
            },
            seq: actionSeqRef.current++,
            sessionId: session.session_id,
          }),
          { reliable: false },
        )
        if (!cancelled) {
          lastScrollNonceRef.current = queuedScroll.nonce
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish scroll input",
          )
        }
      }
    }

    void publishScroll()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    onActionError,
    queuedScroll,
    send,
    session.state,
    session.session_id,
    sessionAcceptsControlMessages,
    supportsScroll,
  ])

  if (!remoteTrack) {
    return <>{fallback}</>
  }

  return (
    <VideoTrack
      trackRef={remoteTrack}
      className="console-video"
    />
  )
}
