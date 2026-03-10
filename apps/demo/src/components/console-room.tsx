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
import type { WaypointControlState } from "@/components/waypoint-controls"

export interface QueuedPrompt {
  nonce: number
  prompt: string
}

interface ConsoleRoomProps {
  session: SessionRecord | null
  token: string | null
  capabilities: Capabilities | null
  queuedPrompt: QueuedPrompt | null
  controlState: WaypointControlState
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onPromptSent: () => void
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
  queuedPrompt,
  controlState,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onPromptSent,
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
        queuedPrompt={queuedPrompt}
        controlState={controlState}
        fallback={fallback}
        onConnectionChange={onConnectionChange}
        onTrackReadyChange={onTrackReadyChange}
        onPromptSent={onPromptSent}
        onActionError={onActionError}
      />
    </LiveKitRoom>
  )
}

interface ConsoleRoomContentProps {
  session: SessionRecord
  capabilities: Capabilities
  queuedPrompt: QueuedPrompt | null
  controlState: WaypointControlState
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onPromptSent: () => void
  onActionError: (message: string | null) => void
}

function ConsoleRoomContent({
  session,
  capabilities,
  queuedPrompt,
  controlState,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onPromptSent,
  onActionError,
}: ConsoleRoomContentProps) {
  const room = useRoomContext()
  const connectionState = useConnectionState(room)
  const cameraTracks = useTracks([Track.Source.Camera]) as TrackReference[]
  const { send } = useDataChannel(capabilities.control_topic)
  const actionSeqRef = useRef(0)
  const lastPromptNonceRef = useRef<number | null>(null)
  const lastControlSignatureRef = useRef<string | null>(null)

  const videoBinding = useMemo(
    () => capabilities.output_bindings.find((binding) => binding.kind === "video"),
    [capabilities.output_bindings],
  )
  const supportsControlState = useMemo(
    () => capabilities.manifest.actions.some((action) => action.name === "set_controls"),
    [capabilities.manifest.actions],
  )

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
    lastPromptNonceRef.current = null
    lastControlSignatureRef.current = null
  }, [session.session_id])

  useEffect(() => {
    if (!queuedPrompt) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (lastPromptNonceRef.current === queuedPrompt.nonce) {
      return
    }

    let cancelled = false

    const publishPrompt = async () => {
      try {
        onActionError(null)
        await send(
          encodeActionMessage({
            name: "set_prompt",
            args: { prompt: queuedPrompt.prompt },
            seq: actionSeqRef.current++,
            sessionId: session.session_id,
          }),
          { reliable: true },
        )
        if (!cancelled) {
          lastPromptNonceRef.current = queuedPrompt.nonce
          onPromptSent()
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
    onPromptSent,
    queuedPrompt,
    send,
    session.session_id,
  ])

  useEffect(() => {
    if (!supportsControlState) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }

    const signature = JSON.stringify(controlState)
    if (lastControlSignatureRef.current === signature) {
      return
    }

    let cancelled = false

    const publishControls = async () => {
      try {
        onActionError(null)
        await send(
          encodeActionMessage({
            name: "set_controls",
            args: {
              forward: controlState.forward,
              backward: controlState.backward,
              left: controlState.left,
              right: controlState.right,
              jump: controlState.jump,
              sprint: controlState.sprint,
              crouch: controlState.crouch,
              primary_fire: controlState.primary_fire,
              secondary_fire: controlState.secondary_fire,
              mouse_x: controlState.mouse_x,
              mouse_y: controlState.mouse_y,
              scroll_wheel: controlState.scroll_wheel,
            },
            seq: actionSeqRef.current++,
            sessionId: session.session_id,
          }),
          { reliable: true },
        )
        if (!cancelled) {
          lastControlSignatureRef.current = signature
        }
      } catch (error) {
        if (!cancelled) {
          onActionError(
            error instanceof Error ? error.message : "failed to publish controls",
          )
        }
      }
    }

    void publishControls()

    return () => {
      cancelled = true
    }
  }, [
    connectionState,
    controlState,
    onActionError,
    send,
    session.session_id,
    supportsControlState,
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
