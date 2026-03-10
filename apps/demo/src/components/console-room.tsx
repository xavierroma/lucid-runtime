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
import { encodeActionMessage } from "@/lib/generated/lucid"

export interface QueuedPrompt {
  nonce: number
  prompt: string
}

interface ConsoleRoomProps {
  session: SessionRecord | null
  token: string | null
  capabilities: Capabilities | null
  queuedPrompt: QueuedPrompt | null
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onPromptSent: () => void
  onPromptError: (message: string | null) => void
  onRoomError: (message: string | null) => void
}

export function ConsoleRoom({
  session,
  token,
  capabilities,
  queuedPrompt,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onPromptSent,
  onPromptError,
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
        fallback={fallback}
        onConnectionChange={onConnectionChange}
        onTrackReadyChange={onTrackReadyChange}
        onPromptSent={onPromptSent}
        onPromptError={onPromptError}
      />
    </LiveKitRoom>
  )
}

interface ConsoleRoomContentProps {
  session: SessionRecord
  capabilities: Capabilities
  queuedPrompt: QueuedPrompt | null
  fallback: ReactNode
  onConnectionChange: (connected: boolean) => void
  onTrackReadyChange: (ready: boolean) => void
  onPromptSent: () => void
  onPromptError: (message: string | null) => void
}

function ConsoleRoomContent({
  session,
  capabilities,
  queuedPrompt,
  fallback,
  onConnectionChange,
  onTrackReadyChange,
  onPromptSent,
  onPromptError,
}: ConsoleRoomContentProps) {
  const room = useRoomContext()
  const connectionState = useConnectionState(room)
  const cameraTracks = useTracks([Track.Source.Camera]) as TrackReference[]
  const { send } = useDataChannel(capabilities.control_topic)
  const actionSeqRef = useRef(0)
  const lastPromptNonceRef = useRef<number | null>(null)

  const videoBinding = useMemo(
    () => capabilities.output_bindings.find((binding) => binding.kind === "video"),
    [capabilities.output_bindings],
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
        onPromptError(null)
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
          onPromptError(
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
    onPromptError,
    onPromptSent,
    queuedPrompt,
    send,
    session.session_id,
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
