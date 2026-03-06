import { useCallback, useEffect, useMemo, useRef, useState } from "react"
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

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { demoEnv } from "@/lib/env"
import type { SessionRecord } from "@/lib/coordinator"
import {
  decodeStatusMessage,
  encodePromptMessage,
  formatTopicPayload,
  type StatusEnvelope,
} from "@/lib/protocol"

export interface PromptCommand {
  nonce: number
  prompt: string
}

interface SessionRoomProps {
  session: SessionRecord
  token: string
  promptCommand: PromptCommand | null
  onPromptSent: (prompt: string) => void
  onRoomError: (message: string | null) => void
  onStatusEnvelope: (envelope: StatusEnvelope) => void
}

export function SessionRoom({
  session,
  token,
  promptCommand,
  onPromptSent,
  onRoomError,
  onStatusEnvelope,
}: SessionRoomProps) {
  if (!demoEnv.livekitUrl) {
    return (
      <Card className="border border-border/70 bg-card/80 backdrop-blur-sm">
        <CardHeader>
          <CardTitle>Live preview unavailable</CardTitle>
          <CardDescription>
            Set <code>VITE_LIVEKIT_URL</code> before starting the demo.
          </CardDescription>
        </CardHeader>
      </Card>
    )
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
      <SessionRoomContent
        session={session}
        promptCommand={promptCommand}
        onPromptSent={onPromptSent}
        onStatusEnvelope={onStatusEnvelope}
      />
    </LiveKitRoom>
  )
}

function SessionRoomContent({
  session,
  promptCommand,
  onPromptSent,
  onStatusEnvelope,
}: Omit<SessionRoomProps, "token" | "onRoomError">) {
  const room = useRoomContext()
  const connectionState = useConnectionState(room)
  const cameraTracks = useTracks([Track.Source.Camera]) as TrackReference[]
  const { isSending, send } = useDataChannel(demoEnv.controlTopic)
  const [latestStatusText, setLatestStatusText] = useState<string>("")
  const [promptSendError, setPromptSendError] = useState<string | null>(null)
  const promptSeqRef = useRef(0)
  const lastPromptNonceRef = useRef<number | null>(null)

  useDataChannel(demoEnv.statusTopic, (message) => {
    const decoded = decodeStatusMessage(message.payload)
    setLatestStatusText(formatTopicPayload(message.payload))
    if (!decoded) {
      return
    }
    if (decoded.session_id && decoded.session_id !== session.session_id) {
      return
    }
    onStatusEnvelope(decoded)
  })

  const remoteTrack = useMemo(() => {
    const preferredTrack = cameraTracks.find(
      (trackRef) =>
        trackRef.publication.trackName === demoEnv.videoTrackName &&
        trackRef.participant.identity !== room.localParticipant.identity,
    )

    if (preferredTrack) {
      return preferredTrack
    }

    return cameraTracks.find(
      (trackRef) => trackRef.participant.identity !== room.localParticipant.identity,
    )
  }, [cameraTracks, room.localParticipant.identity])

  const sendPrompt = useCallback(
    async (command: PromptCommand) => {
      const prompt = command.prompt.trim()
      if (!prompt) {
        return
      }

      try {
        setPromptSendError(null)
        await send(
          encodePromptMessage({
            prompt,
            seq: promptSeqRef.current++,
            sessionId: session.session_id,
          }),
          { reliable: true },
        )
        lastPromptNonceRef.current = command.nonce
        onPromptSent(prompt)
      } catch (error) {
        setPromptSendError(
          error instanceof Error ? error.message : "failed to publish prompt message",
        )
      }
    },
    [onPromptSent, send, session.session_id],
  )

  useEffect(() => {
    if (!promptCommand) {
      return
    }
    if (connectionState !== ConnectionState.Connected) {
      return
    }
    if (lastPromptNonceRef.current === promptCommand.nonce) {
      return
    }
    void sendPrompt(promptCommand)
  }, [connectionState, promptCommand, sendPrompt])

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
      <Card className="overflow-hidden border border-border/70 bg-card/85 shadow-[0_24px_90px_-48px_var(--color-foreground)] backdrop-blur-sm">
        <CardHeader className="border-b border-border/70">
          <CardTitle>Published LiveKit video</CardTitle>
          <CardDescription>
            Subscribed to the worker camera track and control topic <code>{demoEnv.controlTopic}</code>.
          </CardDescription>
          <CardAction>
            <Badge variant="outline">{String(connectionState)}</Badge>
          </CardAction>
        </CardHeader>
        <CardContent className="p-0">
          <div className="relative aspect-video overflow-hidden bg-[color:color-mix(in_oklab,var(--color-card)_84%,var(--color-background))]">
            {remoteTrack ? (
              <VideoTrack
                trackRef={remoteTrack}
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
                <Badge variant="secondary">
                  {connectionState === ConnectionState.Connected
                    ? "Waiting for worker video"
                    : "Connecting to room"}
                </Badge>
                <p className="max-w-md text-sm text-muted-foreground">
                  {connectionState === ConnectionState.Connected
                    ? `Connected to ${session.room_name}. Waiting for track ${demoEnv.videoTrackName}.`
                    : "The viewer will render the stream as soon as the client joins and the worker publishes frames."}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="border border-border/70 bg-card/85 backdrop-blur-sm">
        <CardHeader>
          <CardTitle>Control transport</CardTitle>
          <CardDescription>
            Prompt updates are JSON envelopes over LiveKit data packets.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div className="grid gap-2">
            <div className="flex items-center justify-between gap-3 rounded-lg border border-border/70 px-3 py-2">
              <span className="text-muted-foreground">Control topic</span>
              <code className="text-xs">{demoEnv.controlTopic}</code>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-lg border border-border/70 px-3 py-2">
              <span className="text-muted-foreground">Status topic</span>
              <code className="text-xs">{demoEnv.statusTopic}</code>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-lg border border-border/70 px-3 py-2">
              <span className="text-muted-foreground">Prompt publish</span>
              <Badge variant={isSending ? "default" : "outline"}>
                {isSending ? "Sending" : "Idle"}
              </Badge>
            </div>
          </div>

          <div className="space-y-2 rounded-xl border border-border/70 bg-muted/35 p-3">
            <p className="text-xs font-medium tracking-[0.18em] text-muted-foreground uppercase">
              Latest status payload
            </p>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words text-xs leading-5 text-foreground/80">
              {latestStatusText || "No status message received yet."}
            </pre>
          </div>

          {promptSendError ? (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {promptSendError}
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  )
}
