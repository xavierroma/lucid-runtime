import { useEffect, useMemo, useState, type FormEvent } from "react"
import { LoaderCircle, Power, SendHorizonal } from "lucide-react"

import { ConsoleRoom, type QueuedPrompt } from "@/components/console-room"
import {
  createSession,
  endSession,
  getSession,
  isTerminalSessionState,
  type Capabilities,
  type SessionResponse,
} from "@/lib/coordinator"
import { getMissingConfig } from "@/lib/env"

type DisplayTone = "off" | "warm" | "live" | "fault"

interface DisplayStatus {
  label: string
  detail: string
  tone: DisplayTone
}

function hasPromptAction(capabilities: Capabilities | null) {
  return Boolean(
    capabilities?.manifest.actions.find((action) => action.name === "set_prompt"),
  )
}

function buildDisplayStatus(args: {
  session: SessionResponse["session"] | null
  missingConfig: string[]
  requestError: string | null
  roomError: string | null
  createPending: boolean
  endPending: boolean
  roomConnected: boolean
  trackReady: boolean
}): DisplayStatus {
  const {
    session,
    missingConfig,
    requestError,
    roomError,
    createPending,
    endPending,
    roomConnected,
    trackReady,
  } = args

  if (missingConfig.length) {
    return {
      label: "CONFIG",
      detail: missingConfig.join(" · "),
      tone: "fault",
    }
  }

  if (requestError || roomError) {
    return {
      label: "FAULT",
      detail: requestError ?? roomError ?? "Unexpected runtime error",
      tone: "fault",
    }
  }

  if (createPending && !session) {
    return {
      label: "BOOTING",
      detail: "Requesting a fresh world session.",
      tone: "warm",
    }
  }

  if (!session || session.state === "ENDED") {
    return {
      label: "OFF",
      detail: "Press power to wake the console.",
      tone: "off",
    }
  }

  if (session.state === "FAILED") {
    return {
      label: "FAULT",
      detail: session.error_code ?? "Session failed.",
      tone: "fault",
    }
  }

  if (endPending || session.state === "CANCELING") {
    return {
      label: "SHUTDOWN",
      detail: "Powering the session down cleanly.",
      tone: "warm",
    }
  }

  if (session.state === "STARTING") {
    return {
      label: "STARTING",
      detail: "Loading Yume and joining the room.",
      tone: "warm",
    }
  }

  if (trackReady) {
    return {
      label: "LIVE",
      detail: "Video stream locked to the main display.",
      tone: "live",
    }
  }

  if (roomConnected) {
    return {
      label: "SYNCING",
      detail: "Connected to LiveKit. Waiting for the first frame.",
      tone: "warm",
    }
  }

  return {
    label: "LINKING",
    detail: "Connecting the handheld to the session room.",
    tone: "warm",
  }
}

export function App() {
  const missingConfig = useMemo(() => getMissingConfig(), [])
  const [sessionResponse, setSessionResponse] = useState<SessionResponse | null>(null)
  const [sessionToken, setSessionToken] = useState<string | null>(null)
  const [prompt, setPrompt] = useState("")
  const [queuedPrompt, setQueuedPrompt] = useState<QueuedPrompt | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [roomError, setRoomError] = useState<string | null>(null)
  const [createPending, setCreatePending] = useState(false)
  const [endPending, setEndPending] = useState(false)
  const [promptPending, setPromptPending] = useState(false)
  const [roomConnected, setRoomConnected] = useState(false)
  const [trackReady, setTrackReady] = useState(false)

  const session = sessionResponse?.session ?? null
  const capabilities = sessionResponse?.capabilities ?? null
  const canCreateSession = !session || isTerminalSessionState(session.state)
  const hasActiveSession = Boolean(session && !isTerminalSessionState(session.state))
  const promptSupported = hasPromptAction(capabilities)
  const promptText = prompt.trim()
  const canSendPrompt = Boolean(hasActiveSession && promptSupported && promptText)
  const status = buildDisplayStatus({
    session,
    missingConfig,
    requestError,
    roomError,
    createPending,
    endPending,
    roomConnected,
    trackReady,
  })

  useEffect(() => {
    if (!session?.session_id || isTerminalSessionState(session.state)) {
      setRoomConnected(false)
      setTrackReady(false)
      if (isTerminalSessionState(session?.state ?? "ENDED")) {
        setPromptPending(false)
        setQueuedPrompt(null)
      }
      return
    }

    let cancelled = false

    const poll = async () => {
      try {
        const latest = await getSession(session.session_id)
        if (!cancelled) {
          setSessionResponse((current) => ({
            ...latest,
            client_access_token: current?.client_access_token ?? latest.client_access_token,
          }))
        }
      } catch (error) {
        if (!cancelled) {
          setRequestError(
            error instanceof Error ? error.message : "failed to poll session",
          )
        }
      }
    }

    void poll()
    const intervalId = window.setInterval(() => {
      void poll()
    }, 1000)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [session?.session_id, session?.state])

  const queuePrompt = (value: string) => {
    setPromptPending(true)
    setQueuedPrompt({
      nonce: Date.now(),
      prompt: value,
    })
  }

  const handleCreateSession = async () => {
    if (missingConfig.length) {
      setRequestError(`Missing configuration: ${missingConfig.join(", ")}`)
      return
    }

    setCreatePending(true)
    setRequestError(null)
    setRoomError(null)
    setPromptPending(false)
    setRoomConnected(false)
    setTrackReady(false)

    try {
      const response = await createSession()
      if (!response.client_access_token) {
        throw new Error("session server did not return a client access token")
      }

      setSessionResponse(response)
      setSessionToken(response.client_access_token)

      if (promptText && hasPromptAction(response.capabilities)) {
        queuePrompt(promptText)
      } else {
        setQueuedPrompt(null)
      }
    } catch (error) {
      setRequestError(
        error instanceof Error ? error.message : "failed to create session",
      )
    } finally {
      setCreatePending(false)
    }
  }

  const handleEndSession = async () => {
    if (!session) {
      return
    }

    setEndPending(true)
    setRequestError(null)

    try {
      await endSession(session.session_id)
    } catch (error) {
      setRequestError(
        error instanceof Error ? error.message : "failed to end session",
      )
    } finally {
      setEndPending(false)
    }
  }

  const handlePowerToggle = async () => {
    if (canCreateSession) {
      await handleCreateSession()
      return
    }
    await handleEndSession()
  }

  const handlePromptSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!canSendPrompt) {
      return
    }
    setRequestError(null)
    queuePrompt(promptText)
  }

  return (
    <main className="console-stage">
      <section className="console-shell" aria-label="Lucid Yume console">
        <div className="console-joycon console-joycon-left" aria-hidden="true" />
        <div className="console-joycon console-joycon-right" aria-hidden="true" />

        <div className="console-body">
          <div className="console-screen-frame">
            <div className={`screen-badge screen-badge-${status.tone}`}>
              <span className="screen-badge-dot" aria-hidden="true" />
              {status.label}
            </div>

            <div className="console-screen">
              <ConsoleRoom
                session={session}
                token={sessionToken}
                capabilities={capabilities}
                queuedPrompt={queuedPrompt}
                fallback={
                  <div className="screen-fallback">
                    <p className="screen-fallback-kicker">{status.label}</p>
                    <p className="screen-fallback-copy">{status.detail}</p>
                  </div>
                }
                onConnectionChange={setRoomConnected}
                onTrackReadyChange={setTrackReady}
                onPromptSent={() => {
                  setPromptPending(false)
                  setQueuedPrompt(null)
                  setRequestError(null)
                }}
                onPromptError={(message) => {
                  setPromptPending(false)
                  setRequestError(message)
                }}
                onRoomError={setRoomError}
              />
            </div>
          </div>

          <div className="console-controls">
            <button
              type="button"
              className={`power-button ${hasActiveSession ? "power-button-on" : ""}`}
              onClick={() => void handlePowerToggle()}
              disabled={createPending || endPending}
              aria-label={hasActiveSession ? "End session" : "Start session"}
            >
              {createPending || endPending ? (
                <LoaderCircle className="power-icon animate-spin" />
              ) : (
                <Power className="power-icon" />
              )}
            </button>

            <form className="prompt-panel" onSubmit={handlePromptSubmit}>
              <textarea
                className="prompt-textarea"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Type a new prompt for the world..."
                spellCheck={false}
              />
              <button
                type="submit"
                className="prompt-send"
                disabled={!canSendPrompt || promptPending}
              >
                {promptPending ? (
                  <LoaderCircle className="size-4 animate-spin" />
                ) : (
                  <SendHorizonal className="size-4" />
                )}
                Send
              </button>
            </form>
          </div>
        </div>
      </section>
    </main>
  )
}

export default App
