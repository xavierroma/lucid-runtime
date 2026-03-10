import { useEffect, useMemo, useState, type FormEvent } from "react"
import { LoaderCircle, Power, SendHorizonal } from "lucide-react"

import { ConsoleRoom, type QueuedPrompt } from "@/components/console-room"
import type {
  WaypointControlState,
  WaypointHoldControl,
} from "@/components/waypoint-controls"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  createSession,
  endSession,
  getSession,
  isTerminalSessionState,
  type Capabilities,
  type SessionResponse,
} from "@/lib/coordinator"
import { demoEnv, getMissingConfig } from "@/lib/env"

type DemoModelName = "yume" | "waypoint"

type DisplayTone = "off" | "warm" | "live" | "fault"

interface DisplayStatus {
  label: string
  detail: string
  tone: DisplayTone
}

const DEFAULT_WAYPOINT_CONTROLS: WaypointControlState = {
  forward: false,
  backward: false,
  left: false,
  right: false,
  jump: false,
  sprint: false,
  crouch: false,
  primary_fire: false,
  secondary_fire: false,
  mouse_x: 0,
  mouse_y: 0,
  scroll_wheel: 0,
}

const MODEL_OPTIONS: Array<{
  name: DemoModelName
  label: string
  fullLabel: string
}> = [
  {
    name: "yume",
    label: "Yume",
    fullLabel: "Yume-1.5",
  },
  {
    name: "waypoint",
    label: "Waypoint",
    fullLabel: "Waypoint-1.1-Small",
  },
]

function hasPromptAction(capabilities: Capabilities | null) {
  return Boolean(
    capabilities?.manifest.actions.find((action) => action.name === "set_prompt"),
  )
}

function hasControlAction(capabilities: Capabilities | null) {
  return Boolean(
    capabilities?.manifest.actions.find((action) => action.name === "set_controls"),
  )
}

function normalizeModelName(value: string | null | undefined): DemoModelName | null {
  if (value === "yume" || value === "waypoint") {
    return value
  }
  return null
}

function modelLabel(modelName: DemoModelName) {
  return MODEL_OPTIONS.find((option) => option.name === modelName)?.label ?? modelName
}

function buildDisplayStatus(args: {
  session: SessionResponse["session"] | null
  modelName: DemoModelName
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
    modelName,
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
      detail: `Requesting a fresh ${modelLabel(modelName)} session.`,
      tone: "warm",
    }
  }

  if (!session || session.state === "ENDED") {
    return {
      label: "OFF",
      detail: `Press power to wake ${modelLabel(modelName)}.`,
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
      detail: `Loading ${modelLabel(modelName)} and joining the room.`,
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
  const [selectedModel, setSelectedModel] = useState<DemoModelName>(demoEnv.defaultModel)
  const [waypointControls, setWaypointControls] =
    useState<WaypointControlState>(DEFAULT_WAYPOINT_CONTROLS)
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
  const resolvedModel =
    normalizeModelName(capabilities?.manifest.model.name) ?? selectedModel
  const promptSupported = hasPromptAction(capabilities)
  const controlsSupported = hasControlAction(capabilities)
  const waypointModelActive = resolvedModel === "waypoint"
  const promptText = prompt.trim()
  const canSendPrompt = Boolean(hasActiveSession && promptSupported && promptText)
  const status = buildDisplayStatus({
    session,
    modelName: resolvedModel,
    missingConfig,
    requestError,
    roomError,
    createPending,
    endPending,
    roomConnected,
    trackReady,
  })

  useEffect(() => {
    const actualModel = normalizeModelName(capabilities?.manifest.model.name)
    if (actualModel) {
      setSelectedModel(actualModel)
    }
  }, [capabilities?.manifest.model.name])

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
    }, 5000)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [session?.session_id, session?.state])

  useEffect(() => {
    if (hasActiveSession && waypointModelActive) {
      return
    }
    setWaypointControls(DEFAULT_WAYPOINT_CONTROLS)
  }, [hasActiveSession, waypointModelActive])

  useEffect(() => {
    if (!hasActiveSession || !controlsSupported || resolvedModel !== "waypoint") {
      return
    }

    const isEditableTarget = (target: EventTarget | null) => {
      if (!(target instanceof HTMLElement)) {
        return false
      }
      const tagName = target.tagName
      return tagName === "TEXTAREA" || tagName === "INPUT" || target.isContentEditable
    }

    const updateHold = (control: WaypointHoldControl, pressed: boolean) => {
      setWaypointControls((current) =>
        current[control] === pressed
          ? current
          : {
            ...current,
            [control]: pressed,
          },
      )
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) {
        return
      }

      switch (event.code) {
        case "KeyW":
          event.preventDefault()
          updateHold("forward", true)
          break
        case "KeyS":
          event.preventDefault()
          updateHold("backward", true)
          break
        case "KeyA":
          event.preventDefault()
          updateHold("left", true)
          break
        case "KeyD":
          event.preventDefault()
          updateHold("right", true)
          break
        case "Space":
          event.preventDefault()
          updateHold("jump", true)
          break
        case "ShiftLeft":
        case "ShiftRight":
          event.preventDefault()
          updateHold("sprint", true)
          break
        case "ControlLeft":
        case "ControlRight":
          event.preventDefault()
          updateHold("crouch", true)
          break
        case "KeyJ":
          event.preventDefault()
          updateHold("primary_fire", true)
          break
        case "KeyK":
          event.preventDefault()
          updateHold("secondary_fire", true)
          break
        default:
          break
      }
    }

    const handleKeyUp = (event: KeyboardEvent) => {
      switch (event.code) {
        case "KeyW":
          updateHold("forward", false)
          break
        case "KeyS":
          updateHold("backward", false)
          break
        case "KeyA":
          updateHold("left", false)
          break
        case "KeyD":
          updateHold("right", false)
          break
        case "Space":
          updateHold("jump", false)
          break
        case "ShiftLeft":
        case "ShiftRight":
          updateHold("sprint", false)
          break
        case "ControlLeft":
        case "ControlRight":
          updateHold("crouch", false)
          break
        case "KeyJ":
          updateHold("primary_fire", false)
          break
        case "KeyK":
          updateHold("secondary_fire", false)
          break
        default:
          break
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
    }
  }, [controlsSupported, hasActiveSession, resolvedModel])

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
      const response = await createSession(selectedModel)
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
      <section
        className="console-shell"
        aria-label={`Lucid ${modelLabel(resolvedModel)} console`}
      >
        <div className="console-body">
          <div className="console-toolbar">
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
          </div>

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
                controlState={waypointControls}
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
                onActionError={(message) => {
                  setPromptPending(false)
                  setRequestError(message)
                }}
                onRoomError={setRoomError}
              />
            </div>
          </div>

          <div className="console-inputs">
            <form className="prompt-panel" onSubmit={handlePromptSubmit}>
              <textarea
                className="prompt-textarea"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder={
                  waypointModelActive
                    ? "Guide the world, then drive it with the keyboard..."
                    : "Type a new prompt for the world..."
                }
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
            <div className="model-picker">
              <Select
                value={selectedModel}
                onValueChange={(value) => {
                  const nextModel = normalizeModelName(value)
                  if (nextModel) {
                    setSelectedModel(nextModel)
                  }
                }}
                disabled={hasActiveSession || createPending || endPending}
              >
                <SelectTrigger
                  aria-label="Model"
                  className="h-10 w-full max-w-[18rem] rounded-full px-4 text-sm shadow-none sm:w-fit sm:min-w-[16rem]"
                >
                  <SelectValue placeholder="Model" />
                </SelectTrigger>
                <SelectContent align="start" position="popper" sideOffset={8} className="shadow-none">
                  {MODEL_OPTIONS.map((option) => (
                    <SelectItem key={option.name} value={option.name} className="text-sm">
                      {option.fullLabel}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}

export default App
