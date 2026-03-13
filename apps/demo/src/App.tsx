import { useEffect, useMemo, useRef, useState } from "react"
import { LoaderCircle, Pause, Play, Power } from "lucide-react"

import { ConsoleRoom, type TransportControlSignal } from "@/components/console-room"
import {
  EnvironmentStudio,
  type SaveEnvironmentInput,
} from "@/components/environment-studio"
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
import {
  createEnvironmentId,
  loadSavedEnvironments,
  loadSelectedEnvironmentId,
  persistSavedEnvironments,
  persistSelectedEnvironmentId,
  type SavedEnvironment,
} from "@/lib/environments"

type DemoModelName = "yume" | "waypoint"
type DisplayTone = "off" | "warm" | "live" | "fault"
type AppRoute = "/" | "/environments"

interface DisplayStatus {
  label: string
  detail: string
  tone: DisplayTone
}

const ENVIRONMENT_ROUTE: AppRoute = "/environments"

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

function hasPromptInput(capabilities: Capabilities | null) {
  return Boolean(
    capabilities?.manifest.inputs.find((input) => input.name === "set_prompt"),
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

function normalizeRoute(pathname: string): AppRoute {
  return pathname === ENVIRONMENT_ROUTE ? ENVIRONMENT_ROUTE : "/"
}

function sortEnvironments(environments: SavedEnvironment[]) {
  return [...environments].sort((left, right) =>
    right.updatedAt.localeCompare(left.updatedAt),
  )
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

  if (session.state === "READY") {
    return {
      label: "READY",
      detail: "Worker allocated. Waiting for prompt and resume.",
      tone: "warm",
    }
  }

  if (session.state === "PAUSED") {
    return {
      label: "PAUSED",
      detail: "Frame generation paused. Press resume to continue.",
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
  const [route, setRoute] = useState<AppRoute>(() =>
    normalizeRoute(window.location.pathname),
  )
  const [environments, setEnvironments] = useState<SavedEnvironment[]>(() =>
    loadSavedEnvironments(),
  )
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<string | null>(() =>
    loadSelectedEnvironmentId(),
  )
  const [sessionResponse, setSessionResponse] = useState<SessionResponse | null>(null)
  const [sessionToken, setSessionToken] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<DemoModelName>(demoEnv.defaultModel)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [roomError, setRoomError] = useState<string | null>(null)
  const [createPending, setCreatePending] = useState(false)
  const [endPending, setEndPending] = useState(false)
  const [roomConnected, setRoomConnected] = useState(false)
  const [trackReady, setTrackReady] = useState(false)
  const [transportControlSignal, setTransportControlSignal] =
    useState<TransportControlSignal | null>(null)
  const interactionTargetRef = useRef<HTMLDivElement | null>(null)
  const transportControlSeqRef = useRef(0)

  const session = sessionResponse?.session ?? null
  const capabilities = sessionResponse?.capabilities ?? null
  const selectedEnvironment = useMemo(
    () =>
      environments.find((environment) => environment.id === selectedEnvironmentId) ?? null,
    [environments, selectedEnvironmentId],
  )
  const selectedEnvironmentPrompt = selectedEnvironment?.prompt.trim() ?? ""
  const canCreateSession = !session || isTerminalSessionState(session.state)
  const hasActiveSession = Boolean(session && !isTerminalSessionState(session.state))
  const canToggleTransport =
    session?.state === "RUNNING" || session?.state === "PAUSED"
  const resolvedModel =
    normalizeModelName(capabilities?.manifest.model.name) ?? selectedModel
  const promptSupported = hasPromptInput(capabilities)
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
    const handlePopState = () => {
      setRoute(normalizeRoute(window.location.pathname))
    }

    window.addEventListener("popstate", handlePopState)
    return () => {
      window.removeEventListener("popstate", handlePopState)
    }
  }, [])

  useEffect(() => {
    persistSavedEnvironments(environments)
  }, [environments])

  useEffect(() => {
    persistSelectedEnvironmentId(selectedEnvironmentId)
  }, [selectedEnvironmentId])

  useEffect(() => {
    if (!environments.length) {
      if (selectedEnvironmentId !== null) {
        setSelectedEnvironmentId(null)
      }
      return
    }

    if (!selectedEnvironmentId) {
      setSelectedEnvironmentId(environments[0].id)
      return
    }

    if (!environments.some((environment) => environment.id === selectedEnvironmentId)) {
      setSelectedEnvironmentId(environments[0].id)
    }
  }, [environments, selectedEnvironmentId])

  useEffect(() => {
    const actualModel = normalizeModelName(capabilities?.manifest.model.name)
    if (actualModel) {
      setSelectedModel(actualModel)
    }
  }, [capabilities?.manifest.model.name])

  useEffect(() => {
    setTransportControlSignal(null)
  }, [session?.session_id])

  useEffect(() => {
    if (!session?.session_id || isTerminalSessionState(session.state)) {
      setRoomConnected(false)
      setTrackReady(false)
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
    }, 500)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [session?.session_id, session?.state])

  const navigateTo = (nextRoute: AppRoute) => {
    if (window.location.pathname !== nextRoute) {
      window.history.pushState({}, "", nextRoute)
    }
    setRoute(nextRoute)
  }

  const handleSaveEnvironment = (input: SaveEnvironmentInput) => {
    const timestamp = new Date().toISOString()
    const existing = input.environmentId
      ? environments.find((environment) => environment.id === input.environmentId) ?? null
      : null
    const nextEnvironment: SavedEnvironment = existing
      ? {
        ...existing,
        name: input.name,
        prompt: input.prompt,
        updatedAt: timestamp,
      }
      : {
        id: createEnvironmentId(),
        name: input.name,
        prompt: input.prompt,
        createdAt: timestamp,
        updatedAt: timestamp,
      }

    const nextEnvironments = existing
      ? sortEnvironments(
        environments.map((environment) =>
          environment.id === existing.id ? nextEnvironment : environment,
        ),
      )
      : sortEnvironments([nextEnvironment, ...environments])

    setEnvironments(nextEnvironments)
    setSelectedEnvironmentId(nextEnvironment.id)
    return nextEnvironment
  }

  const handleDeleteEnvironment = (environmentId: string) => {
    setEnvironments((current) =>
      current.filter((environment) => environment.id !== environmentId),
    )
  }

  const handleCreateSession = async () => {
    if (missingConfig.length) {
      setRequestError(`Missing configuration: ${missingConfig.join(", ")}`)
      return
    }

    setCreatePending(true)
    setRequestError(null)
    setRoomError(null)
    setRoomConnected(false)
    setTrackReady(false)

    try {
      const response = await createSession(selectedModel)
      if (!response.client_access_token) {
        throw new Error("session server did not return a client access token")
      }

      setSessionResponse(response)
      setSessionToken(response.client_access_token)
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

  const handleTransportToggle = () => {
    if (!session || !canToggleTransport) {
      return
    }
    transportControlSeqRef.current += 1
    setTransportControlSignal({
      id: transportControlSeqRef.current,
      type: session.state === "PAUSED" ? "resume" : "pause",
    })
  }

  if (route === ENVIRONMENT_ROUTE) {
    return (
      <EnvironmentStudio
        environments={environments}
        selectedEnvironmentId={selectedEnvironmentId}
        onSelectEnvironment={setSelectedEnvironmentId}
        onSaveEnvironment={handleSaveEnvironment}
        onDeleteEnvironment={handleDeleteEnvironment}
        onNavigateHome={() => navigateTo("/")}
      />
    )
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
              className="route-button"
              onClick={() => navigateTo(ENVIRONMENT_ROUTE)}
            >
              Environments
            </button>
            {canToggleTransport ? (
              <button
                type="button"
                className={`transport-button ${
                  session?.state === "PAUSED" ? "transport-button-resume" : ""
                }`}
                onClick={handleTransportToggle}
                disabled={createPending || endPending}
                aria-label={session?.state === "PAUSED" ? "Resume session" : "Pause session"}
              >
                {session?.state === "PAUSED" ? (
                  <Play className="transport-icon" />
                ) : (
                  <Pause className="transport-icon" />
                )}
                {session?.state === "PAUSED" ? "Resume" : "Pause"}
              </button>
            ) : null}
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

            <div ref={interactionTargetRef} className="console-screen">
              <ConsoleRoom
                session={session}
                token={sessionToken}
                capabilities={capabilities}
                promptValue={promptSupported ? selectedEnvironmentPrompt : null}
                transportControlSignal={transportControlSignal}
                interactionTargetRef={interactionTargetRef}
                fallback={
                  <div className="screen-fallback">
                    <p className="screen-fallback-kicker">{status.label}</p>
                    <p className="screen-fallback-copy">{status.detail}</p>
                  </div>
                }
                onConnectionChange={setRoomConnected}
                onTrackReadyChange={setTrackReady}
                onActionError={setRequestError}
                onRoomError={setRoomError}
              />
            </div>
          </div>

          <div className="console-inputs">
            <section className="environment-selector-panel">
              <div className="environment-selector-header">
                <div>
                  <p className="environment-selector-kicker">Startup environment</p>
                  <h2>
                    {selectedEnvironment?.name ?? "Choose a saved environment"}
                  </h2>
                </div>
                <button
                  type="button"
                  className="environment-manage-button"
                  onClick={() => navigateTo(ENVIRONMENT_ROUTE)}
                >
                  Manage Worlds
                </button>
              </div>

              {environments.length ? (
                <>
                  <div className="environment-select-row">
                    <Select
                      value={selectedEnvironmentId ?? undefined}
                      onValueChange={setSelectedEnvironmentId}
                    >
                      <SelectTrigger
                        aria-label="Environment"
                        className="h-10 w-full rounded-full px-4 text-sm shadow-none"
                      >
                        <SelectValue placeholder="Choose environment" />
                      </SelectTrigger>
                      <SelectContent
                        align="start"
                        position="popper"
                        sideOffset={8}
                        className="shadow-none"
                      >
                        {environments.map((environment) => (
                          <SelectItem
                            key={environment.id}
                            value={environment.id}
                            className="text-sm"
                          >
                            {environment.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="environment-preview">
                    <p className="environment-preview-kicker">Prompt synced to the session</p>
                    <p className="environment-preview-copy">
                      {selectedEnvironment?.prompt}
                    </p>
                  </div>
                </>
              ) : (
                <div className="environment-empty-console">
                  <p>No saved environments yet.</p>
                  <p>
                    Create one on the environments route, then come back here to
                    power on a session.
                  </p>
                </div>
              )}
            </section>

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
                <SelectContent
                  align="start"
                  position="popper"
                  sideOffset={8}
                  className="shadow-none"
                >
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
