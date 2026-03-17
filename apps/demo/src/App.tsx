import { useEffect, useMemo, useRef, useState } from "react"
import { LoaderCircle, Pause, Play, Power } from "lucide-react"

import { ConsoleRoom, type TransportControlSignal } from "@/components/console-room"
import {
  EnvironmentStudio,
  type SaveEnvironmentInput,
} from "@/components/environment-studio"
import { Card, CardHeader, CardTitle } from "@/components/ui/card"
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
  getModels,
  getSession,
  isTerminalSessionState,
  type Capabilities,
  type LucidManifest,
  type SupportedModel,
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
import { lucidManifest as heliosManifest } from "@/lib/generated/lucid.helios"
import { lucidManifest as defaultManifest } from "@/lib/generated/lucid"
import { lucidManifest as waypointManifest } from "@/lib/generated/lucid.waypoint"
import {
  findInitialFrameInput,
  findPromptInput,
  manifestForModel,
} from "@/lib/lucid"
import { dataUrlToFile } from "@/lib/input-files"

type DisplayTone = "off" | "warm" | "live" | "fault"
type AppRoute = "/" | "/environments"

interface DisplayStatus {
  label: string
  detail: string
  tone: DisplayTone
}

const ENVIRONMENT_ROUTE: AppRoute = "/environments"
const STATIC_MANIFESTS: Record<string, LucidManifest> = {
  helios: heliosManifest as unknown as LucidManifest,
  waypoint: waypointManifest as unknown as LucidManifest,
  yume: defaultManifest as unknown as LucidManifest,
}

function hasPromptInput(capabilities: Capabilities | null) {
  return Boolean(findPromptInput(capabilities?.manifest.inputs ?? []))
}

function normalizeModelId(value: string | null | undefined) {
  const normalized = value?.trim().toLowerCase()
  return normalized ? normalized : null
}

function selectPreferredModel(models: SupportedModel[], preferredModel: string, current: string | null) {
  if (current && models.some((model) => model.id === current)) {
    return current
  }

  const preferred = normalizeModelId(preferredModel)
  if (preferred && models.some((model) => model.id === preferred)) {
    return preferred
  }

  return models[0]?.id ?? null
}

function modelLabel(modelName: string | null, models: SupportedModel[]) {
  if (!modelName) {
    return "session"
  }

  return (
    models.find((model) => model.id === modelName)?.display_name ?? modelName
  )
}

function normalizeRoute(pathname: string): AppRoute {
  return pathname === ENVIRONMENT_ROUTE ? ENVIRONMENT_ROUTE : "/"
}

function sortEnvironments(environments: SavedEnvironment[]) {
  return [...environments].sort((left, right) =>
    right.updatedAt.localeCompare(left.updatedAt),
  )
}

function environmentCardStyle(seedImageDataUrl: string | null | undefined) {
  if (!seedImageDataUrl) {
    return undefined
  }

  return {
    backgroundImage: `url(${seedImageDataUrl})`,
  }
}

function buildDisplayStatus(args: {
  session: SessionResponse["session"] | null
  hasEnvironment: boolean
  modelName: string | null
  supportedModels: SupportedModel[]
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
    hasEnvironment,
    modelName,
    supportedModels,
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
      detail: `Requesting a fresh ${modelLabel(modelName, supportedModels)} session.`,
      tone: "warm",
    }
  }

  if (!session || session.state === "ENDED") {
    return {
      label: "OFF",
      detail: !hasEnvironment
        ? "Choose or create an environment."
        : modelName
        ? `Press power to wake ${modelLabel(modelName, supportedModels)}.`
        : "No supported models available from the coordinator.",
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
      detail: `Loading ${modelLabel(modelName, supportedModels)} and joining the room.`,
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
  const [supportedModels, setSupportedModels] = useState<SupportedModel[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
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
  const canCreateSession =
    Boolean(selectedModel && selectedEnvironment) &&
    (!session || isTerminalSessionState(session.state))
  const hasActiveSession = Boolean(session && !isTerminalSessionState(session.state))
  const canToggleTransport =
    session?.state === "RUNNING" || session?.state === "PAUSED"
  const resolvedModel =
    normalizeModelId(session?.model_name) ??
    normalizeModelId(capabilities?.manifest.model.name) ??
    selectedModel
  const promptSupported = hasPromptInput(capabilities)
  const activeManifest = useMemo(
    () => manifestForModel(selectedModel, capabilities?.manifest ?? null, STATIC_MANIFESTS),
    [capabilities?.manifest, selectedModel],
  )
  const initialFrameInput = useMemo(
    () => findInitialFrameInput(activeManifest?.inputs ?? []),
    [activeManifest],
  )
  const activeInputFile = useMemo(() => {
    if (!initialFrameInput || !selectedEnvironment?.seedImageDataUrl) {
      return null
    }

    const lastModified = Date.parse(selectedEnvironment.updatedAt)
    return dataUrlToFile(
      selectedEnvironment.seedImageDataUrl,
      `${selectedEnvironment.name}-seed`,
      Number.isFinite(lastModified) ? lastModified : 0,
    )
  }, [
    initialFrameInput,
    selectedEnvironment?.name,
    selectedEnvironment?.seedImageDataUrl,
    selectedEnvironment?.updatedAt,
  ])
  const status = buildDisplayStatus({
    session,
    hasEnvironment: Boolean(selectedEnvironment),
    modelName: resolvedModel,
    supportedModels,
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
    if (!demoEnv.coordinatorBaseUrl) {
      return
    }

    let cancelled = false

    const loadModels = async () => {
      try {
        const response = await getModels()
        if (cancelled) {
          return
        }

        setSupportedModels(response.models)
        setSelectedModel((current) =>
          selectPreferredModel(response.models, demoEnv.defaultModel, current),
        )
      } catch (error) {
        if (!cancelled) {
          setRequestError(
            error instanceof Error ? error.message : "failed to load models",
          )
        }
      }
    }

    void loadModels()
    return () => {
      cancelled = true
    }
  }, [])

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
    setSelectedModel((current) =>
      selectPreferredModel(supportedModels, demoEnv.defaultModel, current),
    )
  }, [supportedModels])

  useEffect(() => {
    const actualModel =
      normalizeModelId(session?.model_name) ??
      normalizeModelId(capabilities?.manifest.model.name)
    if (
      actualModel &&
      supportedModels.some((model) => model.id === actualModel)
    ) {
      setSelectedModel(actualModel)
    }
  }, [capabilities?.manifest.model.name, session?.model_name, supportedModels])

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
        seedImageDataUrl: input.seedImageDataUrl,
        updatedAt: timestamp,
      }
      : {
        id: createEnvironmentId(),
        name: input.name,
        prompt: input.prompt,
        seedImageDataUrl: input.seedImageDataUrl,
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
    if (!selectedModel) {
      setRequestError("No supported models available from the coordinator")
      return
    }
    if (!selectedEnvironment) {
      setRequestError("Choose an environment before starting a session")
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
        aria-label={`Lucid ${modelLabel(resolvedModel, supportedModels)} console`}
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
              disabled={createPending || endPending || (!hasActiveSession && !selectedModel)}
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
                inputFile={activeInputFile}
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
            {environments.length ? (
              <section className="environment-card-grid" aria-label="Environments">
                {environments.map((environment) => {
                  const isSelected = environment.id === selectedEnvironmentId
                  return (
                    <button
                      key={environment.id}
                      type="button"
                      className={`environment-card-button ${
                        isSelected ? "environment-card-button-selected" : ""
                      }`}
                      onClick={() => setSelectedEnvironmentId(environment.id)}
                    >
                      <Card className="environment-card-surface" size="sm">
                        <div
                          className="environment-card-media"
                          style={environmentCardStyle(environment.seedImageDataUrl)}
                        />
                        <div className="environment-card-scrim" />
                        <CardHeader className="environment-card-content">
                          <CardTitle className="environment-card-title">
                            {environment.name}
                          </CardTitle>
                        </CardHeader>
                      </Card>
                    </button>
                  )
                })}
              </section>
            ) : (
              <div className="environment-empty-console">
                <p>No environments</p>
              </div>
            )}

            <div className="model-picker">
              <Select
                value={selectedModel ?? undefined}
                onValueChange={setSelectedModel}
                disabled={
                  hasActiveSession ||
                  createPending ||
                  endPending ||
                  supportedModels.length === 0
                }
              >
                <SelectTrigger
                  aria-label="Model"
                  className="h-10 w-full max-w-[18rem] rounded-full px-4 text-sm shadow-none sm:w-fit sm:min-w-[16rem]"
                >
                  <SelectValue placeholder="No models available" />
                </SelectTrigger>
                <SelectContent
                  align="start"
                  position="popper"
                  sideOffset={8}
                  className="shadow-none"
                >
                  {supportedModels.map((model) => (
                    <SelectItem key={model.id} value={model.id} className="text-sm">
                      {model.display_name}
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
