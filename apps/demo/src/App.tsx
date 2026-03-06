import { useEffect, useMemo, useState } from "react"
import {
  Loader,
  Play,
  Radio,
  SendHorizonal,
  Sparkles,
  Square,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Field, FieldContent, FieldLabel } from "@/components/ui/field"
import { Textarea } from "@/components/ui/textarea"
import {
  createSession,
  endSession,
  getSession,
  type SessionRecord,
} from "@/lib/coordinator"
import { demoEnv, getMissingConfig } from "@/lib/env"
import type { StatusEnvelope } from "@/lib/protocol"
import { SessionRoom, type PromptCommand } from "@/components/session-room"

const promptPresets = [
  "A first-person walk through a lantern market during light rain, cinematic reflections, grounded motion, realistic storefront depth.",
  "A first-person hike along a volcanic ridge at sunrise, drifting ash, sweeping horizon, stable camera path, natural parallax.",
  "A first-person glide through a brutalist museum atrium with skylight shafts, quiet crowds, polished stone, measured turns.",
] as const

function formatTimestamp(value: number | null | undefined) {
  if (!value) {
    return "-"
  }

  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(value)
}

function summarizePayload(payload: Record<string, unknown>) {
  const entries = Object.entries(payload)
  if (!entries.length) {
    return "No payload"
  }

  return entries
    .slice(0, 4)
    .map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : String(value)}`)
    .join(" · ")
}

export function App() {
  const missingConfig = useMemo(() => getMissingConfig(), [])
  const [promptDraft, setPromptDraft] = useState<string>(promptPresets[0])
  const [session, setSession] = useState<SessionRecord | null>(null)
  const [sessionToken, setSessionToken] = useState<string | null>(null)
  const [promptCommand, setPromptCommand] = useState<PromptCommand | null>(null)
  const [workerEvents, setWorkerEvents] = useState<StatusEnvelope[]>([])
  const [requestError, setRequestError] = useState<string | null>(null)
  const [roomError, setRoomError] = useState<string | null>(null)
  const [createPending, setCreatePending] = useState(false)
  const [endPending, setEndPending] = useState(false)
  const [lastPromptSentAt, setLastPromptSentAt] = useState<number | null>(null)

  const canCreateSession = !session || session.state === "ENDED"
  const hasActiveToken = Boolean(session && sessionToken)

  useEffect(() => {
    if (!session?.session_id || session.state === "ENDED") {
      return
    }

    let cancelled = false

    const poll = async () => {
      try {
        const latest = await getSession(session.session_id)
        if (!cancelled) {
          setSession(latest)
        }
      } catch (error) {
        if (!cancelled) {
          setRequestError(
            error instanceof Error ? error.message : "failed to poll coordinator session",
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

  const latestWorkerEvent = workerEvents[0] ?? null

  const handleCreateSession = async () => {
    if (missingConfig.length) {
      setRequestError(`Missing demo configuration: ${missingConfig.join(", ")}`)
      return
    }

    setCreatePending(true)
    setRequestError(null)
    setRoomError(null)

    try {
      const response = await createSession()
      if (!response.client_access_token) {
        throw new Error("coordinator did not return a client_access_token")
      }

      setSession(response.session)
      setSessionToken(response.client_access_token)
      setPromptCommand({ nonce: Date.now(), prompt: promptDraft })
      setWorkerEvents([])
      setLastPromptSentAt(null)
    } catch (error) {
      setRequestError(
        error instanceof Error ? error.message : "failed to create session",
      )
    } finally {
      setCreatePending(false)
    }
  }

  const handlePublishPrompt = () => {
    if (!session) {
      return
    }
    setPromptCommand({ nonce: Date.now(), prompt: promptDraft })
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

  const handleWorkerEvent = (event: StatusEnvelope) => {
    setWorkerEvents((current) => [event, ...current].slice(0, 10))
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-6 lg:px-8">
      <div className="space-y-6">
        <Card className="border border-border/70 bg-background/55 shadow-none">
          <CardHeader>
            <CardTitle>
              Session composer
              <Badge className="ml-2" variant={canCreateSession ? "outline" : "secondary"}>
                {canCreateSession ? "Ready" : "Active session"}
              </Badge>

            </CardTitle>
            <CardAction>
              <div className="flex flex-row gap-2 justify-end">
                <Button
                  type="button"
                  disabled={createPending || !canCreateSession}
                  onClick={() => void handleCreateSession()}
                >
                  <span className="inline-flex items-center gap-2">
                    {createPending ? (
                      <Loader className="size-4 animate-spin" />
                    ) : (
                      <Play className="size-4" />
                    )}
                    Start
                  </span>
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  disabled={!session || session.state === "ENDED" || endPending}
                  onClick={() => void handleEndSession()}
                >
                  {endPending ? (
                    <Loader className="size-4 animate-spin" />
                  ) : (
                    <Square className="size-4" />
                  )}
                  End
                </Button>
              </div>
            </CardAction>
          </CardHeader>

          <CardContent className="space-y-5">

            {session && sessionToken && (
              <SessionRoom
                session={session}
                token={sessionToken}
                promptCommand={promptCommand}
                onPromptSent={(prompt) => {
                  setLastPromptSentAt(Date.now())
                  setRequestError(null)
                  setPromptDraft(prompt)
                }}
                onRoomError={setRoomError}
                onStatusEnvelope={handleWorkerEvent}
              />
            )}

            <Field>
              <FieldLabel htmlFor="prompt">Prompt</FieldLabel>
              <FieldContent>
                <Textarea
                  id="prompt"
                  value={promptDraft}
                  onChange={(event) => setPromptDraft(event.target.value)}
                  className="min-h-36 resize-y"
                  placeholder="Describe the world you want the worker to render"
                />
              </FieldContent>
            </Field>


            <div className="flex flex-wrap gap-2">
              {promptPresets.map((preset) => (
                <Button
                  key={preset}
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-auto max-w-full whitespace-normal text-left"
                  onClick={() => setPromptDraft(preset)}
                >
                  <Sparkles className="mt-0.5 size-3.5 shrink-0" />
                  <span className="line-clamp-2">{preset}</span>
                </Button>
              ))}
            </div>

            <div className="flex flex-row gap-2 justify-end">

              <Button
                type="button"
                variant="secondary"
                disabled={!hasActiveToken || !promptDraft.trim()}
                onClick={handlePublishPrompt}
              >
                <SendHorizonal className="size-4" />
                Send prompt
              </Button>

            </div>
          </CardContent>
        </Card>


      </div>

      <div className="space-y-6">


        <Card className="border border-border/70 bg-card/85 backdrop-blur-sm">
          <CardHeader>
            <CardTitle>Worker event feed</CardTitle>
            <CardDescription>
              Messages decoded from <code>{demoEnv.statusTopic}</code>.
            </CardDescription>
            <CardAction>
              <Badge variant="outline" className="gap-1.5">
                <Radio className="size-3.5" />
                {workerEvents.length} buffered
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {workerEvents.length ? (
              <div className="space-y-3">
                {workerEvents.map((event) => (
                  <div
                    key={`${event.seq}-${event.ts_ms}`}
                    className="rounded-xl border border-border/70 bg-background/65 p-3"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">{event.type}</Badge>
                        <span className="text-xs text-muted-foreground">
                          #{event.seq}
                        </span>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {formatTimestamp(event.ts_ms)}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-foreground/85">
                      {summarizePayload(event.payload)}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-border/70 bg-muted/25 px-4 py-6 text-sm text-muted-foreground">
                Waiting for status messages. Once the worker joins, this panel will show <code>started</code>, frame metrics, and terminal events.
              </div>
            )}
          </CardContent>
        </Card>
        <Card className="border border-border/70 bg-background/55 shadow-none">
          <CardHeader>
            <CardTitle>Runtime snapshot</CardTitle>
            <CardDescription>
              Coordinator state and worker feedback from the status topic.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div className="grid gap-2 rounded-xl border border-border/70 bg-muted/35 p-3">
              <Row label="Session state" value={session?.state ?? "IDLE"} />
              <Row label="Room" value={session?.room_name ?? "-"} mono />
              <Row label="Session ID" value={session?.session_id ?? "-"} mono />
              <Row
                label="Prompt last sent"
                value={lastPromptSentAt ? formatTimestamp(lastPromptSentAt) : "Not sent yet"}
              />
              <Row
                label="Latest worker event"
                value={latestWorkerEvent ? latestWorkerEvent.type : "No messages yet"}
              />
            </div>

            {session?.error_code ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                Session error: {session.error_code}
              </div>
            ) : null}

            {requestError ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {requestError}
              </div>
            ) : null}

            {roomError ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                LiveKit room error: {roomError}
              </div>
            ) : null}

            {missingConfig.length ? (
              <div className="rounded-xl border border-border/70 bg-card px-3 py-2 text-sm text-muted-foreground">
                Missing configuration: {missingConfig.join(", ")}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </main>
  )
}

function Row({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className={mono ? "max-w-[15rem] truncate font-mono text-xs" : "font-medium"}>
        {value}
      </span>
    </div>
  )
}

export default App
