import { demoEnv } from "@/lib/env"
import { CoordinatorClient } from "@/lib/lucid"

export type {
  AxisInputBinding,
  Capabilities,
  HoldInputBinding,
  InputBinding,
  LucidManifest,
  ManifestInput,
  ManifestOutput,
  ModelsResponse,
  OutputBinding,
  PointerInputBinding,
  PressInputBinding,
  SessionRecord,
  SessionResponse,
  SessionState,
  SupportedModel,
  WheelInputBinding,
} from "@/lib/lucid"
export { isTerminalSessionState } from "@/lib/lucid"

const coordinator = new CoordinatorClient({
  baseUrl: demoEnv.coordinatorBaseUrl,
  apiKey: demoEnv.coordinatorApiKey,
})

export function getModels() {
  return coordinator.getModels()
}

export function createSession(modelName: string) {
  return coordinator.createSession(modelName)
}

export function getSession(sessionId: string) {
  return coordinator.getSession(sessionId)
}

export function endSession(sessionId: string) {
  return coordinator.endSession(sessionId)
}
