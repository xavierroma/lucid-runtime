import { demoEnv } from "@/lib/env"
import { CoordinatorClient } from "../../../../packages/lucid-ts/src/index.ts"

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
} from "../../../../packages/lucid-ts/src/index.ts"
export { isTerminalSessionState } from "../../../../packages/lucid-ts/src/index.ts"

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
