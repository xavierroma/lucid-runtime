const trim = (value: string | undefined) => value?.trim() ?? ""

const explicitCoordinatorBaseUrl = trim(import.meta.env.VITE_COORDINATOR_BASE_URL)
const proxyTarget = trim(import.meta.env.VITE_COORDINATOR_PROXY_TARGET)
const devProxyBaseUrl =
  import.meta.env.DEV && !explicitCoordinatorBaseUrl && proxyTarget ? "/api" : ""
const defaultModel = trim(import.meta.env.VITE_DEFAULT_MODEL).toLowerCase()

export const demoEnv = {
  coordinatorApiKey: trim(import.meta.env.VITE_COORDINATOR_API_KEY),
  coordinatorBaseUrl: explicitCoordinatorBaseUrl || devProxyBaseUrl,
  livekitUrl: trim(import.meta.env.VITE_LIVEKIT_URL),
  defaultModel,
}

export function getMissingConfig(): string[] {
  const missing: string[] = []
  const usingDirectCoordinator = Boolean(explicitCoordinatorBaseUrl)
  const usingProxyCoordinator = Boolean(devProxyBaseUrl)

  if (usingDirectCoordinator && !demoEnv.coordinatorApiKey) {
    missing.push("VITE_COORDINATOR_API_KEY")
  }
  if (!demoEnv.livekitUrl) {
    missing.push("VITE_LIVEKIT_URL")
  }
  if (!demoEnv.coordinatorBaseUrl) {
    missing.push("VITE_COORDINATOR_BASE_URL or VITE_COORDINATOR_PROXY_TARGET")
  }
  if (import.meta.env.DEV && !usingDirectCoordinator && !usingProxyCoordinator) {
    missing.push("VITE_COORDINATOR_PROXY_TARGET")
  }

  return missing
}
