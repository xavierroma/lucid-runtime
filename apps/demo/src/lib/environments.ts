export interface SavedEnvironment {
  id: string
  name: string
  prompt: string
  seedImageDataUrl: string | null
  createdAt: string
  updatedAt: string
}

const ENVIRONMENTS_STORAGE_KEY = "lucid.demo.environments.v1"
const SELECTED_ENVIRONMENT_STORAGE_KEY = "lucid.demo.selected-environment.v1"

function canUseStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined"
}

function coerceEnvironment(value: unknown): SavedEnvironment | null {
  if (!value || typeof value !== "object") {
    return null
  }

  const candidate = value as Record<string, unknown>
  const id = typeof candidate.id === "string" ? candidate.id.trim() : ""
  const name = typeof candidate.name === "string" ? candidate.name.trim() : ""
  const prompt = typeof candidate.prompt === "string" ? candidate.prompt.trim() : ""
  const seedImageDataUrl =
    typeof candidate.seedImageDataUrl === "string"
      ? candidate.seedImageDataUrl.trim()
      : null
  const createdAt =
    typeof candidate.createdAt === "string" ? candidate.createdAt : new Date().toISOString()
  const updatedAt =
    typeof candidate.updatedAt === "string" ? candidate.updatedAt : createdAt

  if (!id || !name || !prompt) {
    return null
  }

  return {
    id,
    name,
    prompt,
    seedImageDataUrl: seedImageDataUrl || null,
    createdAt,
    updatedAt,
  }
}

function sortEnvironments(environments: SavedEnvironment[]) {
  return [...environments].sort((left, right) =>
    right.updatedAt.localeCompare(left.updatedAt),
  )
}

export function createEnvironmentId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `env-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

export function loadSavedEnvironments() {
  if (!canUseStorage()) {
    return [] as SavedEnvironment[]
  }

  try {
    const raw = window.localStorage.getItem(ENVIRONMENTS_STORAGE_KEY)
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return []
    }

    return sortEnvironments(
      parsed
        .map((value) => coerceEnvironment(value))
        .filter((environment): environment is SavedEnvironment => environment !== null),
    )
  } catch {
    return []
  }
}

export function persistSavedEnvironments(environments: SavedEnvironment[]) {
  if (!canUseStorage()) {
    return
  }

  window.localStorage.setItem(
    ENVIRONMENTS_STORAGE_KEY,
    JSON.stringify(sortEnvironments(environments)),
  )
}

export function loadSelectedEnvironmentId() {
  if (!canUseStorage()) {
    return null
  }

  const value = window.localStorage.getItem(SELECTED_ENVIRONMENT_STORAGE_KEY)
  return value?.trim() || null
}

export function persistSelectedEnvironmentId(environmentId: string | null) {
  if (!canUseStorage()) {
    return
  }

  if (!environmentId) {
    window.localStorage.removeItem(SELECTED_ENVIRONMENT_STORAGE_KEY)
    return
  }

  window.localStorage.setItem(SELECTED_ENVIRONMENT_STORAGE_KEY, environmentId)
}
