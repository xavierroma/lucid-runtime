/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_COORDINATOR_API_KEY?: string
  readonly VITE_COORDINATOR_BASE_URL?: string
  readonly VITE_COORDINATOR_PROXY_TARGET?: string
  readonly VITE_LIVEKIT_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
