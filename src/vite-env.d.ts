/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Railway production API base URL for live-camera cloud fallback. */
  readonly VITE_DETECTION_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
