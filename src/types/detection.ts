export interface Detection {
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface DetectionImage {
  width: number
  height: number
}

export interface DetectionResponse {
  people: number
  inference_ms: number
  image: DetectionImage
  detections: Detection[]
  /** Present only if backend supports configurable imgsz. */
  model_imgsz?: number
}

export interface HealthResponse {
  status: string
  model: string
  image_size: number
}

export type ApiStatus = 'checking' | 'online' | 'offline'

export type AppPhase = 'upload' | 'preview' | 'detecting' | 'results' | 'error'

export type WorkspaceMode = 'upload' | 'live'

export type CameraLifecycle =
  | 'off'
  | 'requesting'
  | 'active'
  | 'permission_denied'
  | 'unavailable'

export interface LiveDetectionStats {
  people: number
  inferenceMs: number | null
  requestTotalMs: number | null
  detectionFps: number | null
  cameraFps: number | null
}
