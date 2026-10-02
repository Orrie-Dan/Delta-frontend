import type {
  LiveCaptureWidth,
  LiveJpegQuality,
  QualityMarkerType,
  TestScenario,
} from './liveConfig'

export interface LiveFrameMetrics {
  requestId: number
  sessionId: number
  timestamp: string
  sourceWidth: number
  sourceHeight: number
  captureWidth: number
  captureHeight: number
  jpegQuality: number
  payloadBytes: number
  payloadKb: number
  captureMs: number
  encodeMs: number
  requestTotalMs: number
  serverInferenceMs: number
  nonInferenceOverheadMs: number
  parseMs: number
  peopleReturned: number
  visibleDetections: number
  averageConfidence: number | null
  modelImgsz: number | null
  detectionFps: number | null
  cameraFps: number | null
  activeRequests: number
  confidenceThreshold: number
}

export interface LiveDiagnosticsSnapshot {
  cameraWidth: number | null
  cameraHeight: number | null
  captureWidth: number | null
  captureHeight: number | null
  payloadKb: number | null
  captureMs: number | null
  encodeMs: number | null
  requestTotalMs: number | null
  serverInferenceMs: number | null
  nonInferenceOverheadMs: number | null
  detectionFps: number | null
  cameraFps: number | null
  visibleDetections: number
  returnedDetections: number
  confidenceThreshold: number
  activeRequests: number
  currentCaptureWidthSetting: LiveCaptureWidth
  currentJpegQuality: LiveJpegQuality
  modelImgsz: number | null
  completedInferences: number
  droppedFrames: number
  latestInferenceMs: number | null
  medianInferenceMs: number | null
  medianTotalMs: number | null
  effectiveInferenceFps: number | null
  maxConcurrentInference: number
}

export interface QualityMarkerEvent {
  type: QualityMarkerType
  timestamp: string
  requestId: number | null
  peopleCount: number
  captureWidth: number
  jpegQuality: number
  modelImgsz: number | null
  scenario: TestScenario
}

export interface TestSessionExport {
  startedAt: string
  stoppedAt: string
  scenario: TestScenario
  captureWidth: number
  jpegQuality: number
  modelImgszNote: string
  records: LiveFrameMetrics[]
  markers: QualityMarkerEvent[]
}
