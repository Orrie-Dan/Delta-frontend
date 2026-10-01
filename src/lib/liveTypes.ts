import type { Detection, DetectionResponse } from '../types/detection'
import type {
  LiveCaptureWidth,
  LiveJpegQuality,
  QualityMarkerType,
  TestScenario,
} from './liveConfig'

export interface FrameCaptureMetrics {
  blob: Blob
  sourceWidth: number
  sourceHeight: number
  captureWidth: number
  captureHeight: number
  captureMs: number
  encodeMs: number
  payloadBytes: number
}

export interface RequestTimingMetrics {
  requestTotalMs: number
  parseMs: number
  serverInferenceMs: number
  /** Approximate: request_total_ms - server_inference_ms (includes HTTP, decode, serialize, etc.) */
  nonInferenceOverheadMs: number
  modelImgsz: number | null
}

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
}

export interface BenchmarkSample {
  captureWidth: number
  captureHeight: number
  jpegQuality: number
  modelImgsz: number | null
  payloadKb: number
  captureMs: number
  encodeMs: number
  requestTotalMs: number
  serverInferenceMs: number
  nonInferenceOverheadMs: number
  peopleReturned: number
  averageConfidence: number | null
  timestamp: string
}

export interface BenchmarkConfigSummary {
  captureWidth: number
  modelImgsz: number | null
  samples: number
  avgPayloadKb: number
  avgCaptureMs: number
  avgEncodeMs: number
  avgRequestTotalMs: number
  medianRequestTotalMs: number
  p95RequestTotalMs: number | null
  avgServerInferenceMs: number
  avgNonInferenceOverheadMs: number
  avgPeopleReturned: number
  avgConfidence: number | null
  effectiveDetectionFps: number
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
  captureWidth: LiveCaptureWidth
  jpegQuality: LiveJpegQuality
  modelImgszNote: string
  records: LiveFrameMetrics[]
  markers: QualityMarkerEvent[]
}

export type { Detection, DetectionResponse }
