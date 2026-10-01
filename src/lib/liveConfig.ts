export const LIVE_CAPTURE_WIDTHS = [640, 768, 960, 1280] as const
export type LiveCaptureWidth = (typeof LIVE_CAPTURE_WIDTHS)[number]

export const LIVE_JPEG_QUALITIES = [0.6, 0.75, 0.85] as const
export type LiveJpegQuality = (typeof LIVE_JPEG_QUALITIES)[number]

export const DEFAULT_LIVE_CAPTURE_WIDTH: LiveCaptureWidth = 960
export const DEFAULT_LIVE_JPEG_QUALITY: LiveJpegQuality = 0.75

export const MODEL_IMGSZ_OPTIONS = [640, 768, 960, 1280] as const
export type ModelImgsz = (typeof MODEL_IMGSZ_OPTIONS)[number]

/** Deployed backend currently uses imgsz=1280; frontend cannot change it. */
export const DEPLOYED_MODEL_IMGSZ: ModelImgsz = 1280

export const BENCHMARK_SAMPLES_DEFAULT = 5
export const DETECTION_FPS_WINDOW_MS = 10_000

export const PERFORMANCE_TARGETS = {
  cameraFps: 30,
  detectionFpsMin: 3,
  detectionFpsMax: 5,
  totalLatencyMs: 500,
  maxInFlight: 1,
} as const

export const TEST_SCENARIOS = [
  'single_near',
  'single_far',
  'two_people',
  'walking',
  'crowded',
  'partial_occlusion',
  'person_enters_leaves',
  'low_light',
  'empty_scene',
  'camera_motion',
  'custom',
] as const

export type TestScenario = (typeof TEST_SCENARIOS)[number]

export const QUALITY_MARKERS = [
  'good',
  'missed_person',
  'false_positive',
  'bad_box',
] as const

export type QualityMarkerType = (typeof QUALITY_MARKERS)[number]
