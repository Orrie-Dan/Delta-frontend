import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from 'react'
import { ApiError, detectPeopleInstrumented } from '../api/detection'
import { captureVideoFrame } from '../lib/captureFrame'
import {
  BENCHMARK_SAMPLES_DEFAULT,
  DEFAULT_LIVE_CAPTURE_WIDTH,
  DEFAULT_LIVE_JPEG_QUALITY,
  DEPLOYED_MODEL_IMGSZ,
  DETECTION_FPS_WINDOW_MS,
  type LiveCaptureWidth,
  type LiveJpegQuality,
  type QualityMarkerType,
  type TestScenario,
} from '../lib/liveConfig'
import type {
  BenchmarkConfigSummary,
  BenchmarkSample,
  LiveDiagnosticsSnapshot,
  LiveFrameMetrics,
  QualityMarkerEvent,
  TestSessionExport,
} from '../lib/liveTypes'
import { average, averageNullable, median, percentile95 } from '../lib/stats'
import type {
  CameraLifecycle,
  DetectionResponse,
  LiveDetectionStats,
} from '../types/detection'

const LOOP_GAP_MS = 40

interface UseLiveDetectionOptions {
  enabled: boolean
  captureWidth: LiveCaptureWidth
  jpegQuality: LiveJpegQuality
  confidenceThreshold: number
  scenario: TestScenario
  recording: boolean
}

interface UseLiveDetectionResult {
  videoRef: RefObject<HTMLVideoElement | null>
  cameraState: CameraLifecycle
  isDetecting: boolean
  result: DetectionResponse | null
  errorMessage: string | null
  liveStats: LiveDetectionStats
  diagnostics: LiveDiagnosticsSnapshot
  videoDevices: MediaDeviceInfo[]
  activeDeviceId: string | null
  activeRequests: number
  lastRequestId: number | null
  sessionRecords: LiveFrameMetrics[]
  sessionMarkers: QualityMarkerEvent[]
  benchmarkRunning: boolean
  benchmarkProgress: string | null
  benchmarkSummaries: BenchmarkConfigSummary[]
  benchmarkSamples: BenchmarkSample[]
  startCamera: (deviceId?: string) => Promise<void>
  stopCamera: () => void
  switchCamera: () => Promise<void>
  addQualityMarker: (type: QualityMarkerType) => void
  clearSessionData: () => void
  exportSession: () => TestSessionExport | null
  runCaptureWidthBenchmark: (samplesPerConfig?: number) => Promise<void>
}

function classifyCameraError(error: unknown): {
  state: CameraLifecycle
  message: string
} {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      return {
        state: 'permission_denied',
        message:
          'Camera permission was denied. Allow camera access in the browser settings and try again.',
      }
    }
    if (
      error.name === 'NotFoundError' ||
      error.name === 'DevicesNotFoundError' ||
      error.name === 'OverconstrainedError'
    ) {
      return {
        state: 'unavailable',
        message: 'No suitable camera was found on this device.',
      }
    }
    if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      return {
        state: 'unavailable',
        message: 'The camera is already in use by another application.',
      }
    }
  }

  return {
    state: 'unavailable',
    message:
      error instanceof Error
        ? error.message
        : 'Unable to access the camera on this device.',
  }
}

function computeDetectionFps(timestamps: number[], now: number): number | null {
  const recent = timestamps.filter((ts) => now - ts <= DETECTION_FPS_WINDOW_MS)
  if (recent.length < 2) return null
  const elapsedSec = (recent[recent.length - 1]! - recent[0]!) / 1000
  if (elapsedSec <= 0) return null
  return (recent.length - 1) / elapsedSec
}

function averageConfidence(
  detections: DetectionResponse['detections'],
): number | null {
  if (detections.length === 0) return null
  return (
    detections.reduce((sum, detection) => sum + detection.confidence, 0) /
    detections.length
  )
}

function summarizeBenchmark(
  captureWidth: number,
  samples: BenchmarkSample[],
): BenchmarkConfigSummary {
  const totals = samples.map((s) => s.requestTotalMs)
  const servers = samples.map((s) => s.serverInferenceMs)
  const payloads = samples.map((s) => s.payloadKb)
  const people = samples.map((s) => s.peopleReturned)
  const confidences = samples.map((s) => s.averageConfidence)
  const avgTotal = average(totals)

  return {
    captureWidth,
    modelImgsz: samples[0]?.modelImgsz ?? DEPLOYED_MODEL_IMGSZ,
    samples: samples.length,
    avgPayloadKb: average(payloads),
    avgCaptureMs: average(samples.map((s) => s.captureMs)),
    avgEncodeMs: average(samples.map((s) => s.encodeMs)),
    avgRequestTotalMs: avgTotal,
    medianRequestTotalMs: median(totals),
    p95RequestTotalMs: percentile95(totals),
    avgServerInferenceMs: average(servers),
    avgNonInferenceOverheadMs: average(
      samples.map((s) => s.nonInferenceOverheadMs),
    ),
    avgPeopleReturned: average(people),
    avgConfidence: averageNullable(confidences),
    effectiveDetectionFps: avgTotal > 0 ? 1000 / avgTotal : 0,
  }
}

export function useLiveDetection({
  enabled,
  captureWidth,
  jpegQuality,
  confidenceThreshold,
  scenario,
  recording,
}: UseLiveDetectionOptions): UseLiveDetectionResult {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const runningRef = useRef(false)
  const inFlightRef = useRef(false)
  const activeRequestsRef = useRef(0)
  const sessionIdRef = useRef(0)
  const requestSeqRef = useRef(0)
  const appliedSeqRef = useRef(0)
  const abortRef = useRef<AbortController | null>(null)
  const completionTimestampsRef = useRef<number[]>([])
  const loopTimerRef = useRef<number | null>(null)
  const captureWidthRef = useRef(captureWidth)
  const jpegQualityRef = useRef(jpegQuality)
  const confidenceRef = useRef(confidenceThreshold)
  const recordingRef = useRef(recording)
  const scenarioRef = useRef(scenario)
  const cameraFpsRef = useRef<number | null>(null)
  const pauseLoopRef = useRef(false)
  const rafRef = useRef<number | null>(null)
  const cameraFrameCountRef = useRef(0)
  const cameraFpsWindowStartRef = useRef(0)

  const [cameraState, setCameraState] = useState<CameraLifecycle>('off')
  const [isDetecting, setIsDetecting] = useState(false)
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [activeRequests, setActiveRequests] = useState(0)
  const [lastRequestId, setLastRequestId] = useState<number | null>(null)
  const [liveStats, setLiveStats] = useState<LiveDetectionStats>({
    people: 0,
    inferenceMs: null,
    requestTotalMs: null,
    detectionFps: null,
    cameraFps: null,
  })
  const [diagnostics, setDiagnostics] = useState<LiveDiagnosticsSnapshot>({
    cameraWidth: null,
    cameraHeight: null,
    captureWidth: null,
    captureHeight: null,
    payloadKb: null,
    captureMs: null,
    encodeMs: null,
    requestTotalMs: null,
    serverInferenceMs: null,
    nonInferenceOverheadMs: null,
    detectionFps: null,
    cameraFps: null,
    visibleDetections: 0,
    returnedDetections: 0,
    confidenceThreshold,
    activeRequests: 0,
    currentCaptureWidthSetting: captureWidth,
    currentJpegQuality: jpegQuality,
    modelImgsz: DEPLOYED_MODEL_IMGSZ,
  })
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([])
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null)
  const [sessionRecords, setSessionRecords] = useState<LiveFrameMetrics[]>([])
  const [sessionMarkers, setSessionMarkers] = useState<QualityMarkerEvent[]>([])
  const [benchmarkRunning, setBenchmarkRunning] = useState(false)
  const [benchmarkProgress, setBenchmarkProgress] = useState<string | null>(null)
  const [benchmarkSummaries, setBenchmarkSummaries] = useState<
    BenchmarkConfigSummary[]
  >([])
  const [benchmarkSamples, setBenchmarkSamples] = useState<BenchmarkSample[]>([])
  const sessionStartedAtRef = useRef<string | null>(null)

  useEffect(() => {
    captureWidthRef.current = captureWidth
    jpegQualityRef.current = jpegQuality
    confidenceRef.current = confidenceThreshold
    recordingRef.current = recording
    scenarioRef.current = scenario
    setDiagnostics((prev) => ({
      ...prev,
      confidenceThreshold,
      currentCaptureWidthSetting: captureWidth,
      currentJpegQuality: jpegQuality,
    }))
  }, [captureWidth, jpegQuality, confidenceThreshold, recording, scenario])

  useEffect(() => {
    if (recording && !sessionStartedAtRef.current) {
      sessionStartedAtRef.current = new Date().toISOString()
    }
    if (!recording) {
      // keep startedAt until export/clear
    }
  }, [recording])

  const clearLoopTimer = useCallback(() => {
    if (loopTimerRef.current !== null) {
      window.clearTimeout(loopTimerRef.current)
      loopTimerRef.current = null
    }
  }, [])

  const stopCameraFpsMonitor = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  const startCameraFpsMonitor = useCallback(() => {
    stopCameraFpsMonitor()
    cameraFrameCountRef.current = 0
    cameraFpsWindowStartRef.current = performance.now()

    const tick = () => {
      cameraFrameCountRef.current += 1
      const now = performance.now()
      const elapsed = now - cameraFpsWindowStartRef.current
      if (elapsed >= 1000) {
        const fps = (cameraFrameCountRef.current * 1000) / elapsed
        cameraFpsRef.current = fps
        setLiveStats((prev) => ({ ...prev, cameraFps: fps }))
        setDiagnostics((prev) => ({ ...prev, cameraFps: fps }))
        cameraFrameCountRef.current = 0
        cameraFpsWindowStartRef.current = now
      }
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
  }, [stopCameraFpsMonitor])

  const stopTracks = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop()
      }
      streamRef.current = null
    }
    const video = videoRef.current
    if (video) {
      video.srcObject = null
    }
  }, [])

  const setInFlight = useCallback((value: boolean) => {
    inFlightRef.current = value
    activeRequestsRef.current = value ? 1 : 0
    setActiveRequests(activeRequestsRef.current)
    if (import.meta.env.DEV && activeRequestsRef.current > 1) {
      console.error('[live] activeRequests exceeded 1:', activeRequestsRef.current)
    }
  }, [])

  const stopCamera = useCallback(() => {
    runningRef.current = false
    pauseLoopRef.current = false
    clearLoopTimer()
    abortRef.current?.abort()
    abortRef.current = null
    setInFlight(false)
    sessionIdRef.current += 1
    requestSeqRef.current = 0
    appliedSeqRef.current = 0
    completionTimestampsRef.current = []
    stopCameraFpsMonitor()
    stopTracks()
    setIsDetecting(false)
    setResult(null)
    setErrorMessage(null)
    setLastRequestId(null)
    cameraFpsRef.current = null
    setLiveStats({
      people: 0,
      inferenceMs: null,
      requestTotalMs: null,
      detectionFps: null,
      cameraFps: null,
    })
    setDiagnostics((prev) => ({
      ...prev,
      cameraWidth: null,
      cameraHeight: null,
      captureWidth: null,
      captureHeight: null,
      payloadKb: null,
      captureMs: null,
      encodeMs: null,
      requestTotalMs: null,
      serverInferenceMs: null,
      nonInferenceOverheadMs: null,
      detectionFps: null,
      cameraFps: null,
      visibleDetections: 0,
      returnedDetections: 0,
      activeRequests: 0,
    }))
    setCameraState('off')
    setActiveDeviceId(null)
  }, [clearLoopTimer, setInFlight, stopCameraFpsMonitor, stopTracks])

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setVideoDevices([])
      return
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      setVideoDevices(devices.filter((device) => device.kind === 'videoinput'))
    } catch {
      setVideoDevices([])
    }
  }, [])

  const runDetectionLoop = useCallback(
    async (sessionId: number) => {
      if (!runningRef.current || sessionId !== sessionIdRef.current) return
      if (pauseLoopRef.current) {
        loopTimerRef.current = window.setTimeout(() => {
          void runDetectionLoop(sessionId)
        }, 200)
        return
      }
      if (inFlightRef.current) return

      const video = videoRef.current
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        loopTimerRef.current = window.setTimeout(() => {
          void runDetectionLoop(sessionId)
        }, 120)
        return
      }

      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas')
      }

      setInFlight(true)
      setIsDetecting(true)
      const requestId = ++requestSeqRef.current
      const abortController = new AbortController()
      abortRef.current = abortController

      try {
        const capture = await captureVideoFrame(video, canvasRef.current, {
          captureWidth: captureWidthRef.current,
          jpegQuality: jpegQualityRef.current,
        })

        if (!runningRef.current || sessionId !== sessionIdRef.current) return

        const { response, timing } = await detectPeopleInstrumented(capture.blob, {
          filename: 'frame.jpg',
          signal: abortController.signal,
        })

        // Blob is not retained after this function scope ends.
        if (!runningRef.current || sessionId !== sessionIdRef.current) return
        if (requestId < appliedSeqRef.current) return

        appliedSeqRef.current = requestId
        setLastRequestId(requestId)
        setResult(response)
        setErrorMessage(null)

        const now = performance.now()
        completionTimestampsRef.current.push(now)
        completionTimestampsRef.current = completionTimestampsRef.current.filter(
          (ts) => now - ts <= DETECTION_FPS_WINDOW_MS,
        )
        const detectionFps = computeDetectionFps(
          completionTimestampsRef.current,
          now,
        )
        const visible = response.detections.filter(
          (d) => d.confidence >= confidenceRef.current,
        ).length
        const avgConf = averageConfidence(response.detections)

        setLiveStats({
          people: response.people,
          inferenceMs: timing.serverInferenceMs,
          requestTotalMs: timing.requestTotalMs,
          detectionFps,
          cameraFps: cameraFpsRef.current,
        })

        const metrics: LiveFrameMetrics = {
          requestId,
          sessionId,
          timestamp: new Date().toISOString(),
          sourceWidth: capture.sourceWidth,
          sourceHeight: capture.sourceHeight,
          captureWidth: capture.captureWidth,
          captureHeight: capture.captureHeight,
          jpegQuality: jpegQualityRef.current,
          payloadBytes: capture.payloadBytes,
          payloadKb: capture.payloadBytes / 1024,
          captureMs: capture.captureMs,
          encodeMs: capture.encodeMs,
          requestTotalMs: timing.requestTotalMs,
          serverInferenceMs: timing.serverInferenceMs,
          nonInferenceOverheadMs: timing.nonInferenceOverheadMs,
          parseMs: timing.parseMs,
          peopleReturned: response.people,
          visibleDetections: visible,
          averageConfidence: avgConf,
          modelImgsz: timing.modelImgsz ?? DEPLOYED_MODEL_IMGSZ,
          detectionFps,
          cameraFps: cameraFpsRef.current,
          activeRequests: 1,
          confidenceThreshold: confidenceRef.current,
        }

        setDiagnostics({
          cameraWidth: capture.sourceWidth,
          cameraHeight: capture.sourceHeight,
          captureWidth: capture.captureWidth,
          captureHeight: capture.captureHeight,
          payloadKb: metrics.payloadKb,
          captureMs: metrics.captureMs,
          encodeMs: metrics.encodeMs,
          requestTotalMs: metrics.requestTotalMs,
          serverInferenceMs: metrics.serverInferenceMs,
          nonInferenceOverheadMs: metrics.nonInferenceOverheadMs,
          detectionFps,
          cameraFps: cameraFpsRef.current,
          visibleDetections: visible,
          returnedDetections: response.people,
          confidenceThreshold: confidenceRef.current,
          activeRequests: 1,
          currentCaptureWidthSetting: captureWidthRef.current,
          currentJpegQuality: jpegQualityRef.current,
          modelImgsz: metrics.modelImgsz,
        })

        if (recordingRef.current) {
          setSessionRecords((prev) => [...prev, metrics])
        }

        if (import.meta.env.DEV) {
          console.debug('[live] detect complete', {
            requestId,
            activeRequests: activeRequestsRef.current,
            capture: `${capture.captureWidth}x${capture.captureHeight}`,
            payloadKb: metrics.payloadKb.toFixed(1),
            requestTotalMs: metrics.requestTotalMs.toFixed(0),
            serverMs: metrics.serverInferenceMs.toFixed(0),
          })
        }
      } catch (error) {
        if (!runningRef.current || sessionId !== sessionIdRef.current) return
        if (error instanceof ApiError && error.message.includes('cancelled')) {
          return
        }

        const message =
          error instanceof ApiError
            ? error.message
            : error instanceof Error
              ? error.message
              : 'Live detection failed.'

        setErrorMessage(message)
      } finally {
        if (abortRef.current === abortController) {
          abortRef.current = null
        }
        setInFlight(false)
        if (sessionId === sessionIdRef.current) {
          setIsDetecting(false)
          setDiagnostics((prev) => ({ ...prev, activeRequests: 0 }))
        }

        if (runningRef.current && sessionId === sessionIdRef.current) {
          clearLoopTimer()
          loopTimerRef.current = window.setTimeout(() => {
            void runDetectionLoop(sessionId)
          }, LOOP_GAP_MS)
        }
      }
    },
    [clearLoopTimer, setInFlight],
  )

  const startCamera = useCallback(
    async (deviceId?: string) => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraState('unavailable')
        setErrorMessage('Camera APIs are not available in this browser.')
        return
      }

      stopCamera()
      setCameraState('requesting')
      setErrorMessage(null)

      const constraints: MediaStreamConstraints = {
        audio: false,
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          : {
              facingMode: { ideal: 'environment' },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints)
        streamRef.current = stream

        const video = videoRef.current
        if (!video) {
          for (const track of stream.getTracks()) track.stop()
          streamRef.current = null
          setCameraState('unavailable')
          setErrorMessage('Camera preview element is not ready.')
          return
        }

        video.srcObject = stream
        video.muted = true
        video.playsInline = true
        await video.play()

        const track = stream.getVideoTracks()[0]
        const settings = track?.getSettings()
        setActiveDeviceId(settings?.deviceId ?? deviceId ?? null)
        await refreshDevices()

        const sessionId = sessionIdRef.current + 1
        sessionIdRef.current = sessionId
        requestSeqRef.current = 0
        appliedSeqRef.current = 0
        completionTimestampsRef.current = []
        runningRef.current = true
        pauseLoopRef.current = false

        setCameraState('active')
        setResult(null)
        setLiveStats({
          people: 0,
          inferenceMs: null,
          requestTotalMs: null,
          detectionFps: null,
          cameraFps: null,
        })
        startCameraFpsMonitor()
        void runDetectionLoop(sessionId)
      } catch (error) {
        stopTracks()
        const classified = classifyCameraError(error)
        setCameraState(classified.state)
        setErrorMessage(classified.message)
      }
    },
    [
      refreshDevices,
      runDetectionLoop,
      startCameraFpsMonitor,
      stopCamera,
      stopTracks,
    ],
  )

  const switchCamera = useCallback(async () => {
    if (videoDevices.length < 2) return
    const currentIndex = videoDevices.findIndex(
      (device) => device.deviceId === activeDeviceId,
    )
    const next =
      videoDevices[(currentIndex + 1 + videoDevices.length) % videoDevices.length]
    if (!next) return
    await startCamera(next.deviceId)
  }, [activeDeviceId, startCamera, videoDevices])

  const addQualityMarker = useCallback(
    (type: QualityMarkerType) => {
      if (!recordingRef.current) return
      const marker: QualityMarkerEvent = {
        type,
        timestamp: new Date().toISOString(),
        requestId: lastRequestId,
        peopleCount: result?.people ?? 0,
        captureWidth: captureWidthRef.current,
        jpegQuality: jpegQualityRef.current,
        modelImgsz: DEPLOYED_MODEL_IMGSZ,
        scenario: scenarioRef.current,
      }
      setSessionMarkers((prev) => [...prev, marker])
    },
    [lastRequestId, result],
  )

  const clearSessionData = useCallback(() => {
    setSessionRecords([])
    setSessionMarkers([])
    sessionStartedAtRef.current = null
  }, [])

  const exportSession = useCallback((): TestSessionExport | null => {
    const startedAt = sessionStartedAtRef.current
    if (!startedAt) return null
    return {
      startedAt,
      stoppedAt: new Date().toISOString(),
      scenario: scenarioRef.current,
      captureWidth: captureWidthRef.current,
      jpegQuality: jpegQualityRef.current,
      modelImgszNote: `Deployed backend fixed at imgsz=${DEPLOYED_MODEL_IMGSZ} (not configurable from this frontend).`,
      records: sessionRecords,
      markers: sessionMarkers,
    }
  }, [sessionMarkers, sessionRecords])

  const runCaptureWidthBenchmark = useCallback(
    async (samplesPerConfig = BENCHMARK_SAMPLES_DEFAULT) => {
      if (benchmarkRunning) return
      const video = videoRef.current
      if (!video || cameraState !== 'active') {
        setErrorMessage('Start the camera before running a live benchmark.')
        return
      }
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas')
      }

      setBenchmarkRunning(true)
      pauseLoopRef.current = true
      setBenchmarkProgress('Starting capture-width benchmark…')
      setBenchmarkSummaries([])
      setBenchmarkSamples([])

      const widths: LiveCaptureWidth[] = [640, 768, 960, 1280]
      const allSamples: BenchmarkSample[] = []
      const summaries: BenchmarkConfigSummary[] = []

      try {
        for (const width of widths) {
          const configSamples: BenchmarkSample[] = []
          for (let i = 0; i < samplesPerConfig; i += 1) {
            if (inFlightRef.current) {
              throw new Error('Benchmark refused to start while a request is in flight.')
            }

            setBenchmarkProgress(
              `Capture width ${width} — sample ${i + 1}/${samplesPerConfig}`,
            )
            setInFlight(true)

            try {
              const capture = await captureVideoFrame(video, canvasRef.current, {
                captureWidth: width,
                jpegQuality: jpegQualityRef.current,
              })
              const { response, timing } = await detectPeopleInstrumented(
                capture.blob,
                { filename: 'frame.jpg' },
              )

              const sample: BenchmarkSample = {
                captureWidth: capture.captureWidth,
                captureHeight: capture.captureHeight,
                jpegQuality: jpegQualityRef.current,
                modelImgsz: timing.modelImgsz ?? DEPLOYED_MODEL_IMGSZ,
                payloadKb: capture.payloadBytes / 1024,
                captureMs: capture.captureMs,
                encodeMs: capture.encodeMs,
                requestTotalMs: timing.requestTotalMs,
                serverInferenceMs: timing.serverInferenceMs,
                nonInferenceOverheadMs: timing.nonInferenceOverheadMs,
                peopleReturned: response.people,
                averageConfidence: averageConfidence(response.detections),
                timestamp: new Date().toISOString(),
              }
              configSamples.push(sample)
              allSamples.push(sample)
            } finally {
              setInFlight(false)
            }
          }
          summaries.push(summarizeBenchmark(width, configSamples))
        }

        setBenchmarkSamples(allSamples)
        setBenchmarkSummaries(summaries)
        setBenchmarkProgress('Benchmark complete')
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'Benchmark failed.'
        setErrorMessage(message)
        setBenchmarkProgress('Benchmark failed')
      } finally {
        pauseLoopRef.current = false
        setBenchmarkRunning(false)
        if (runningRef.current) {
          void runDetectionLoop(sessionIdRef.current)
        }
      }
    },
    [benchmarkRunning, cameraState, runDetectionLoop, setInFlight],
  )

  useEffect(() => {
    if (!enabled && runningRef.current) {
      stopCamera()
    }
  }, [enabled, stopCamera])

  useEffect(() => {
    return () => {
      runningRef.current = false
      clearLoopTimer()
      abortRef.current?.abort()
      stopCameraFpsMonitor()
      stopTracks()
    }
  }, [clearLoopTimer, stopCameraFpsMonitor, stopTracks])

  return {
    videoRef,
    cameraState,
    isDetecting,
    result,
    errorMessage,
    liveStats,
    diagnostics,
    videoDevices,
    activeDeviceId,
    activeRequests,
    lastRequestId,
    sessionRecords,
    sessionMarkers,
    benchmarkRunning,
    benchmarkProgress,
    benchmarkSummaries,
    benchmarkSamples,
    startCamera,
    stopCamera,
    switchCamera,
    addQualityMarker,
    clearSessionData,
    exportSession,
    runCaptureWidthBenchmark,
  }
}

// Re-export defaults for panel convenience
export { DEFAULT_LIVE_CAPTURE_WIDTH, DEFAULT_LIVE_JPEG_QUALITY }
