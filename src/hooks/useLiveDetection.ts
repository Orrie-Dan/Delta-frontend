import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from 'react'
import {
  DEFAULT_LIVE_CAPTURE_WIDTH,
  DEFAULT_LIVE_JPEG_QUALITY,
  DETECTION_FPS_WINDOW_MS,
  type LiveCaptureWidth,
  type LiveJpegQuality,
  type QualityMarkerType,
  type TestScenario,
} from '../lib/liveConfig'
import type {
  LiveDiagnosticsSnapshot,
  LiveFrameMetrics,
  QualityMarkerEvent,
  TestSessionExport,
} from '../lib/liveTypes'
import { runIsolatedBrowserInference } from '../lib/onnxInference'
import {
  LIVE_WEBGPU_IMGSZ,
  endWebGpuInference,
  getWebGpuInferenceOptions,
  getWebGpuInferenceSlots,
  getWebGpuModelFetchCount,
  getWebGpuRuntimeSnapshot,
  getWebGpuSessionCreations,
  tryBeginWebGpuInference,
} from '../lib/onnxWebGpu'
import { median, percentile95 } from '../lib/stats'
import type {
  CameraLifecycle,
  DetectionResponse,
  LiveDetectionStats,
} from '../types/detection'

const WEBGPU_REQUIRED_MESSAGE =
  'Live detection requires WebGPU on this device/browser.'

export interface LiveInferenceDebugStats {
  completed: number
  dropped: number
  latestInferenceMs: number | null
  latestTotalMs: number | null
  medianInferenceMs: number | null
  medianTotalMs: number | null
  p95TotalMs: number | null
  effectiveFps: number | null
  maxConcurrentInference: number
  activeInFlight: number
  sessionIdentity: number | null
  modelImgsz: number
  executionProvider: 'webgpu'
  sessionStartedAtMs: number | null
  modelFetchCount: number
  sessionCreations: number
}

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
  startCamera: (deviceId?: string) => Promise<void>
  stopCamera: () => void
  switchCamera: () => Promise<void>
  addQualityMarker: (type: QualityMarkerType) => void
  clearSessionData: () => void
  exportSession: () => TestSessionExport | null
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
  const runningRef = useRef(false)
  const inFlightRef = useRef(false)
  const activeRequestsRef = useRef(0)
  const sessionIdRef = useRef(0)
  const requestSeqRef = useRef(0)
  const appliedSeqRef = useRef(0)
  const abortRef = useRef<AbortController | null>(null)
  const completionTimestampsRef = useRef<number[]>([])
  const loopTimerRef = useRef<number | null>(null)
  const detectionRafRef = useRef<number | null>(null)
  const completedRef = useRef(0)
  const droppedRef = useRef(0)
  const maxConcurrentRef = useRef(0)
  const liveStartedAtRef = useRef<number | null>(null)
  const inferenceSamplesRef = useRef<number[]>([])
  const totalSamplesRef = useRef<number[]>([])
  const latestInferenceMsRef = useRef<number | null>(null)
  const latestTotalMsRef = useRef<number | null>(null)
  const sessionIdentityRef = useRef<number | null>(null)
  const lastDropUiRef = useRef(0)
  const webGpuBlockedRef = useRef(false)
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
    modelImgsz: LIVE_WEBGPU_IMGSZ,
    completedInferences: 0,
    droppedFrames: 0,
    latestInferenceMs: null,
    medianInferenceMs: null,
    medianTotalMs: null,
    effectiveInferenceFps: null,
    maxConcurrentInference: 0,
  })
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([])
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null)
  const [sessionRecords, setSessionRecords] = useState<LiveFrameMetrics[]>([])
  const [sessionMarkers, setSessionMarkers] = useState<QualityMarkerEvent[]>([])
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

  const stopDetectionLoop = useCallback(() => {
    if (detectionRafRef.current !== null) {
      cancelAnimationFrame(detectionRafRef.current)
      detectionRafRef.current = null
    }
    clearLoopTimer()
  }, [clearLoopTimer])

  const stopCamera = useCallback(() => {
    runningRef.current = false
    pauseLoopRef.current = false
    stopDetectionLoop()
    abortRef.current?.abort()
    abortRef.current = null
    setInFlight(false)
    sessionIdRef.current += 1
    requestSeqRef.current = 0
    appliedSeqRef.current = 0
    completionTimestampsRef.current = []
    completedRef.current = 0
    droppedRef.current = 0
    maxConcurrentRef.current = 0
    liveStartedAtRef.current = null
    inferenceSamplesRef.current = []
    totalSamplesRef.current = []
    latestInferenceMsRef.current = null
    latestTotalMsRef.current = null
    sessionIdentityRef.current = null
    webGpuBlockedRef.current = false
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
      completedInferences: 0,
      droppedFrames: 0,
      latestInferenceMs: null,
      medianInferenceMs: null,
      medianTotalMs: null,
      effectiveInferenceFps: null,
      maxConcurrentInference: 0,
      modelImgsz: LIVE_WEBGPU_IMGSZ,
    }))
    setCameraState('off')
    setActiveDeviceId(null)
  }, [setInFlight, stopCameraFpsMonitor, stopDetectionLoop, stopTracks])

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

  const publishDebugStats = useCallback(() => {
    const started = liveStartedAtRef.current
    const elapsedSec =
      started == null ? 0 : Math.max(0, (performance.now() - started) / 1000)
    const inferenceSamples = inferenceSamplesRef.current
    const totalSamples = totalSamplesRef.current
    const stats: LiveInferenceDebugStats = {
      completed: completedRef.current,
      dropped: droppedRef.current,
      latestInferenceMs: latestInferenceMsRef.current,
      latestTotalMs: latestTotalMsRef.current,
      medianInferenceMs:
        inferenceSamples.length > 0 ? median(inferenceSamples) : null,
      medianTotalMs: totalSamples.length > 0 ? median(totalSamples) : null,
      p95TotalMs: percentile95(totalSamples),
      effectiveFps:
        elapsedSec > 0 ? completedRef.current / elapsedSec : null,
      maxConcurrentInference: maxConcurrentRef.current,
      activeInFlight: getWebGpuInferenceSlots(),
      sessionIdentity: sessionIdentityRef.current,
      modelImgsz: LIVE_WEBGPU_IMGSZ,
      executionProvider: 'webgpu',
      sessionStartedAtMs: started,
      modelFetchCount: getWebGpuModelFetchCount(),
      sessionCreations: getWebGpuSessionCreations(),
    }
    ;(
      window as Window & { __liveInferenceStats?: LiveInferenceDebugStats }
    ).__liveInferenceStats = stats
    return stats
  }, [])

  const runDetectionLoop = useCallback(
    (sessionId: number) => {
      const schedule = () => {
        detectionRafRef.current = requestAnimationFrame(tick)
      }

      const tick = () => {
        if (!runningRef.current || sessionId !== sessionIdRef.current) return

        if (pauseLoopRef.current) {
          loopTimerRef.current = window.setTimeout(schedule, 200)
          return
        }

        const video = videoRef.current
        if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
          schedule()
          return
        }

        schedule()

        const runtime = getWebGpuInferenceOptions()
        if (!runtime) {
          if (!webGpuBlockedRef.current) {
            webGpuBlockedRef.current = true
            setErrorMessage(WEBGPU_REQUIRED_MESSAGE)
          }
          return
        }

        if (inFlightRef.current || !tryBeginWebGpuInference()) {
          droppedRef.current += 1
          if (import.meta.env.DEV) {
            const liveStats = (
              window as Window & { __liveInferenceStats?: LiveInferenceDebugStats }
            ).__liveInferenceStats
            if (liveStats) liveStats.dropped = droppedRef.current
          }
          const now = performance.now()
          if (now - lastDropUiRef.current >= 500) {
            lastDropUiRef.current = now
            const stats = publishDebugStats()
            setDiagnostics((prev) => ({
              ...prev,
              droppedFrames: stats.dropped,
              completedInferences: stats.completed,
              effectiveInferenceFps: stats.effectiveFps,
              maxConcurrentInference: stats.maxConcurrentInference,
              activeRequests: inFlightRef.current ? 1 : 0,
            }))
          }
          return
        }

        inFlightRef.current = true
        activeRequestsRef.current = 1
        maxConcurrentRef.current = Math.max(
          maxConcurrentRef.current,
          getWebGpuInferenceSlots(),
        )
        setInFlight(true)
        setIsDetecting(true)
        const requestId = ++requestSeqRef.current

        void (async () => {
          let releaseSlot = true
          try {
            const result = await runIsolatedBrowserInference(video, {
              ...runtime,
              modelImgsz: LIVE_WEBGPU_IMGSZ,
            })

            if (!runningRef.current || sessionId !== sessionIdRef.current) return
            if (requestId < appliedSeqRef.current) return

            const { response, timing, diagnostics: inferenceDiagnostics } = result
            appliedSeqRef.current = requestId
            completedRef.current += 1
            latestInferenceMsRef.current = timing.inferenceMs
            latestTotalMsRef.current = timing.totalMs
            inferenceSamplesRef.current.push(timing.inferenceMs)
            totalSamplesRef.current.push(timing.totalMs)
            sessionIdentityRef.current = inferenceDiagnostics.sessionIdentity
            const debugStats = publishDebugStats()

            setLastRequestId(requestId)
            setResult(response)
            setErrorMessage(null)

            const now = performance.now()
            completionTimestampsRef.current.push(now)
            completionTimestampsRef.current =
              completionTimestampsRef.current.filter(
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
            const sourceWidth = video.videoWidth
            const sourceHeight = video.videoHeight

            setLiveStats({
              people: response.people,
              inferenceMs: timing.inferenceMs,
              requestTotalMs: timing.totalMs,
              detectionFps,
              cameraFps: cameraFpsRef.current,
            })

            const metrics: LiveFrameMetrics = {
              requestId,
              sessionId,
              timestamp: new Date().toISOString(),
              sourceWidth,
              sourceHeight,
              captureWidth: inferenceDiagnostics.letterbox.canvasWidth,
              captureHeight: inferenceDiagnostics.letterbox.canvasHeight,
              jpegQuality: jpegQualityRef.current,
              payloadBytes: 0,
              payloadKb: 0,
              captureMs: timing.preprocessMs,
              encodeMs: 0,
              requestTotalMs: timing.totalMs,
              serverInferenceMs: timing.inferenceMs,
              nonInferenceOverheadMs:
                timing.preprocessMs + timing.postprocessMs,
              parseMs: timing.postprocessMs,
              peopleReturned: response.people,
              visibleDetections: visible,
              averageConfidence: avgConf,
              modelImgsz: LIVE_WEBGPU_IMGSZ,
              detectionFps,
              cameraFps: cameraFpsRef.current,
              activeRequests: 1,
              confidenceThreshold: confidenceRef.current,
            }

            setDiagnostics({
              cameraWidth: sourceWidth,
              cameraHeight: sourceHeight,
              captureWidth: inferenceDiagnostics.letterbox.canvasWidth,
              captureHeight: inferenceDiagnostics.letterbox.canvasHeight,
              payloadKb: null,
              captureMs: timing.preprocessMs,
              encodeMs: null,
              requestTotalMs: timing.totalMs,
              serverInferenceMs: timing.inferenceMs,
              nonInferenceOverheadMs:
                timing.preprocessMs + timing.postprocessMs,
              detectionFps,
              cameraFps: cameraFpsRef.current,
              visibleDetections: visible,
              returnedDetections: response.people,
              confidenceThreshold: confidenceRef.current,
              activeRequests: 1,
              currentCaptureWidthSetting: captureWidthRef.current,
              currentJpegQuality: jpegQualityRef.current,
              modelImgsz: LIVE_WEBGPU_IMGSZ,
              completedInferences: debugStats.completed,
              droppedFrames: debugStats.dropped,
              latestInferenceMs: debugStats.latestInferenceMs,
              medianInferenceMs: debugStats.medianInferenceMs,
              medianTotalMs: debugStats.medianTotalMs,
              effectiveInferenceFps: debugStats.effectiveFps,
              maxConcurrentInference: debugStats.maxConcurrentInference,
            })

            if (recordingRef.current) {
              setSessionRecords((prev) => [...prev, metrics])
            }

            if (import.meta.env.DEV) {
              console.debug('[live] webgpu detect complete', {
                requestId,
                activeRequests: activeRequestsRef.current,
                imgsz: LIVE_WEBGPU_IMGSZ,
                tensor: `${inferenceDiagnostics.letterbox.canvasWidth}x${inferenceDiagnostics.letterbox.canvasHeight}`,
                inferenceMs: timing.inferenceMs.toFixed(0),
                totalMs: timing.totalMs.toFixed(0),
                completed: debugStats.completed,
                dropped: debugStats.dropped,
                sessionIdentity: inferenceDiagnostics.sessionIdentity,
              })
            }
          } catch (error) {
            if (!runningRef.current || sessionId !== sessionIdRef.current) return
            const message =
              error instanceof Error ? error.message : 'Live detection failed.'
            setErrorMessage(message)
          } finally {
            if (releaseSlot) {
              endWebGpuInference()
              releaseSlot = false
            }
            if (sessionId === sessionIdRef.current) {
              setInFlight(false)
              setIsDetecting(false)
              setDiagnostics((prev) => ({ ...prev, activeRequests: 0 }))
            } else {
              inFlightRef.current = false
              activeRequestsRef.current = 0
            }
          }
        })()
      }

      stopDetectionLoop()
      schedule()
    },
    [publishDebugStats, setInFlight, stopDetectionLoop],
  )

  const startCamera = useCallback(
    async (deviceId?: string) => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraState('unavailable')
        setErrorMessage('Camera APIs are not available in this browser.')
        return
      }

      const webGpuStatus = getWebGpuRuntimeSnapshot().status
      if (webGpuStatus !== 'ready') {
        stopCamera()
        setErrorMessage(
          webGpuStatus === 'loading'
            ? 'WebGPU runtime is still loading.'
            : WEBGPU_REQUIRED_MESSAGE,
        )
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
        liveStartedAtRef.current = performance.now()
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
        modelImgsz: LIVE_WEBGPU_IMGSZ,
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
      modelImgszNote: `Browser WebGPU live detection at imgsz=${LIVE_WEBGPU_IMGSZ}. Frames are not uploaded.`,
      records: sessionRecords,
      markers: sessionMarkers,
    }
  }, [sessionMarkers, sessionRecords])

  useEffect(() => {
    if (!enabled && runningRef.current) {
      stopCamera()
    }
  }, [enabled, stopCamera])

  useEffect(() => {
    return () => {
      runningRef.current = false
      stopDetectionLoop()
      abortRef.current?.abort()
      stopCameraFpsMonitor()
      stopTracks()
    }
  }, [stopCameraFpsMonitor, stopDetectionLoop, stopTracks])

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
    startCamera,
    stopCamera,
    switchCamera,
    addQualityMarker,
    clearSessionData,
    exportSession,
  }
}

// Re-export defaults for panel convenience
export { DEFAULT_LIVE_CAPTURE_WIDTH, DEFAULT_LIVE_JPEG_QUALITY }
