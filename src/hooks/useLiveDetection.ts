import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { ApiError, detectPeople } from '../api/detection'
import { captureVideoFrame } from '../lib/captureFrame'
import type {
  CameraLifecycle,
  DetectionResponse,
  LiveDetectionStats,
} from '../types/detection'

const FPS_WINDOW_MS = 12_000
const LOOP_GAP_MS = 40

interface UseLiveDetectionOptions {
  enabled: boolean
}

interface UseLiveDetectionResult {
  videoRef: RefObject<HTMLVideoElement | null>
  cameraState: CameraLifecycle
  isDetecting: boolean
  result: DetectionResponse | null
  errorMessage: string | null
  liveStats: LiveDetectionStats
  videoDevices: MediaDeviceInfo[]
  activeDeviceId: string | null
  startCamera: (deviceId?: string) => Promise<void>
  stopCamera: () => void
  switchCamera: () => Promise<void>
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
  const recent = timestamps.filter((ts) => now - ts <= FPS_WINDOW_MS)
  if (recent.length < 2) {
    if (recent.length === 1) return null
    return null
  }
  const elapsedSec = (recent[recent.length - 1]! - recent[0]!) / 1000
  if (elapsedSec <= 0) return null
  return (recent.length - 1) / elapsedSec
}

export function useLiveDetection({
  enabled,
}: UseLiveDetectionOptions): UseLiveDetectionResult {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const runningRef = useRef(false)
  const inFlightRef = useRef(false)
  const sessionIdRef = useRef(0)
  const requestSeqRef = useRef(0)
  const appliedSeqRef = useRef(0)
  const abortRef = useRef<AbortController | null>(null)
  const completionTimestampsRef = useRef<number[]>([])
  const loopTimerRef = useRef<number | null>(null)

  const [cameraState, setCameraState] = useState<CameraLifecycle>('off')
  const [isDetecting, setIsDetecting] = useState(false)
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [liveStats, setLiveStats] = useState<LiveDetectionStats>({
    people: 0,
    inferenceMs: null,
    detectionFps: null,
  })
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([])
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null)

  const clearLoopTimer = useCallback(() => {
    if (loopTimerRef.current !== null) {
      window.clearTimeout(loopTimerRef.current)
      loopTimerRef.current = null
    }
  }, [])

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

  const stopCamera = useCallback(() => {
    runningRef.current = false
    clearLoopTimer()
    abortRef.current?.abort()
    abortRef.current = null
    inFlightRef.current = false
    sessionIdRef.current += 1
    requestSeqRef.current = 0
    appliedSeqRef.current = 0
    completionTimestampsRef.current = []
    stopTracks()
    setIsDetecting(false)
    setResult(null)
    setErrorMessage(null)
    setLiveStats({ people: 0, inferenceMs: null, detectionFps: null })
    setCameraState('off')
    setActiveDeviceId(null)
  }, [clearLoopTimer, stopTracks])

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

  const runDetectionLoop = useCallback(async (sessionId: number) => {
    if (!runningRef.current || sessionId !== sessionIdRef.current) return
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

    inFlightRef.current = true
    setIsDetecting(true)
    const requestId = ++requestSeqRef.current
    const abortController = new AbortController()
    abortRef.current = abortController

    try {
      const blob = await captureVideoFrame(video, canvasRef.current)
      if (!runningRef.current || sessionId !== sessionIdRef.current) return

      const response = await detectPeople(blob, {
        filename: 'frame.jpg',
        signal: abortController.signal,
      })

      if (!runningRef.current || sessionId !== sessionIdRef.current) return
      if (requestId < appliedSeqRef.current) return

      appliedSeqRef.current = requestId
      setResult(response)
      setErrorMessage(null)

      const now = performance.now()
      completionTimestampsRef.current.push(now)
      completionTimestampsRef.current = completionTimestampsRef.current.filter(
        (ts) => now - ts <= FPS_WINDOW_MS,
      )

      setLiveStats({
        people: response.people,
        inferenceMs: response.inference_ms,
        detectionFps: computeDetectionFps(completionTimestampsRef.current, now),
      })
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
      inFlightRef.current = false
      if (sessionId === sessionIdRef.current) {
        setIsDetecting(false)
      }

      if (runningRef.current && sessionId === sessionIdRef.current) {
        clearLoopTimer()
        loopTimerRef.current = window.setTimeout(() => {
          void runDetectionLoop(sessionId)
        }, LOOP_GAP_MS)
      }
    }
  }, [clearLoopTimer])

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

        setCameraState('active')
        setResult(null)
        setLiveStats({ people: 0, inferenceMs: null, detectionFps: null })

        void runDetectionLoop(sessionId)
      } catch (error) {
        stopTracks()
        const classified = classifyCameraError(error)
        setCameraState(classified.state)
        setErrorMessage(classified.message)
      }
    },
    [refreshDevices, runDetectionLoop, stopCamera, stopTracks],
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
      stopTracks()
    }
  }, [clearLoopTimer, stopTracks])

  return {
    videoRef,
    cameraState,
    isDetecting,
    result,
    errorMessage,
    liveStats,
    videoDevices,
    activeDeviceId,
    startCamera,
    stopCamera,
    switchCamera,
  }
}
