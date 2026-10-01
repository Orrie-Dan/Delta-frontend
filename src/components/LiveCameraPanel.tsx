import { useEffect, useMemo, useState } from 'react'
import { useDebugMode } from '../hooks/useDebugMode'
import { useLiveDetection } from '../hooks/useLiveDetection'
import {
  DEFAULT_LIVE_CAPTURE_WIDTH,
  DEFAULT_LIVE_JPEG_QUALITY,
  LIVE_CAPTURE_WIDTHS,
  LIVE_JPEG_QUALITIES,
  type LiveCaptureWidth,
  type LiveJpegQuality,
  type TestScenario,
} from '../lib/liveConfig'
import { getObjectFitContainRect } from '../lib/overlayLayout'
import type { ApiStatus, Detection } from '../types/detection'
import { ConfidenceFilter } from './ConfidenceFilter'
import { LiveBenchmark } from './LiveBenchmark'
import { LiveDiagnostics } from './LiveDiagnostics'
import { LiveTestSession } from './LiveTestSession'

interface LiveCameraPanelProps {
  apiStatus: ApiStatus
  confidenceThreshold: number
  onConfidenceChange: (value: number) => void
  active: boolean
}

function formatMs(ms: number | null): string {
  if (ms === null) return '—'
  return `${(ms / 1000).toFixed(1)}s`
}

function formatFps(fps: number | null): string {
  if (fps === null) return '—'
  return `${fps.toFixed(2)}`
}

export function LiveCameraPanel({
  apiStatus,
  confidenceThreshold,
  onConfidenceChange,
  active,
}: LiveCameraPanelProps) {
  const debug = useDebugMode()
  const [captureWidth, setCaptureWidth] = useState<LiveCaptureWidth>(
    DEFAULT_LIVE_CAPTURE_WIDTH,
  )
  const [jpegQuality, setJpegQuality] = useState<LiveJpegQuality>(
    DEFAULT_LIVE_JPEG_QUALITY,
  )
  const [scenario, setScenario] = useState<TestScenario>('single_near')
  const [recording, setRecording] = useState(false)
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(true)
  const [overlayRect, setOverlayRect] = useState({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
  })
  const [videoIntrinsic, setVideoIntrinsic] = useState({ w: 0, h: 0 })

  const live = useLiveDetection({
    enabled: active,
    captureWidth,
    jpegQuality,
    confidenceThreshold,
    scenario,
    recording,
  })

  const {
    videoRef,
    cameraState,
    isDetecting,
    result,
    errorMessage,
    liveStats,
    diagnostics,
    videoDevices,
    startCamera,
    stopCamera,
    switchCamera,
    addQualityMarker,
    clearSessionData,
    exportSession,
    runCaptureWidthBenchmark,
    benchmarkRunning,
    benchmarkProgress,
    benchmarkSummaries,
    benchmarkSamples,
    sessionRecords,
    sessionMarkers,
    activeRequests,
  } = live

  const filteredDetections = useMemo(() => {
    if (!result) return [] as Detection[]
    return result.detections.filter((d) => d.confidence >= confidenceThreshold)
  }, [result, confidenceThreshold])

  const filteredPeople = filteredDetections.length
  const isLive = cameraState === 'active'
  const showOverlay = Boolean(result && isLive && overlayRect.width > 0)
  const canSwitchCamera = videoDevices.length > 1 && isLive

  const statusHeadline =
    cameraState === 'requesting'
      ? 'Requesting camera permission…'
      : cameraState === 'permission_denied'
        ? 'Camera permission denied'
        : cameraState === 'unavailable'
          ? 'Camera unavailable'
          : apiStatus === 'offline' && isLive
            ? 'API unavailable'
            : isDetecting
              ? 'Detecting…'
              : isLive
                ? 'Camera active'
                : 'Camera off'

  useEffect(() => {
    const video = videoRef.current
    if (!video || !isLive) return

    const updateLayout = () => {
      const w = video.videoWidth
      const h = video.videoHeight
      if (w && h) setVideoIntrinsic({ w, h })
      const rect = getObjectFitContainRect(
        w || 1,
        h || 1,
        video.clientWidth,
        video.clientHeight,
      )
      setOverlayRect(rect)
    }

    updateLayout()
    video.addEventListener('loadedmetadata', updateLayout)
    const observer = new ResizeObserver(updateLayout)
    observer.observe(video)
    window.addEventListener('orientationchange', updateLayout)

    return () => {
      video.removeEventListener('loadedmetadata', updateLayout)
      observer.disconnect()
      window.removeEventListener('orientationchange', updateLayout)
    }
  }, [isLive, videoRef, result])

  const frameAspectStyle =
    videoIntrinsic.w > 0 && videoIntrinsic.h > 0
      ? { aspectRatio: `${videoIntrinsic.w} / ${videoIntrinsic.h}` }
      : undefined

  return (
    <section className="panel panel--live" aria-label="Live camera detection">
      <div className="live-toolbar">
        <ConfidenceFilter
          value={confidenceThreshold}
          onChange={onConfidenceChange}
          disabled={!result}
        />
      </div>

      {debug && (
        <div className="live-debug-controls">
          <label className="live-field">
            <span>Capture width (network frame)</span>
            <select
              value={captureWidth}
              onChange={(event) =>
                setCaptureWidth(Number(event.target.value) as LiveCaptureWidth)
              }
            >
              {LIVE_CAPTURE_WIDTHS.map((width) => (
                <option key={width} value={width}>
                  {width}px
                </option>
              ))}
            </select>
          </label>
          <label className="live-field">
            <span>JPEG quality</span>
            <select
              value={jpegQuality}
              onChange={(event) =>
                setJpegQuality(Number(event.target.value) as LiveJpegQuality)
              }
            >
              {LIVE_JPEG_QUALITIES.map((quality) => (
                <option key={quality} value={quality}>
                  {quality.toFixed(2)}
                </option>
              ))}
            </select>
          </label>
          <p className="live-debug__hint">
            Capture width ≠ YOLO imgsz. Deployed model imgsz remains 1280.
            In flight: {activeRequests}
          </p>
        </div>
      )}

      <div className="live-viewer">
        <div
          className={`live-viewer__frame${isLive ? '' : ' live-viewer__frame--idle'}`}
          style={isLive ? frameAspectStyle : undefined}
        >
          <video
            ref={videoRef}
            className="live-viewer__video"
            playsInline
            muted
            autoPlay
            aria-label="Live camera preview"
          />

          {showOverlay && result && (
            <svg
              className="live-viewer__overlay"
              viewBox={`0 0 ${result.image.width} ${result.image.height}`}
              preserveAspectRatio="none"
              aria-hidden="true"
              style={{
                left: overlayRect.x,
                top: overlayRect.y,
                width: overlayRect.width,
                height: overlayRect.height,
              }}
            >
              {filteredDetections.map((detection, index) => {
                const width = Math.max(detection.x2 - detection.x1, 1)
                const height = Math.max(detection.y2 - detection.y1, 1)
                return (
                  <rect
                    key={`${detection.x1}-${detection.y1}-${index}`}
                    x={detection.x1}
                    y={detection.y1}
                    width={width}
                    height={height}
                    fill="rgba(56, 189, 248, 0.12)"
                    stroke="#38bdf8"
                    strokeWidth={Math.max(
                      1.5,
                      Math.min(result.image.width, result.image.height) * 0.0025,
                    )}
                    vectorEffect="non-scaling-stroke"
                  >
                    <title>{`${(detection.confidence * 100).toFixed(1)}%`}</title>
                  </rect>
                )
              })}
            </svg>
          )}

          <div className="live-hud" aria-live="polite">
            <div className="live-hud__top">
              <span className={`live-hud__badge${isLive ? ' live-hud__badge--on' : ''}`}>
                {isLive ? 'LIVE' : statusHeadline}
              </span>
              <span className="live-hud__people">
                {isLive ? filteredPeople : '—'}
              </span>
            </div>
            <div className="live-hud__bottom">
              <span>
                Detection {formatMs(liveStats.requestTotalMs ?? liveStats.inferenceMs)}
              </span>
              <span aria-hidden="true">•</span>
              <span>{formatFps(liveStats.detectionFps)} detection FPS</span>
              {debug && (
                <>
                  <span aria-hidden="true">•</span>
                  <span>~{formatFps(liveStats.cameraFps)} camera FPS</span>
                </>
              )}
            </div>
          </div>

          {!isLive && (
            <div className="live-viewer__placeholder">
              <p className="live-viewer__placeholder-title">Live person detection</p>
              <p className="live-viewer__placeholder-text">
                Start the camera to capture frames and run YOLO26s detection.
                The preview stays smooth while inference runs in the background.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="action-bar live-actions">
        {isLive ? (
          <button type="button" className="btn btn--primary" onClick={stopCamera}>
            Stop Camera
          </button>
        ) : (
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => void startCamera()}
            disabled={cameraState === 'requesting'}
          >
            {cameraState === 'requesting' ? 'Starting…' : 'Start Camera'}
          </button>
        )}

        {canSwitchCamera && (
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() => void switchCamera()}
          >
            Switch Camera
          </button>
        )}
      </div>

      {errorMessage && (
        <p className="inline-warning" role="alert">
          {errorMessage}
        </p>
      )}

      {apiStatus === 'offline' && isLive && (
        <p className="inline-warning" role="status">
          Model API appears offline. Camera preview continues; detection will
          resume when the service responds.
        </p>
      )}

      {cameraState === 'permission_denied' && (
        <p className="live-help">
          Tip: use HTTPS (or localhost) and allow camera access when prompted.
        </p>
      )}

      {debug && (
        <div className="live-debug-stack">
          <LiveDiagnostics
            diagnostics={diagnostics}
            open={diagnosticsOpen}
            onToggle={() => setDiagnosticsOpen((open) => !open)}
          />

          <LiveTestSession
            scenario={scenario}
            onScenarioChange={setScenario}
            recording={recording}
            onStart={() => {
              clearSessionData()
              setRecording(true)
            }}
            onStop={() => setRecording(false)}
            recordCount={sessionRecords.length}
            markerCount={sessionMarkers.length}
            markers={sessionMarkers}
            onMarker={addQualityMarker}
            onExport={exportSession}
            onClear={() => {
              setRecording(false)
              clearSessionData()
            }}
            cameraActive={isLive}
          />

          <LiveBenchmark
            running={benchmarkRunning}
            progress={benchmarkProgress}
            summaries={benchmarkSummaries}
            samples={benchmarkSamples}
            cameraActive={isLive}
            onRun={() => void runCaptureWidthBenchmark()}
          />
        </div>
      )}
    </section>
  )
}
