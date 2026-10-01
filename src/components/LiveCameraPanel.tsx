import { useMemo } from 'react'
import { useLiveDetection } from '../hooks/useLiveDetection'
import type { ApiStatus, Detection } from '../types/detection'
import { ConfidenceFilter } from './ConfidenceFilter'

interface LiveCameraPanelProps {
  apiStatus: ApiStatus
  confidenceThreshold: number
  onConfidenceChange: (value: number) => void
  active: boolean
}

function formatInference(ms: number | null): string {
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
  const {
    videoRef,
    cameraState,
    isDetecting,
    result,
    errorMessage,
    liveStats,
    videoDevices,
    startCamera,
    stopCamera,
    switchCamera,
  } = useLiveDetection({
    enabled: active,
  })

  const filteredDetections = useMemo(() => {
    if (!result) return [] as Detection[]
    return result.detections.filter((d) => d.confidence >= confidenceThreshold)
  }, [result, confidenceThreshold])

  const filteredPeople = filteredDetections.length
  const isLive = cameraState === 'active'
  const showOverlay = Boolean(result && isLive)
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

  return (
    <section className="panel panel--live" aria-label="Live camera detection">
      <div className="live-toolbar">
        <ConfidenceFilter
          value={confidenceThreshold}
          onChange={onConfidenceChange}
          disabled={!result}
        />
      </div>

      <div className="live-viewer">
        <div className={`live-viewer__frame${isLive ? '' : ' live-viewer__frame--idle'}`}>
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
              <span>Inference {formatInference(liveStats.inferenceMs)}</span>
              <span aria-hidden="true">•</span>
              <span>Detection {formatFps(liveStats.detectionFps)} fps</span>
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
    </section>
  )
}
