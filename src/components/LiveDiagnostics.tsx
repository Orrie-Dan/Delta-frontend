import type { LiveDiagnosticsSnapshot } from '../lib/liveTypes'
import { PERFORMANCE_TARGETS, RAILWAY_MODEL_IMGSZ } from '../lib/liveConfig'
import { LIVE_WEBGPU_IMGSZ } from '../lib/onnxWebGpu'

interface LiveDiagnosticsProps {
  diagnostics: LiveDiagnosticsSnapshot
  open: boolean
  onToggle: () => void
}

function fmt(value: number | null | undefined, digits = 1, suffix = ''): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value.toFixed(digits)}${suffix}`
}

function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return String(Math.round(value))
}

export function LiveDiagnostics({
  diagnostics,
  open,
  onToggle,
}: LiveDiagnosticsProps) {
  const isRailway = diagnostics.runtime === 'railway'
  const isWebGpu = diagnostics.runtime === 'webgpu'

  return (
    <div className="live-diagnostics">
      <button
        type="button"
        className="detection-details__toggle"
        aria-expanded={open}
        onClick={onToggle}
      >
        Live Diagnostics
        <span aria-hidden="true">{open ? '−' : '+'}</span>
      </button>

      {open && (
        <div className="live-diagnostics__panel">
          <p className="live-diagnostics__note">
            {isRailway ? (
              <>
                Cloud CPU fallback uploads sampled JPEG frames to Railway{' '}
                <code>/detect</code> at{' '}
                <strong>imgsz={RAILWAY_MODEL_IMGSZ}</strong>. Max one HTTP
                request in flight; busy frames are skipped.
              </>
            ) : isWebGpu ? (
              <>
                Live frames stay in the browser. WebGPU inference uses{' '}
                <strong>imgsz={LIVE_WEBGPU_IMGSZ}</strong> with one inference at
                a time. Extra camera frames are skipped.
              </>
            ) : (
              <>No live inference runtime selected.</>
            )}
          </p>

          <dl className="diag-grid">
            <div>
              <dt>Runtime</dt>
              <dd>
                {diagnostics.runtime === 'webgpu'
                  ? 'webgpu'
                  : diagnostics.runtime === 'railway'
                    ? 'railway'
                    : '—'}
              </dd>
            </div>
            <div>
              <dt>Camera</dt>
              <dd>
                {diagnostics.cameraWidth && diagnostics.cameraHeight
                  ? `${diagnostics.cameraWidth} × ${diagnostics.cameraHeight}`
                  : '—'}
              </dd>
            </div>
            <div>
              <dt>Captured frame</dt>
              <dd>
                {diagnostics.captureWidth && diagnostics.captureHeight
                  ? `${diagnostics.captureWidth} × ${diagnostics.captureHeight}`
                  : '—'}
              </dd>
            </div>
            <div>
              <dt>Payload</dt>
              <dd>{fmt(diagnostics.payloadKb, 1, ' KB')}</dd>
            </div>
            <div>
              <dt>Capture</dt>
              <dd>{fmt(diagnostics.captureMs, 1, ' ms')}</dd>
            </div>
            <div>
              <dt>JPEG encode</dt>
              <dd>{fmt(diagnostics.encodeMs, 1, ' ms')}</dd>
            </div>
            <div>
              <dt>
                {isRailway
                  ? 'Client round-trip'
                  : 'Total (preprocess + inference + post)'}
              </dt>
              <dd>
                {fmt(
                  isRailway
                    ? diagnostics.clientRoundTripMs ?? diagnostics.requestTotalMs
                    : diagnostics.requestTotalMs,
                  0,
                  ' ms',
                )}
              </dd>
            </div>
            <div>
              <dt>{isRailway ? 'Server inference' : 'Browser inference'}</dt>
              <dd>{fmt(diagnostics.serverInferenceMs, 0, ' ms')}</dd>
            </div>
            <div>
              <dt>
                {isRailway
                  ? 'Non-inference overhead'
                  : 'Preprocess + postprocess'}
              </dt>
              <dd>{fmt(diagnostics.nonInferenceOverheadMs, 0, ' ms')}</dd>
            </div>
            <div>
              <dt>Detection FPS</dt>
              <dd>{fmt(diagnostics.detectionFps, 2)}</dd>
            </div>
            <div>
              <dt>Camera FPS (approx.)</dt>
              <dd>{fmt(diagnostics.cameraFps, 0)}</dd>
            </div>
            <div>
              <dt>Visible detections</dt>
              <dd>{fmtInt(diagnostics.visibleDetections)}</dd>
            </div>
            <div>
              <dt>Returned detections</dt>
              <dd>{fmtInt(diagnostics.returnedDetections)}</dd>
            </div>
            <div>
              <dt>Confidence threshold</dt>
              <dd>{diagnostics.confidenceThreshold.toFixed(2)}</dd>
            </div>
            <div>
              <dt>
                {isRailway ? 'HTTP requests in flight' : 'Inferences in flight'}
              </dt>
              <dd
                className={
                  diagnostics.activeRequests > PERFORMANCE_TARGETS.maxInFlight
                    ? 'diag-bad'
                    : 'diag-ok'
                }
              >
                {diagnostics.activeRequests}
              </dd>
            </div>
            <div>
              <dt>Completed inferences</dt>
              <dd>{fmtInt(diagnostics.completedInferences)}</dd>
            </div>
            <div>
              <dt>Frames skipped while busy</dt>
              <dd>{fmtInt(diagnostics.droppedFrames)}</dd>
            </div>
            {isRailway && (
              <>
                <div>
                  <dt>Railway requests started</dt>
                  <dd>{fmtInt(diagnostics.railwayRequestsStarted)}</dd>
                </div>
                <div>
                  <dt>Railway requests completed</dt>
                  <dd>{fmtInt(diagnostics.railwayRequestsCompleted)}</dd>
                </div>
                <div>
                  <dt>Railway requests failed</dt>
                  <dd>{fmtInt(diagnostics.railwayRequestsFailed)}</dd>
                </div>
              </>
            )}
            <div>
              <dt>Latest inference</dt>
              <dd>{fmt(diagnostics.latestInferenceMs, 0, ' ms')}</dd>
            </div>
            <div>
              <dt>Median inference</dt>
              <dd>{fmt(diagnostics.medianInferenceMs, 0, ' ms')}</dd>
            </div>
            <div>
              <dt>Median total</dt>
              <dd>{fmt(diagnostics.medianTotalMs, 0, ' ms')}</dd>
            </div>
            <div>
              <dt>Effective inference FPS</dt>
              <dd>{fmt(diagnostics.effectiveInferenceFps, 2)}</dd>
            </div>
            <div>
              <dt>Max concurrent requests</dt>
              <dd
                className={
                  diagnostics.maxConcurrentInference >
                  PERFORMANCE_TARGETS.maxInFlight
                    ? 'diag-bad'
                    : 'diag-ok'
                }
              >
                {fmtInt(diagnostics.maxConcurrentInference)}
              </dd>
            </div>
            <div>
              <dt>Capture width setting</dt>
              <dd>{diagnostics.currentCaptureWidthSetting}px</dd>
            </div>
            <div>
              <dt>JPEG quality</dt>
              <dd>{diagnostics.currentJpegQuality.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Model imgsz</dt>
              <dd>
                {diagnostics.modelImgsz ??
                  (isRailway ? RAILWAY_MODEL_IMGSZ : LIVE_WEBGPU_IMGSZ)}
              </dd>
            </div>
          </dl>

          <div className="live-diagnostics__targets">
            <p className="live-diagnostics__targets-title">Targets (not claims)</p>
            <ul>
              <li>Camera preview ~{PERFORMANCE_TARGETS.cameraFps} FPS</li>
              <li>
                Detection update {PERFORMANCE_TARGETS.detectionFpsMin}–
                {PERFORMANCE_TARGETS.detectionFpsMax} FPS
              </li>
              <li>End-to-end latency &lt; {PERFORMANCE_TARGETS.totalLatencyMs} ms</li>
              <li>Requests in flight ≤ {PERFORMANCE_TARGETS.maxInFlight}</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
