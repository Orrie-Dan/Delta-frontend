import type { LiveDiagnosticsSnapshot } from '../lib/liveTypes'
import { DEPLOYED_MODEL_IMGSZ, PERFORMANCE_TARGETS } from '../lib/liveConfig'

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
            Capture resolution is the JPEG sent to the API. Model inference
            resolution is currently fixed server-side at{' '}
            <strong>imgsz={DEPLOYED_MODEL_IMGSZ}</strong> and is not changed by
            capture width.
          </p>

          <dl className="diag-grid">
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
              <dt>API round trip</dt>
              <dd>{fmt(diagnostics.requestTotalMs, 0, ' ms')}</dd>
            </div>
            <div>
              <dt>Server inference</dt>
              <dd>{fmt(diagnostics.serverInferenceMs, 0, ' ms')}</dd>
            </div>
            <div>
              <dt>Non-inference overhead (approx.)</dt>
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
              <dt>Requests in flight</dt>
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
              <dt>Capture width setting</dt>
              <dd>{diagnostics.currentCaptureWidthSetting}px</dd>
            </div>
            <div>
              <dt>JPEG quality</dt>
              <dd>{diagnostics.currentJpegQuality.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Model imgsz</dt>
              <dd>{diagnostics.modelImgsz ?? DEPLOYED_MODEL_IMGSZ}</dd>
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
