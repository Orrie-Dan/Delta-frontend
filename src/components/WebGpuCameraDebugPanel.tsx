import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  WebGpuCameraProbeProgress,
  WebGpuCameraProbeReport,
} from '../lib/onnxWebGpuCameraProbe'

const DEFAULT_PHASE_DURATION_MS = 30_000
const MIN_PHASE_DURATION_MS = 15_000
const MAX_PHASE_DURATION_MS = 60_000

function formatMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return '—'
  return `${ms.toFixed(1)} ms`
}

function formatFps(fps: number | null | undefined): string {
  if (fps == null || Number.isNaN(fps)) return '—'
  return `${fps.toFixed(2)} fps`
}

function formatNum(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

type CameraState = 'off' | 'requesting' | 'active' | 'error'

export function WebGpuCameraDebugPanel() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const [cameraState, setCameraState] = useState<CameraState>('off')
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [phaseDurationSec, setPhaseDurationSec] = useState(
    DEFAULT_PHASE_DURATION_MS / 1000,
  )
  const [probing, setProbing] = useState(false)
  const [progress, setProgress] = useState<WebGpuCameraProbeProgress | null>(
    null,
  )
  const [probeError, setProbeError] = useState<string | null>(null)
  const [report, setReport] = useState<WebGpuCameraProbeReport | null>(null)

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    const video = videoRef.current
    if (video) video.srcObject = null
    setCameraState('off')
  }, [])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error')
      setCameraError('Camera APIs are not available in this browser.')
      return
    }
    stopStream()
    setCameraState('requesting')
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })
      streamRef.current = stream
      const video = videoRef.current
      if (!video) {
        stream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        setCameraState('error')
        setCameraError('Camera preview element is not ready.')
        return
      }
      video.srcObject = stream
      video.muted = true
      video.playsInline = true
      await video.play()
      setCameraState('active')
    } catch (error) {
      stopStream()
      setCameraState('error')
      setCameraError(
        error instanceof Error ? error.message : 'Unable to open the camera.',
      )
    }
  }, [stopStream])

  const runProbe = useCallback(async () => {
    if (probing) return
    const video = videoRef.current
    if (!video || cameraState !== 'active') {
      setProbeError('Start the camera before running the WebGPU probe.')
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setProbing(true)
    setProbeError(null)
    setProgress(null)
    setReport(null)

    try {
      const { runWebGpuCameraProbe } = await import(
        '../lib/onnxWebGpuCameraProbe'
      )
      const next = await runWebGpuCameraProbe({
        video,
        phaseDurationMs: Math.round(phaseDurationSec * 1000),
        onProgress: setProgress,
        signal: controller.signal,
      })
      setReport(next)
      if (next.error) setProbeError(next.error)
    } catch (err) {
      setProbeError(
        err instanceof Error ? err.message : 'WebGPU camera probe failed.',
      )
    } finally {
      setProbing(false)
      if (abortRef.current === controller) abortRef.current = null
    }
  }, [cameraState, phaseDurationSec, probing])

  return (
    <section
      className="panel panel--onnx-test panel--webgpu-camera"
      aria-label="Debug WebGPU camera probe"
    >
      <div className="onnx-test__header">
        <div>
          <h3 className="onnx-test__title">WebGPU camera probe</h3>
          <p className="onnx-test__subtitle">
            Debug-only loop on the shared production WebGPU ONNX session. Runs
            imgsz 1280 then 960 with one inference at a time and drops frames
            while busy. Production live camera uses this session at imgsz 960.
          </p>
        </div>
        <div className="action-bar action-bar--compact">
          {cameraState === 'active' ? (
            <button
              type="button"
              className="btn btn--ghost"
              onClick={stopStream}
              disabled={probing}
            >
              Stop camera
            </button>
          ) : (
            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => void startCamera()}
              disabled={cameraState === 'requesting' || probing}
            >
              {cameraState === 'requesting' ? 'Starting…' : 'Start camera'}
            </button>
          )}
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => void runProbe()}
            disabled={probing || cameraState !== 'active'}
          >
            {probing ? 'Probe running…' : 'Run WebGPU probe (1280 → 960)'}
          </button>
        </div>
      </div>

      {(cameraError || probeError) && (
        <p className="inline-warning" role="alert">
          {cameraError ?? probeError}
        </p>
      )}

      <label className="live-field webgpu-probe__duration">
        <span>Phase duration</span>
        <input
          type="range"
          min={MIN_PHASE_DURATION_MS / 1000}
          max={MAX_PHASE_DURATION_MS / 1000}
          step={1}
          value={phaseDurationSec}
          disabled={probing}
          onChange={(event) => setPhaseDurationSec(Number(event.target.value))}
        />
        <span className="webgpu-probe__duration-value">{phaseDurationSec}s</span>
      </label>

      {progress && (
        <p className="live-debug__progress" role="status">
          {progress.message} · completed {progress.framesCompleted} · dropped{' '}
          {progress.framesDropped}
        </p>
      )}

      <div className="webgpu-probe__preview">
        <video
          ref={videoRef}
          className="webgpu-probe__video"
          playsInline
          muted
          aria-label="WebGPU probe camera preview"
        />
      </div>

      {report && report.phases.length > 0 && (
        <>
          <h4 className="onnx-bench__heading">WebGPU camera probe results</h4>
          <p className="live-debug__hint">
            Session {report.sessionIdentity}. Adapter{' '}
            {report.adapterInfo ?? '—'}. Wall {formatMs(report.totalWallMs)}.
          </p>
          <div className="benchmark-table-wrap">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>imgsz</th>
                  <th>Completed</th>
                  <th>Dropped</th>
                  <th>E2E FPS</th>
                  <th>Inf. FPS</th>
                  <th>Median total</th>
                  <th>p95 total</th>
                  <th>Median run</th>
                </tr>
              </thead>
              <tbody>
                {report.phases.map((phase) => (
                  <tr key={phase.imgsz}>
                    <td>{phase.imgsz}</td>
                    <td>{phase.framesCompleted}</td>
                    <td>{phase.framesDropped}</td>
                    <td>{formatFps(phase.endToEndFps)}</td>
                    <td>{formatFps(phase.inferenceFpsFromMedianRun)}</td>
                    <td>{formatMs(phase.medianTotalMs)}</td>
                    <td>{formatMs(phase.p95TotalMs)}</td>
                    <td>{formatMs(phase.medianInferenceMs)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {report.phases.map((phase) => (
            <p key={`det-${phase.imgsz}`} className="live-debug__hint">
              imgsz {phase.imgsz}: detections min/median/max{' '}
              {formatNum(phase.detectionMin, 0)}/
              {formatNum(phase.detectionMedian, 0)}/
              {formatNum(phase.detectionMax, 0)}; unique counts{' '}
              {phase.uniqueDetectionCounts}; long tasks {phase.longTaskCount} (
              {formatMs(phase.longTaskTotalMs)})
            </p>
          ))}
        </>
      )}
    </section>
  )
}
