import type { BenchmarkConfigSummary, BenchmarkSample } from '../lib/liveTypes'
import { DEPLOYED_MODEL_IMGSZ } from '../lib/liveConfig'
import { exportBenchmarkJson } from '../lib/telemetryExport'

interface LiveBenchmarkProps {
  running: boolean
  progress: string | null
  summaries: BenchmarkConfigSummary[]
  samples: BenchmarkSample[]
  cameraActive: boolean
  onRun: () => void
}

export function LiveBenchmark({
  running,
  progress,
  summaries,
  samples,
  cameraActive,
  onRun,
}: LiveBenchmarkProps) {
  return (
    <div className="live-benchmark">
      <h3 className="live-debug__heading">Live Benchmark</h3>
      <p className="live-debug__hint">
        Sequential capture-width benchmark only. Does not change YOLO{' '}
        <code>imgsz</code> (deployed model stays at {DEPLOYED_MODEL_IMGSZ}). Never
        runs configurations concurrently. 5 samples per width by default.
      </p>

      <div className="action-bar">
        <button
          type="button"
          className="btn btn--primary"
          disabled={running || !cameraActive}
          onClick={onRun}
        >
          {running ? 'Benchmark running…' : 'Run Live Benchmark'}
        </button>
        {summaries.length > 0 && (
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() =>
              exportBenchmarkJson(
                summaries,
                samples,
                new Date().toISOString(),
              )
            }
          >
            Export Benchmark JSON
          </button>
        )}
      </div>

      {progress && <p className="live-debug__progress">{progress}</p>}

      {summaries.length > 0 && (
        <div className="benchmark-table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Capture</th>
                <th>Model imgsz</th>
                <th>Payload</th>
                <th>Total ms</th>
                <th>Server ms</th>
                <th>Overhead</th>
                <th>Det. FPS</th>
                <th>People</th>
                <th>Avg conf</th>
              </tr>
            </thead>
            <tbody>
              {summaries.map((row) => (
                <tr key={row.captureWidth}>
                  <td>{row.captureWidth}px</td>
                  <td>{row.modelImgsz ?? DEPLOYED_MODEL_IMGSZ}</td>
                  <td>{row.avgPayloadKb.toFixed(1)} KB</td>
                  <td>{row.avgRequestTotalMs.toFixed(0)}</td>
                  <td>{row.avgServerInferenceMs.toFixed(0)}</td>
                  <td>{row.avgNonInferenceOverheadMs.toFixed(0)}</td>
                  <td>{row.effectiveDetectionFps.toFixed(2)}</td>
                  <td>{row.avgPeopleReturned.toFixed(1)}</td>
                  <td>
                    {row.avgConfidence === null
                      ? '—'
                      : `${(row.avgConfidence * 100).toFixed(1)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
