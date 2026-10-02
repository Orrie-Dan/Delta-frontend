import { useCallback, useState } from 'react'
import { DetectionViewer } from './DetectionViewer'
import type { BenchmarkReport } from '../lib/onnxBenchmark'
import type { OnnxModelSnapshot } from '../lib/onnxModel'
import {
  TEST_IMAGE_URL,
  runCaltechOnnxTest,
  type OnnxInferenceResult,
} from '../lib/onnxInference'

interface BrowserOnnxTestPanelProps {
  browserModel: OnnxModelSnapshot
}

function formatMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return '—'
  return `${ms.toFixed(1)} ms`
}

function formatNum(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

function formatBox(
  box: { x1: number; y1: number; x2: number; y2: number } | null,
): string {
  if (!box) return '—'
  return `(${box.x1.toFixed(1)}, ${box.y1.toFixed(1)}) → (${box.x2.toFixed(1)}, ${box.y2.toFixed(1)})`
}

export function BrowserOnnxTestPanel({
  browserModel,
}: BrowserOnnxTestPanelProps) {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<OnnxInferenceResult | null>(null)
  const [runCount, setRunCount] = useState(0)
  const [sessionIds, setSessionIds] = useState<number[]>([])
  const [benchmarkRunning, setBenchmarkRunning] = useState(false)
  const [benchmarkProgress, setBenchmarkProgress] = useState<string | null>(
    null,
  )
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null)
  const [benchmarkReport, setBenchmarkReport] =
    useState<BenchmarkReport | null>(null)

  const handleRun = useCallback(async () => {
    if (running || benchmarkRunning) return
    setRunning(true)
    setError(null)

    try {
      const next = await runCaltechOnnxTest()
      setResult(next)
      setRunCount((count) => count + 1)
      setSessionIds((prev) => [...prev, next.diagnostics.sessionIdentity])
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Browser ONNX static test failed.',
      )
    } finally {
      setRunning(false)
    }
  }, [benchmarkRunning, running])

  const handleBenchmark = useCallback(async () => {
    if (running || benchmarkRunning) return
    setBenchmarkRunning(true)
    setBenchmarkError(null)
    setBenchmarkProgress('Starting benchmark')

    try {
      const { runBrowserBenchmark } = await import('../lib/onnxBenchmark')
      const report = await runBrowserBenchmark((message) => {
        setBenchmarkProgress(message)
      })
      setBenchmarkReport(report)
    } catch (err) {
      setBenchmarkError(
        err instanceof Error ? err.message : 'Browser benchmark failed.',
      )
    } finally {
      setBenchmarkRunning(false)
    }
  }, [benchmarkRunning, running])

  const uniqueSessionIds = Array.from(new Set(sessionIds))
  const canRun = browserModel.status === 'ready' && !running && !benchmarkRunning

  return (
    <section
      className="panel panel--onnx-test"
      aria-label="Temporary browser ONNX diagnostic"
    >
      <div className="onnx-test__header">
        <div>
          <h3 className="onnx-test__title">Temporary browser ONNX test</h3>
          <p className="onnx-test__subtitle">
            Isolated static-image correctness check on Caltech
            set00_V001_0398.png. Does not replace upload or live detection.
          </p>
        </div>
        <div className="onnx-test__actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => void handleRun()}
            disabled={!canRun}
          >
            {running
              ? 'Running…'
              : browserModel.status === 'loading'
                ? 'Waiting for model…'
                : browserModel.status === 'error'
                  ? 'Model unavailable'
                  : 'Run browser ONNX test'}
          </button>
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() => void handleBenchmark()}
            disabled={!canRun}
          >
            {benchmarkRunning ? 'Benchmark running…' : 'Run browser benchmark'}
          </button>
        </div>
      </div>

      {benchmarkProgress && (
        <p className="onnx-bench__status" role="status">
          {benchmarkProgress}
        </p>
      )}

      {benchmarkError && (
        <p className="inline-warning" role="alert">
          {benchmarkError}
        </p>
      )}

      {benchmarkReport && (
        <BenchmarkResults report={benchmarkReport} />
      )}

      {error && (
        <p className="inline-warning" role="alert">
          {error}
        </p>
      )}

      {result && (
        <div className="onnx-test__body">
          <dl className="onnx-test__metrics">
            <div>
              <dt>Model status</dt>
              <dd>{browserModel.status}</dd>
            </div>
            <div>
              <dt>Original size</dt>
              <dd>
                {result.diagnostics.letterbox.originalWidth} ×{' '}
                {result.diagnostics.letterbox.originalHeight}
              </dd>
            </div>
            <div>
              <dt>Letterbox</dt>
              <dd>
                r={result.diagnostics.letterbox.ratio.toFixed(5)} · resized{' '}
                {result.diagnostics.letterbox.resizedWidth}×
                {result.diagnostics.letterbox.resizedHeight} · canvas{' '}
                {result.diagnostics.letterbox.canvasWidth}×
                {result.diagnostics.letterbox.canvasHeight} · pad (
                {result.diagnostics.letterbox.padX},{' '}
                {result.diagnostics.letterbox.padY})
              </dd>
            </div>
            <div>
              <dt>Tensor</dt>
              <dd>
                [{result.diagnostics.preprocess.tensorShape.join(', ')}] · min{' '}
                {result.diagnostics.preprocess.tensorMin.toFixed(3)} · max{' '}
                {result.diagnostics.preprocess.tensorMax.toFixed(3)}
              </dd>
            </div>
            <div>
              <dt>Preprocess</dt>
              <dd>{formatMs(result.timing.preprocessMs)}</dd>
            </div>
            <div>
              <dt>Inference</dt>
              <dd>{formatMs(result.timing.inferenceMs)}</dd>
            </div>
            <div>
              <dt>Postprocess</dt>
              <dd>{formatMs(result.timing.postprocessMs)}</dd>
            </div>
            <div>
              <dt>Total</dt>
              <dd>{formatMs(result.timing.totalMs)}</dd>
            </div>
            <div>
              <dt>Output</dt>
              <dd>
                {result.diagnostics.outputName} · [
                {result.diagnostics.outputDims.join(', ')}] ·{' '}
                {result.diagnostics.anchorCount} anchors
              </dd>
            </div>
            <div>
              <dt>Candidates ≥ 0.25</dt>
              <dd>{result.diagnostics.candidatesBeforeNms}</dd>
            </div>
            <div>
              <dt>Final detections</dt>
              <dd>{result.response.people}</dd>
            </div>
            <div>
              <dt>Session reuse</dt>
              <dd>
                runs {runCount} · session id {result.diagnostics.sessionIdentity}{' '}
                · unique ids {uniqueSessionIds.join(', ') || '—'}
              </dd>
            </div>
          </dl>

          <ul className="onnx-test__detections">
            {result.response.detections.length === 0 ? (
              <li>No detections above threshold.</li>
            ) : (
              result.response.detections.map((detection, index) => (
                <li key={`${detection.x1}-${detection.y1}-${index}`}>
                  #{index + 1} conf={(detection.confidence * 100).toFixed(1)}% ·
                  ({detection.x1.toFixed(1)}, {detection.y1.toFixed(1)}) → (
                  {detection.x2.toFixed(1)}, {detection.y2.toFixed(1)})
                </li>
              ))
            )}
          </ul>

          <DetectionViewer
            imageUrl={TEST_IMAGE_URL}
            result={result.response}
            filteredDetections={result.response.detections}
            confidenceThreshold={0.25}
          />
        </div>
      )}
    </section>
  )
}

function BenchmarkResults({ report }: { report: BenchmarkReport }) {
  const { webgpu, stability } = report
  const webgpuLabel =
    webgpu.status === 'ready'
      ? 'ready'
      : webgpu.status === 'unsupported'
        ? 'UNSUPPORTED'
        : 'BLOCKED'

  return (
    <div className="onnx-bench">
      <h4 className="onnx-bench__heading">Browser benchmark</h4>
      <p className="onnx-bench__note">
        {report.environment.userAgent} · onnxruntime-web{' '}
        {report.environment.onnxruntimeWeb} · cores{' '}
        {report.environment.hardwareConcurrency}
        {report.environment.deviceMemoryGb != null
          ? ` · ${report.environment.deviceMemoryGb} GB`
          : ''}
        {report.environment.webGlRenderer
          ? ` · ${report.environment.webGlRenderer}`
          : ''}
        . Images:{' '}
        {report.images
          .map((image) => `${image.file} ${image.width}×${image.height}`)
          .join(' · ')}
        . Confidence 0.25, NMS IoU 0.6, max 1000. One warmup excluded from the
        medians.
      </p>

      <div className="benchmark-table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Runtime</th>
              <th>imgsz</th>
              <th>image</th>
              <th>tensor</th>
              <th>detections</th>
              <th>median inference</th>
              <th>median total</th>
            </tr>
          </thead>
          <tbody>
            {report.rows.map((row) => (
              <tr key={`${row.runtime}-${row.imgsz}-${row.imageId}`}>
                <td>{row.runtime}</td>
                <td>{row.imgsz}</td>
                <td>{row.imageLabel}</td>
                <td>
                  {row.tensorShape ? `[${row.tensorShape.join(', ')}]` : '—'}
                </td>
                <td>
                  {row.status === 'ok' ? row.detections : row.error}
                </td>
                <td>{formatMs(row.medianInferenceMs)}</td>
                <td>{formatMs(row.medianTotalMs)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 className="onnx-bench__heading">Timing detail</h4>
      <div className="benchmark-table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Runtime</th>
              <th>imgsz</th>
              <th>image</th>
              <th>anchors</th>
              <th>candidates</th>
              <th>preprocess</th>
              <th>session.run</th>
              <th>postprocess</th>
              <th>min/max total</th>
              <th>highest conf</th>
            </tr>
          </thead>
          <tbody>
            {report.rows.map((row) => (
              <tr key={`detail-${row.runtime}-${row.imgsz}-${row.imageId}`}>
                <td>{row.runtime}</td>
                <td>{row.imgsz}</td>
                <td>{row.imageLabel}</td>
                <td>{row.anchorCount ?? '—'}</td>
                <td>{row.candidatesBeforeNms ?? '—'}</td>
                <td>{formatMs(row.medianPreprocessMs)}</td>
                <td>{formatMs(row.medianInferenceMs)}</td>
                <td>{formatMs(row.medianPostprocessMs)}</td>
                <td>
                  {formatMs(row.minTotalMs)} / {formatMs(row.maxTotalMs)}
                </td>
                <td>{formatNum(row.highestConfidence, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 className="onnx-bench__heading">Caltech preservation</h4>
      <p className="onnx-bench__note">
        IoU is the highest-confidence box against the validated 1280 reference
        (406.4, 132.7) → (429.6, 211.2), reference confidence ~0.724.
      </p>
      <div className="benchmark-table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Runtime</th>
              <th>imgsz</th>
              <th>detections</th>
              <th>confidence</th>
              <th>box</th>
              <th>IoU</th>
            </tr>
          </thead>
          <tbody>
            {report.caltech.map((row) => (
              <tr key={`caltech-${row.runtime}-${row.imgsz}`}>
                <td>{row.runtime}</td>
                <td>{row.imgsz}</td>
                <td>{row.detections ?? '—'}</td>
                <td>{formatNum(row.confidence, 3)}</td>
                <td>{formatBox(row.box)}</td>
                <td>{formatNum(row.iouVsReference, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 className="onnx-bench__heading">Aerial preservation</h4>
      <p className="onnx-bench__note">
        Retention is versus this run&apos;s WASM_SINGLE imgsz 1280 count (
        {report.aerialWasm1280Count ?? '—'}). Prior browser reference was 124
        people. This is detection preservation, not mAP.
      </p>
      <div className="benchmark-table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Runtime</th>
              <th>imgsz</th>
              <th>detections</th>
              <th>delta vs 1280</th>
              <th>retained</th>
            </tr>
          </thead>
          <tbody>
            {report.aerial.map((row) => (
              <tr key={`aerial-${row.runtime}-${row.imgsz}`}>
                <td>{row.runtime}</td>
                <td>{row.imgsz}</td>
                <td>{row.detections ?? '—'}</td>
                <td>
                  {row.deltaFromWasm1280 == null
                    ? '—'
                    : `${row.deltaFromWasm1280 > 0 ? '+' : ''}${row.deltaFromWasm1280}`}
                </td>
                <td>
                  {row.percentRetained == null
                    ? '—'
                    : `${row.percentRetained.toFixed(1)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 className="onnx-bench__heading">WebGPU validation</h4>
      <p className="onnx-bench__note">
        Status {webgpuLabel}. navigator.gpu {webgpu.navigatorGpu ? 'yes' : 'no'}
        . Session created {webgpu.sessionCreated ? 'yes' : 'no'}. Inference
        executed {webgpu.inferenceExecuted ? 'yes' : 'no'}. Output parity{' '}
        {webgpu.outputParity}. Adapter {webgpu.adapterInfo ?? '—'}.
        {webgpu.reason ? ` Reason: ${webgpu.reason}` : ''}
      </p>
      {webgpu.parity.length > 0 && (
        <div className="benchmark-table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>imgsz</th>
                <th>image</th>
                <th>wasm</th>
                <th>webgpu</th>
                <th>conf Δ</th>
                <th>best IoU</th>
                <th>validated</th>
                <th>notes</th>
              </tr>
            </thead>
            <tbody>
              {webgpu.parity.map((check) => (
                <tr key={`parity-${check.imgsz}-${check.imageId}`}>
                  <td>{check.imgsz}</td>
                  <td>{check.imageLabel}</td>
                  <td>{check.wasmDetections}</td>
                  <td>{check.webgpuDetections}</td>
                  <td>{formatNum(check.highestConfidenceDelta, 4)}</td>
                  <td>{formatNum(check.bestBoxIou, 3)}</td>
                  <td>{check.validated ? 'yes' : 'no'}</td>
                  <td>{check.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h4 className="onnx-bench__heading">Stability</h4>
      <p className="onnx-bench__note">
        Model resource entries {stability.modelResourceEntriesAtStart} →{' '}
        {stability.modelResourceEntriesAfterSessionSetup} after session setup →{' '}
        {stability.modelResourceEntriesAtEnd} at end. Fetch hits during session
        setup {stability.fetchWrapperHitsDuringSessionSetup}, during inference{' '}
        {stability.fetchWrapperHitsDuringInference}. Measured-run model fetch
        violations {stability.inferenceModelFetchViolations}. Production session
        id {stability.productionSessionIdStart} →{' '}
        {stability.productionSessionIdEnd} (
        {stability.productionSessionCreationsDuringRun} created). Production
        session object stable{' '}
        {stability.productionSessionObjectStable ? 'yes' : 'no'}. WebGPU
        sessions this run {stability.webgpuSessionCreationsThisRun}, lifetime{' '}
        {stability.webgpuSessionCreationsLifetime}. Production WASM resource
        delta {stability.productionWasmResourceDelta}. Heap MB{' '}
        {formatNum(stability.heapMbStart, 1)} → wasm{' '}
        {formatNum(stability.heapMbAfterWasm, 1)} → webgpu{' '}
        {formatNum(stability.heapMbAfterWebGpu, 1)} → end{' '}
        {formatNum(stability.heapMbEnd, 1)}. Repeated-run counts stable{' '}
        {stability.repeatedRunCountsStable ? 'yes' : 'no'}.
      </p>
    </div>
  )
}
