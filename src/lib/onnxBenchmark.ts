import * as ort from 'onnxruntime-web/wasm'
import type { Detection } from '../types/detection'
import {
  TEST_IMAGE_URL,
  loadImageElement,
  runIsolatedBrowserInference,
  type IsolatedInferenceOptions,
  type OnnxSessionLike,
} from './onnxInference'
import { getOnnxSession, getOnnxSessionId, initOnnxModel } from './onnxModel'
import {
  getWebGpuInferenceOptions,
  getWebGpuRuntimeSnapshot,
  getWebGpuSessionCreations,
  hasNavigatorGpu,
  initWebGpuRuntime,
} from './onnxWebGpu'

/**
 * Debug-only browser runtime matrix for WEB-ONNX-05.
 * Production upload keeps the shared WASM session. Production live camera uses
 * the WebGPU session in onnxWebGpu.ts. This module must not run unless the
 * debug benchmark is started explicitly.
 */

export const ONNXRUNTIME_WEB_VERSION = '1.30.0'
export const BENCHMARK_IMGSZ = [1280, 960, 640] as const
export const WARMUP_RUNS = 1
export const MEASURED_RUNS = 3

/** Validated WEB-ONNX-03/04 browser box on set00_V001_0398.png at imgsz 1280. */
export const CALTECH_REFERENCE_BOX = {
  x1: 406.4,
  y1: 132.7,
  x2: 429.6,
  y2: 211.2,
} as const

const MODEL_FILENAME = 'yolo26s_1280_mixed_v1.onnx'

export type BenchmarkRuntimeId = 'WASM_SINGLE' | 'WEBGPU'
export type BenchmarkImageId = 'caltech' | 'aerial' | 'camera'
export type BenchmarkImgsz = (typeof BENCHMARK_IMGSZ)[number]

export interface BenchmarkImageSpec {
  id: BenchmarkImageId
  label: string
  /** Filename shown in the report. Aerial bytes match Downloads "test vidrone.webp". */
  file: string
  url: string
}

export const BENCHMARK_IMAGES: readonly BenchmarkImageSpec[] = [
  {
    id: 'caltech',
    label: 'Caltech',
    file: 'set00_V001_0398.png',
    url: TEST_IMAGE_URL,
  },
  {
    id: 'aerial',
    label: 'Aerial',
    file: 'test vidrone.webp',
    url: '/test-images/real-world-visdrone.webp',
  },
  {
    id: 'camera',
    label: 'Camera',
    file: 'real-world-test.jpeg',
    url: '/test-images/real-world-test.jpeg',
  },
]

export interface BenchmarkBox {
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface TimedRun {
  preprocessMs: number
  inferenceMs: number
  postprocessMs: number
  totalMs: number
  detections: number
}

export interface BenchmarkRow {
  runtime: BenchmarkRuntimeId
  imgsz: BenchmarkImgsz
  imageId: BenchmarkImageId
  imageLabel: string
  status: 'ok' | 'error'
  error: string | null
  tensorShape: readonly number[] | null
  anchorCount: number | null
  candidatesBeforeNms: number | null
  detections: number | null
  highestConfidence: number | null
  highestBox: BenchmarkBox | null
  countsStable: boolean
  medianPreprocessMs: number | null
  medianInferenceMs: number | null
  medianPostprocessMs: number | null
  medianTotalMs: number | null
  minTotalMs: number | null
  maxTotalMs: number | null
  sessionIdentity: number | null
  measuredRuns: TimedRun[]
  detectionsSnapshot: BenchmarkBox[]
}

export interface CaltechPreservation {
  runtime: BenchmarkRuntimeId
  imgsz: BenchmarkImgsz
  detections: number | null
  confidence: number | null
  box: BenchmarkBox | null
  iouVsReference: number | null
}

export interface AerialPreservation {
  runtime: BenchmarkRuntimeId
  imgsz: BenchmarkImgsz
  detections: number | null
  deltaFromWasm1280: number | null
  percentRetained: number | null
}

export interface WebGpuParityCheck {
  imgsz: BenchmarkImgsz
  imageId: BenchmarkImageId
  imageLabel: string
  wasmDetections: number
  webgpuDetections: number
  countDelta: number
  highestConfidenceDelta: number | null
  matchedFraction: number | null
  medianMatchedIou: number | null
  bestBoxIou: number | null
  validated: boolean
  notes: string
}

export interface WebGpuReport {
  navigatorGpu: boolean
  adapterInfo: string | null
  sessionCreated: boolean
  inferenceExecuted: boolean
  status: 'ready' | 'unsupported' | 'blocked'
  reason: string | null
  createdThisRun: boolean
  lifetimeSessionCreations: number
  parity: WebGpuParityCheck[]
  outputParity: 'not-run' | 'validated' | 'diverged'
}

export interface StabilityReport {
  modelResourceEntriesAtStart: number
  modelResourceEntriesAfterSessionSetup: number
  modelResourceEntriesAtEnd: number
  fetchWrapperHitsDuringSessionSetup: number
  fetchWrapperHitsDuringInference: number
  inferenceModelFetchViolations: number
  productionWasmResourceDelta: number
  productionSessionIdStart: number
  productionSessionIdEnd: number
  productionSessionCreationsDuringRun: number
  productionSessionObjectStable: boolean
  webgpuSessionCreationsThisRun: number
  webgpuSessionCreationsLifetime: number
  heapMbStart: number | null
  heapMbAfterWasm: number | null
  heapMbAfterWebGpu: number | null
  heapMbEnd: number | null
  repeatedRunCountsStable: boolean
}

export interface BenchmarkEnvironment {
  userAgent: string
  hardwareConcurrency: number
  deviceMemoryGb: number | null
  webGlRenderer: string | null
  onnxruntimeWeb: string
  navigatorGpu: boolean
}

export interface BenchmarkReport {
  environment: BenchmarkEnvironment
  images: Array<{ id: BenchmarkImageId; file: string; width: number; height: number }>
  rows: BenchmarkRow[]
  caltech: CaltechPreservation[]
  aerial: AerialPreservation[]
  aerialWasm1280Count: number | null
  webgpu: WebGpuReport
  stability: StabilityReport
  completed: boolean
}

interface RuntimeHandle {
  id: BenchmarkRuntimeId
  session: OnnxSessionLike
  sessionIdentity: number
  createTensor: IsolatedInferenceOptions['createTensor']
}

export type WebGpuRuntimeInitResult =
  | {
      ok: true
      createdThisCall: boolean
      adapterInfo: string
      sessionIdentity: number
    }
  | {
      ok: false
      status: 'unsupported' | 'blocked'
      reason: string
      adapterInfo: string | null
    }

/**
 * Debug entry. The session itself lives in onnxWebGpu.ts so production live
 * camera does not import this benchmark module.
 */
export async function ensureWebGpuInferenceRuntime(): Promise<WebGpuRuntimeInitResult> {
  const prep = await initWebGpuRuntime()
  if (prep.snapshot.status === 'ready') {
    return {
      ok: true,
      createdThisCall: prep.createdThisCall,
      adapterInfo: prep.snapshot.adapterInfo ?? 'adapter present',
      sessionIdentity: prep.snapshot.sessionIdentity,
    }
  }
  const unavailable =
    prep.snapshot.status === 'unavailable' || prep.snapshot.status === 'loading'
  return {
    ok: false,
    status: unavailable ? 'unsupported' : 'blocked',
    reason: prep.snapshot.errorMessage ?? 'WebGPU runtime is not ready.',
    adapterInfo: prep.snapshot.adapterInfo,
  }
}

export function getWebGpuInferenceRuntimeOptions(): Omit<
  IsolatedInferenceOptions,
  'modelImgsz'
> | null {
  return getWebGpuInferenceOptions()
}

export function getCachedWebGpuAdapterInfo(): string | null {
  return getWebGpuRuntimeSnapshot().adapterInfo
}

function webGpuHandle(): RuntimeHandle | null {
  const options = getWebGpuInferenceOptions()
  if (!options) return null
  return { id: 'WEBGPU', ...options }
}

function errorText(error: unknown): string {
  if (error instanceof Error) {
    const prefix = error.name && error.name !== 'Error' ? `${error.name}: ` : ''
    return `${prefix}${error.message}`
  }
  if (typeof error === 'string') return error
  try {
    return JSON.stringify(error)
  } catch {
    return String(error)
  }
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? null
}

function resourceCount(fragment: string): number {
  return performance
    .getEntriesByType('resource')
    .filter((entry) => entry.name.includes(fragment)).length
}

function usedHeapMb(): number | null {
  const memory = (
    performance as Performance & { memory?: { usedJSHeapSize: number } }
  ).memory
  if (!memory) return null
  return Math.round((memory.usedJSHeapSize / (1024 * 1024)) * 10) / 10
}

function webGlRenderer(): string | null {
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl')
    if (!gl) return null
    const ext = gl.getExtension('WEBGL_debug_renderer_info')
    if (!ext) return String(gl.getParameter(gl.RENDERER))
    return String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL))
  } catch {
    return null
  }
}

function installModelFetchCounter(): {
  count: () => number
  stop: () => void
} {
  let hits = 0
  const original = window.fetch.bind(window)
  const wrapped: typeof window.fetch = (input, init) => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : input.url
    if (url.includes(MODEL_FILENAME)) hits += 1
    return original(input, init)
  }
  window.fetch = wrapped
  return {
    count: () => hits,
    stop: () => {
      if (window.fetch === wrapped) window.fetch = original
    },
  }
}

async function yieldToUi(): Promise<void> {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, 0)
  })
}

function toBox(detection: Detection): BenchmarkBox {
  return {
    confidence: detection.confidence,
    x1: detection.x1,
    y1: detection.y1,
    x2: detection.x2,
    y2: detection.y2,
  }
}

function highestBox(boxes: readonly BenchmarkBox[]): BenchmarkBox | null {
  let best: BenchmarkBox | null = null
  for (const box of boxes) {
    if (!best || box.confidence > best.confidence) best = box
  }
  return best
}

export function boxIou(
  a: { x1: number; y1: number; x2: number; y2: number },
  b: { x1: number; y1: number; x2: number; y2: number },
): number {
  const interX1 = Math.max(a.x1, b.x1)
  const interY1 = Math.max(a.y1, b.y1)
  const interX2 = Math.min(a.x2, b.x2)
  const interY2 = Math.min(a.y2, b.y2)
  const interW = Math.max(0, interX2 - interX1)
  const interH = Math.max(0, interY2 - interY1)
  const inter = interW * interH
  if (inter <= 0) return 0
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1)
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

function compareWebGpuToWasm(
  wasm: readonly BenchmarkBox[],
  webgpu: readonly BenchmarkBox[],
): Omit<WebGpuParityCheck, 'imgsz' | 'imageId' | 'imageLabel'> {
  const wasmCount = wasm.length
  const webgpuCount = webgpu.length
  const countDelta = webgpuCount - wasmCount
  const countOk =
    wasmCount <= 3
      ? countDelta === 0
      : Math.abs(countDelta) <= Math.max(2, Math.round(wasmCount * 0.05))

  const wasmTop = highestBox(wasm)
  const gpuTop = highestBox(webgpu)
  const highestConfidenceDelta =
    wasmTop && gpuTop
      ? Math.abs(wasmTop.confidence - gpuTop.confidence)
      : null
  const confidenceOk =
    wasmCount === 0 && webgpuCount === 0
      ? true
      : highestConfidenceDelta != null && highestConfidenceDelta <= 0.05

  const used = new Set<number>()
  const matchedIous: number[] = []
  const ordered = [...wasm].sort((a, b) => b.confidence - a.confidence)
  for (const box of ordered) {
    let bestJ = -1
    let bestIou = 0
    for (let index = 0; index < webgpu.length; index += 1) {
      if (used.has(index)) continue
      const value = boxIou(box, webgpu[index]!)
      if (value > bestIou) {
        bestIou = value
        bestJ = index
      }
    }
    if (bestJ >= 0 && bestIou >= 0.5) {
      used.add(bestJ)
      matchedIous.push(bestIou)
    }
  }

  const matchedFraction = wasmCount === 0 ? 1 : matchedIous.length / wasmCount
  const medianMatchedIou = median(matchedIous)
  const bestBoxIou =
    wasmTop == null
      ? null
      : webgpu.reduce((best, box) => Math.max(best, boxIou(wasmTop, box)), 0)

  const matchOk =
    wasmCount === 0
      ? webgpuCount === 0
      : matchedFraction >= 0.9 &&
        (medianMatchedIou ?? 0) >= 0.9 &&
        (bestBoxIou ?? 0) >= 0.9

  const validated = countOk && confidenceOk && matchOk
  const notes: string[] = []
  if (!countOk) {
    notes.push(
      `detection count diverged (wasm ${wasmCount}, webgpu ${webgpuCount})`,
    )
  }
  if (!confidenceOk) {
    notes.push(
      highestConfidenceDelta == null
        ? 'highest confidence missing on one runtime'
        : `highest confidence delta ${highestConfidenceDelta.toFixed(4)} exceeds 0.05`,
    )
  }
  if (!matchOk) {
    notes.push(
      `box match fraction ${(matchedFraction * 100).toFixed(1)}%, median IoU ${
        medianMatchedIou == null ? 'n/a' : medianMatchedIou.toFixed(3)
      }, best-box IoU ${bestBoxIou == null ? 'n/a' : bestBoxIou.toFixed(3)}`,
    )
  }
  if (validated) {
    const small =
      (highestConfidenceDelta ?? 0) > 0.02 ||
      (medianMatchedIou != null && medianMatchedIou < 0.98) ||
      (bestBoxIou != null && bestBoxIou < 0.98)
    notes.push(
      small
        ? 'validated with small numerical differences'
        : 'validated',
    )
  }

  return {
    wasmDetections: wasmCount,
    webgpuDetections: webgpuCount,
    countDelta,
    highestConfidenceDelta,
    matchedFraction,
    medianMatchedIou,
    bestBoxIou,
    validated,
    notes: notes.join('; '),
  }
}

function emptyRow(
  runtime: RuntimeHandle,
  image: BenchmarkImageSpec,
  imgsz: BenchmarkImgsz,
  error: string,
): BenchmarkRow {
  return {
    runtime: runtime.id,
    imgsz,
    imageId: image.id,
    imageLabel: image.label,
    status: 'error',
    error,
    tensorShape: null,
    anchorCount: null,
    candidatesBeforeNms: null,
    detections: null,
    highestConfidence: null,
    highestBox: null,
    countsStable: false,
    medianPreprocessMs: null,
    medianInferenceMs: null,
    medianPostprocessMs: null,
    medianTotalMs: null,
    minTotalMs: null,
    maxTotalMs: null,
    sessionIdentity: runtime.sessionIdentity,
    measuredRuns: [],
    detectionsSnapshot: [],
  }
}

async function measureCell(
  runtime: RuntimeHandle,
  image: BenchmarkImageSpec,
  source: HTMLImageElement,
  imgsz: BenchmarkImgsz,
  onProgress: (message: string) => void,
  noteInferenceFetch: () => void,
): Promise<BenchmarkRow> {
  const label = `${runtime.id} · imgsz ${imgsz} · ${image.label}`

  try {
    onProgress(`${label} · warmup`)
    await yieldToUi()
    await runIsolatedBrowserInference(source, {
      modelImgsz: imgsz,
      session: runtime.session,
      sessionIdentity: runtime.sessionIdentity,
      createTensor: runtime.createTensor,
    })
  } catch (error) {
    return emptyRow(runtime, image, imgsz, `warmup failed: ${errorText(error)}`)
  }

  const measuredRuns: TimedRun[] = []
  const snapshots: BenchmarkBox[][] = []
  let tensorShape: readonly number[] | null = null
  let anchorCount: number | null = null
  let candidatesBeforeNms: number | null = null

  try {
    for (let index = 0; index < MEASURED_RUNS; index += 1) {
      onProgress(`${label} · measured ${index + 1}/${MEASURED_RUNS}`)
      await yieldToUi()
      const resourcesBefore = resourceCount(MODEL_FILENAME)
      const result = await runIsolatedBrowserInference(source, {
        modelImgsz: imgsz,
        session: runtime.session,
        sessionIdentity: runtime.sessionIdentity,
        createTensor: runtime.createTensor,
      })
      if (resourceCount(MODEL_FILENAME) > resourcesBefore) noteInferenceFetch()

      measuredRuns.push({
        preprocessMs: result.timing.preprocessMs,
        inferenceMs: result.timing.inferenceMs,
        postprocessMs: result.timing.postprocessMs,
        totalMs: result.timing.totalMs,
        detections: result.response.people,
      })
      snapshots.push(result.response.detections.map(toBox))
      tensorShape = result.diagnostics.preprocess.tensorShape
      anchorCount = result.diagnostics.anchorCount
      candidatesBeforeNms = result.diagnostics.candidatesBeforeNms
    }
  } catch (error) {
    return emptyRow(
      runtime,
      image,
      imgsz,
      `measured run failed: ${errorText(error)}`,
    )
  }

  const totals = measuredRuns.map((run) => run.totalMs)
  const medianTotal = median(totals)
  const representativeIndex = measuredRuns.findIndex(
    (run) => run.totalMs === medianTotal,
  )
  const snapshot =
    snapshots[representativeIndex >= 0 ? representativeIndex : 0] ?? []
  const top = highestBox(snapshot)
  const counts = measuredRuns.map((run) => run.detections)
  const countsStable = counts.every((count) => count === counts[0])

  return {
    runtime: runtime.id,
    imgsz,
    imageId: image.id,
    imageLabel: image.label,
    status: 'ok',
    error: null,
    tensorShape,
    anchorCount,
    candidatesBeforeNms,
    detections: snapshot.length,
    highestConfidence: top?.confidence ?? null,
    highestBox: top,
    countsStable,
    medianPreprocessMs: median(measuredRuns.map((run) => run.preprocessMs)),
    medianInferenceMs: median(measuredRuns.map((run) => run.inferenceMs)),
    medianPostprocessMs: median(measuredRuns.map((run) => run.postprocessMs)),
    medianTotalMs: medianTotal,
    minTotalMs: Math.min(...totals),
    maxTotalMs: Math.max(...totals),
    sessionIdentity: runtime.sessionIdentity,
    measuredRuns,
    detectionsSnapshot: snapshot,
  }
}

function buildPreservation(rows: BenchmarkRow[]): {
  caltech: CaltechPreservation[]
  aerial: AerialPreservation[]
  aerialWasm1280Count: number | null
} {
  const aerialBaseline =
    rows.find(
      (row) =>
        row.runtime === 'WASM_SINGLE' &&
        row.imgsz === 1280 &&
        row.imageId === 'aerial' &&
        row.status === 'ok',
    )?.detections ?? null

  const caltech: CaltechPreservation[] = []
  const aerial: AerialPreservation[] = []

  for (const row of rows) {
    if (row.imageId === 'caltech') {
      caltech.push({
        runtime: row.runtime,
        imgsz: row.imgsz,
        detections: row.detections,
        confidence: row.highestConfidence,
        box: row.highestBox,
        iouVsReference:
          row.highestBox == null
            ? null
            : boxIou(row.highestBox, CALTECH_REFERENCE_BOX),
      })
    }
    if (row.imageId === 'aerial') {
      const detections = row.detections
      aerial.push({
        runtime: row.runtime,
        imgsz: row.imgsz,
        detections,
        deltaFromWasm1280:
          detections == null || aerialBaseline == null
            ? null
            : detections - aerialBaseline,
        percentRetained:
          detections == null || aerialBaseline == null || aerialBaseline === 0
            ? null
            : (detections / aerialBaseline) * 100,
      })
    }
  }

  return { caltech, aerial, aerialWasm1280Count: aerialBaseline }
}

function buildParity(rows: BenchmarkRow[]): WebGpuParityCheck[] {
  const checks: WebGpuParityCheck[] = []
  for (const image of BENCHMARK_IMAGES) {
    for (const imgsz of BENCHMARK_IMGSZ) {
      const wasm = rows.find(
        (row) =>
          row.runtime === 'WASM_SINGLE' &&
          row.imageId === image.id &&
          row.imgsz === imgsz &&
          row.status === 'ok',
      )
      const gpu = rows.find(
        (row) =>
          row.runtime === 'WEBGPU' &&
          row.imageId === image.id &&
          row.imgsz === imgsz &&
          row.status === 'ok',
      )
      if (!wasm || !gpu) continue
      checks.push({
        imgsz,
        imageId: image.id,
        imageLabel: image.label,
        ...compareWebGpuToWasm(wasm.detectionsSnapshot, gpu.detectionsSnapshot),
      })
    }
  }
  return checks
}

export async function runBrowserBenchmark(
  onProgress: (message: string) => void = () => {},
): Promise<BenchmarkReport> {
  const fetchCounter = installModelFetchCounter()
  const productionSessionBefore = getOnnxSession()
  const productionSessionIdStart = getOnnxSessionId()
  const resourcesAtStart = resourceCount(MODEL_FILENAME)
  const wasmResourcesAtStart = resourceCount('ort-wasm-simd-threaded.wasm')
  const heapStart = usedHeapMb()
  let inferenceModelFetchViolations = 0
  let fetchHitsAtSetupEnd = 0
  let heapAfterWasm: number | null = null
  let heapAfterWebGpu: number | null = null
  let webgpuCreatedThisRun = false
  let webgpuReason: string | null = null
  let webgpuStatus: WebGpuReport['status'] = 'unsupported'
  let navigatorGpu = false
  let sessionCreated = false

  const rows: BenchmarkRow[] = []

  try {
    onProgress('Checking WebGPU session')
    await yieldToUi()
    const setupFetchesBefore = fetchCounter.count()
    const webgpuPrep = await ensureWebGpuInferenceRuntime()
    fetchHitsAtSetupEnd = fetchCounter.count() - setupFetchesBefore
    navigatorGpu = hasNavigatorGpu()
    const webgpuRuntime = webGpuHandle()

    if (!webgpuPrep.ok) {
      webgpuStatus = webgpuPrep.status
      webgpuReason = webgpuPrep.reason
    } else {
      webgpuStatus = 'ready'
      webgpuCreatedThisRun = webgpuPrep.createdThisCall
      sessionCreated = true
    }

    const resourcesAfterSetup = resourceCount(MODEL_FILENAME)

    onProgress('Loading benchmark images')
    await yieldToUi()
    const loaded = new Map<BenchmarkImageId, HTMLImageElement>()
    const imageMeta: BenchmarkReport['images'] = []
    for (const image of BENCHMARK_IMAGES) {
      const element = await loadImageElement(image.url)
      loaded.set(image.id, element)
      imageMeta.push({
        id: image.id,
        file: image.file,
        width: element.naturalWidth,
        height: element.naturalHeight,
      })
    }

    onProgress('Preparing production WASM session')
    await yieldToUi()
    const productionSession = await initOnnxModel()
    const wasmRuntime: RuntimeHandle = {
      id: 'WASM_SINGLE',
      session: productionSession as unknown as OnnxSessionLike,
      sessionIdentity: getOnnxSessionId(),
      createTensor: (data, dims) => new ort.Tensor('float32', data, dims),
    }

    for (const imgsz of BENCHMARK_IMGSZ) {
      for (const image of BENCHMARK_IMAGES) {
        const source = loaded.get(image.id)
        if (!source) {
          rows.push(emptyRow(wasmRuntime, image, imgsz, 'Image failed to load.'))
          continue
        }
        rows.push(
          await measureCell(
            wasmRuntime,
            image,
            source,
            imgsz,
            onProgress,
            () => {
              inferenceModelFetchViolations += 1
            },
          ),
        )
      }
    }
    heapAfterWasm = usedHeapMb()

    if (webgpuPrep.ok && webgpuRuntime) {
      for (const imgsz of BENCHMARK_IMGSZ) {
        for (const image of BENCHMARK_IMAGES) {
          const source = loaded.get(image.id)
          if (!source) {
            rows.push(
              emptyRow(webgpuRuntime, image, imgsz, 'Image failed to load.'),
            )
            continue
          }
          rows.push(
            await measureCell(
              webgpuRuntime,
              image,
              source,
              imgsz,
              onProgress,
              () => {
                inferenceModelFetchViolations += 1
              },
            ),
          )
        }
      }
      heapAfterWebGpu = usedHeapMb()
    }

    const inferenceExecuted = rows.some(
      (row) => row.runtime === 'WEBGPU' && row.status === 'ok',
    )
    if (webgpuPrep.ok && !inferenceExecuted) {
      const firstError = rows.find(
        (row) => row.runtime === 'WEBGPU' && row.status === 'error',
      )?.error
      webgpuStatus = 'blocked'
      webgpuReason =
        firstError ?? 'WebGPU session was created but no inference succeeded.'
    }

    const parity = buildParity(rows)
    const outputParity: WebGpuReport['outputParity'] = !inferenceExecuted
      ? 'not-run'
      : parity.length > 0 && parity.every((check) => check.validated)
        ? 'validated'
        : 'diverged'

    const preservation = buildPreservation(rows)
    const productionSessionIdEnd = getOnnxSessionId()
    const heapEnd = usedHeapMb()
    const repeatedRunCountsStable = rows
      .filter((row) => row.status === 'ok')
      .every((row) => row.countsStable)

    const report: BenchmarkReport = {
      environment: {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        deviceMemoryGb:
          (navigator as Navigator & { deviceMemory?: number }).deviceMemory ??
          null,
        webGlRenderer: webGlRenderer(),
        onnxruntimeWeb: ONNXRUNTIME_WEB_VERSION,
        navigatorGpu,
      },
      images: imageMeta,
      rows,
      caltech: preservation.caltech,
      aerial: preservation.aerial,
      aerialWasm1280Count: preservation.aerialWasm1280Count,
      webgpu: {
        navigatorGpu,
        adapterInfo: webgpuPrep.adapterInfo,
        sessionCreated,
        inferenceExecuted,
        status: webgpuStatus,
        reason: webgpuReason,
        createdThisRun: webgpuCreatedThisRun,
        lifetimeSessionCreations: getWebGpuSessionCreations(),
        parity,
        outputParity,
      },
      stability: {
        modelResourceEntriesAtStart: resourcesAtStart,
        modelResourceEntriesAfterSessionSetup: resourcesAfterSetup,
        modelResourceEntriesAtEnd: resourceCount(MODEL_FILENAME),
        fetchWrapperHitsDuringSessionSetup: fetchHitsAtSetupEnd,
        fetchWrapperHitsDuringInference: fetchCounter.count() - fetchHitsAtSetupEnd,
        inferenceModelFetchViolations,
        productionWasmResourceDelta:
          resourceCount('ort-wasm-simd-threaded.wasm') - wasmResourcesAtStart,
        productionSessionIdStart,
        productionSessionIdEnd,
        productionSessionCreationsDuringRun:
          productionSessionIdEnd - productionSessionIdStart,
        productionSessionObjectStable:
          productionSessionBefore != null &&
          getOnnxSession() === productionSessionBefore,
        webgpuSessionCreationsThisRun: webgpuCreatedThisRun ? 1 : 0,
        webgpuSessionCreationsLifetime: getWebGpuSessionCreations(),
        heapMbStart: heapStart,
        heapMbAfterWasm: heapAfterWasm,
        heapMbAfterWebGpu: heapAfterWebGpu,
        heapMbEnd: heapEnd,
        repeatedRunCountsStable,
      },
      completed: true,
    }

    ;(
      window as Window & { __onnxBenchmarkReport?: BenchmarkReport }
    ).__onnxBenchmarkReport = report
    onProgress('Benchmark complete')
    return report
  } finally {
    fetchCounter.stop()
  }
}
