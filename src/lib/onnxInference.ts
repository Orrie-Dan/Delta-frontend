import * as ort from 'onnxruntime-web/wasm'
import { getOnnxSessionId, initOnnxModel } from './onnxModel'
import type { Detection, DetectionResponse } from '../types/detection'

export const MODEL_IMGSZ = 1280
export const MODEL_STRIDE = 32
export const CONF_THRESHOLD = 0.25
export const NMS_IOU_THRESHOLD = 0.6
export const MAX_DETECTIONS = 1000
export const LETTERBOX_COLOR = 114
export const TEST_IMAGE_URL = '/test-images/set00_V001_0398.png'

export interface LetterboxMeta {
  originalWidth: number
  originalHeight: number
  ratio: number
  resizedWidth: number
  resizedHeight: number
  padX: number
  padY: number
  /** Final model canvas width (may be < imgsz when Ultralytics auto=True). */
  canvasWidth: number
  /** Final model canvas height (may be < imgsz when Ultralytics auto=True). */
  canvasHeight: number
  modelImgsz: number
}

export interface PreprocessDiagnostics {
  tensorShape: readonly number[]
  tensorMin: number
  tensorMax: number
}

export interface OnnxInferenceTiming {
  preprocessMs: number
  inferenceMs: number
  postprocessMs: number
  totalMs: number
}

export interface OnnxInferenceDiagnostics {
  outputName: string
  outputDims: readonly number[]
  anchorCount: number
  candidatesBeforeNms: number
  confThreshold: number
  iouThreshold: number
  letterbox: LetterboxMeta
  preprocess: PreprocessDiagnostics
  sessionIdentity: number
}

export interface OnnxInferenceResult {
  response: DetectionResponse
  timing: OnnxInferenceTiming
  diagnostics: OnnxInferenceDiagnostics
}

/** Duck-typed ONNX output so benchmark sessions from another bundle can decode. */
export interface OnnxTensorLike {
  readonly dims: readonly number[]
  readonly data: ArrayLike<number>
}

export interface OnnxSessionLike {
  readonly outputNames: readonly string[]
  run(feeds: Record<string, unknown>): Promise<Record<string, OnnxTensorLike>>
}

export interface IsolatedInferenceOptions {
  /**
   * Requested Ultralytics imgsz. Production callers omit this and stay at 1280.
   * auto=True letterbox may still emit a rectangular tensor.
   */
  modelImgsz?: number
  session: OnnxSessionLike
  sessionIdentity: number
  createTensor: (data: Float32Array, dims: readonly number[]) => unknown
}

export interface PreparedInput {
  data: Float32Array
  dims: readonly number[]
  letterbox: LetterboxMeta
  diagnostics: PreprocessDiagnostics
}

type ImageSource =
  | HTMLImageElement
  | HTMLVideoElement
  | ImageBitmap
  | HTMLCanvasElement

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function sourceSize(source: ImageSource): { width: number; height: number } {
  if (source instanceof HTMLImageElement) {
    return {
      width: source.naturalWidth || source.width,
      height: source.naturalHeight || source.height,
    }
  }
  if (source instanceof HTMLVideoElement) {
    return {
      width: source.videoWidth || source.clientWidth,
      height: source.videoHeight || source.clientHeight,
    }
  }
  return { width: source.width, height: source.height }
}

/**
 * Ultralytics LetterBox(auto=True, stride=32) semantics:
 * - scale so both sides fit within imgsz×imgsz
 * - reduce padding with pad % stride (minimum rectangle)
 * - remaining padding is centered with RGB(114,114,114)
 *
 * For set00_V001_0398.png @ 1280 this yields canvas 1280×960 with zero pad —
 * matching Ultralytics predict shape (1, 3, 960, 1280).
 */
export function letterboxMeta(
  originalWidth: number,
  originalHeight: number,
  modelImgsz = MODEL_IMGSZ,
  stride = MODEL_STRIDE,
): LetterboxMeta {
  if (originalWidth <= 0 || originalHeight <= 0) {
    throw new Error('Image has invalid dimensions.')
  }

  const ratio = Math.min(
    modelImgsz / originalWidth,
    modelImgsz / originalHeight,
  )
  const resizedWidth = Math.round(originalWidth * ratio)
  const resizedHeight = Math.round(originalHeight * ratio)

  // Ultralytics auto=True: keep only the stride remainder of the padding.
  let padW = (modelImgsz - resizedWidth) % stride
  let padH = (modelImgsz - resizedHeight) % stride

  // Center the remaining pad (Ultralytics: round(d/2 - 0.1), round(d/2 + 0.1)).
  const padX = Math.round(padW / 2 - 0.1)
  const padY = Math.round(padH / 2 - 0.1)
  const padRight = Math.round(padW / 2 + 0.1)
  const padBottom = Math.round(padH / 2 + 0.1)

  return {
    originalWidth,
    originalHeight,
    ratio,
    resizedWidth,
    resizedHeight,
    padX,
    padY,
    canvasWidth: resizedWidth + padX + padRight,
    canvasHeight: resizedHeight + padY + padBottom,
    modelImgsz,
  }
}

/** Letterbox + NCHW float packing. Does not create an ONNX tensor. */
export function prepareModelInput(
  source: ImageSource,
  modelImgsz = MODEL_IMGSZ,
): PreparedInput {
  const { width: originalWidth, height: originalHeight } = sourceSize(source)
  const letterbox = letterboxMeta(originalWidth, originalHeight, modelImgsz)
  const { canvasWidth, canvasHeight } = letterbox

  const canvas = document.createElement('canvas')
  canvas.width = canvasWidth
  canvas.height = canvasHeight
  const ctx = canvas.getContext('2d', {
    alpha: false,
    willReadFrequently: true,
  })
  if (!ctx) {
    throw new Error('Unable to create a 2D canvas context for preprocessing.')
  }

  ctx.fillStyle = `rgb(${LETTERBOX_COLOR}, ${LETTERBOX_COLOR}, ${LETTERBOX_COLOR})`
  ctx.fillRect(0, 0, canvasWidth, canvasHeight)
  ctx.drawImage(
    source,
    letterbox.padX,
    letterbox.padY,
    letterbox.resizedWidth,
    letterbox.resizedHeight,
  )

  const { data } = ctx.getImageData(0, 0, canvasWidth, canvasHeight)
  const plane = canvasWidth * canvasHeight
  const floatData = new Float32Array(3 * plane)

  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY

  for (let i = 0; i < plane; i += 1) {
    const px = i * 4
    const r = data[px]! / 255
    const g = data[px + 1]! / 255
    const b = data[px + 2]! / 255
    floatData[i] = r
    floatData[plane + i] = g
    floatData[2 * plane + i] = b
    if (r < min) min = r
    if (g < min) min = g
    if (b < min) min = b
    if (r > max) max = r
    if (g > max) max = g
    if (b > max) max = b
  }

  const dims = [1, 3, canvasHeight, canvasWidth] as const

  return {
    data: floatData,
    dims,
    letterbox,
    diagnostics: {
      tensorShape: dims,
      tensorMin: min,
      tensorMax: max,
    },
  }
}

export function preprocessToTensor(
  source: ImageSource,
  modelImgsz = MODEL_IMGSZ,
): {
  tensor: ort.Tensor
  letterbox: LetterboxMeta
  diagnostics: PreprocessDiagnostics
} {
  const prepared = prepareModelInput(source, modelImgsz)
  const tensor = new ort.Tensor('float32', prepared.data, prepared.dims)
  return {
    tensor,
    letterbox: prepared.letterbox,
    diagnostics: prepared.diagnostics,
  }
}

interface RawCandidate {
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
}

function iou(a: RawCandidate, b: RawCandidate): number {
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

export function nms(
  candidates: RawCandidate[],
  iouThreshold = NMS_IOU_THRESHOLD,
  maxDetections = MAX_DETECTIONS,
): RawCandidate[] {
  const sorted = [...candidates].sort((a, b) => b.confidence - a.confidence)
  const kept: RawCandidate[] = []

  for (const candidate of sorted) {
    if (kept.length >= maxDetections) break
    let suppressed = false
    for (const accepted of kept) {
      if (iou(candidate, accepted) > iouThreshold) {
        suppressed = true
        break
      }
    }
    if (!suppressed) kept.push(candidate)
  }

  return kept
}

function decodeAndPostprocess(
  output: OnnxTensorLike,
  letterbox: LetterboxMeta,
  confThreshold = CONF_THRESHOLD,
  iouThreshold = NMS_IOU_THRESHOLD,
): {
  detections: Detection[]
  anchorCount: number
  candidatesBeforeNms: number
  outputDims: readonly number[]
} {
  const dims = output.dims
  if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== 5) {
    throw new Error(
      `Unexpected ONNX output shape ${JSON.stringify(dims)}; expected [1, 5, anchors].`,
    )
  }

  const channels = dims[1]!
  const anchorCount = dims[2]!
  const data = output.data

  if (channels !== 5) {
    throw new Error(`Unexpected channel count ${channels}; expected 5.`)
  }

  const candidates: RawCandidate[] = []
  for (let i = 0; i < anchorCount; i += 1) {
    const confidence = data[4 * anchorCount + i]!
    if (confidence < confThreshold) continue

    const cx = data[0 * anchorCount + i]!
    const cy = data[1 * anchorCount + i]!
    const w = data[2 * anchorCount + i]!
    const h = data[3 * anchorCount + i]!

    candidates.push({
      confidence,
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
    })
  }

  const kept = nms(candidates, iouThreshold, MAX_DETECTIONS)
  const { ratio, padX, padY, originalWidth, originalHeight } = letterbox

  const detections: Detection[] = kept.map((box) => {
    const x1 = clamp((box.x1 - padX) / ratio, 0, originalWidth)
    const y1 = clamp((box.y1 - padY) / ratio, 0, originalHeight)
    const x2 = clamp((box.x2 - padX) / ratio, 0, originalWidth)
    const y2 = clamp((box.y2 - padY) / ratio, 0, originalHeight)
    return {
      confidence: Number(box.confidence.toFixed(4)),
      x1: Number(x1.toFixed(2)),
      y1: Number(y1.toFixed(2)),
      x2: Number(x2.toFixed(2)),
      y2: Number(y2.toFixed(2)),
    }
  })

  return {
    detections,
    anchorCount,
    candidatesBeforeNms: candidates.length,
    outputDims: dims,
  }
}

export async function loadImageElement(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.decoding = 'async'
    image.onload = () => resolve(image)
    image.onerror = () =>
      reject(new Error(`Failed to load image: ${url}`))
    image.src = url
  })
}

/** Load a user-uploaded File/Blob into an HTMLImageElement for browser inference. */
export async function loadImageFromBlob(blob: Blob): Promise<{
  image: HTMLImageElement
  objectUrl: string
}> {
  const objectUrl = URL.createObjectURL(blob)
  try {
    const image = await loadImageElement(objectUrl)
    return { image, objectUrl }
  } catch (error) {
    URL.revokeObjectURL(objectUrl)
    throw error
  }
}

/**
 * Timed inference against a caller-supplied session.
 * The clock starts after the session already exists, so model download and
 * session creation stay outside preprocess / session.run / postprocess.
 */
export async function runIsolatedBrowserInference(
  source: ImageSource,
  options: IsolatedInferenceOptions,
): Promise<OnnxInferenceResult> {
  const modelImgsz = options.modelImgsz ?? MODEL_IMGSZ
  const totalStarted = performance.now()

  const preprocessStarted = performance.now()
  const prepared = prepareModelInput(source, modelImgsz)
  const tensor = options.createTensor(prepared.data, prepared.dims)
  const preprocessMs = performance.now() - preprocessStarted

  const inferenceStarted = performance.now()
  const outputs = await options.session.run({ images: tensor })
  const inferenceMs = performance.now() - inferenceStarted

  const outputName = options.session.outputNames[0] ?? 'output0'
  const output = outputs[outputName]
  if (!output) {
    throw new Error(`ONNX session did not return output "${outputName}".`)
  }

  const postStarted = performance.now()
  const decoded = decodeAndPostprocess(output, prepared.letterbox)
  const postprocessMs = performance.now() - postStarted
  const totalMs = performance.now() - totalStarted

  const response: DetectionResponse = {
    people: decoded.detections.length,
    inference_ms: Number(inferenceMs.toFixed(2)),
    image: {
      width: prepared.letterbox.originalWidth,
      height: prepared.letterbox.originalHeight,
    },
    detections: decoded.detections,
    model_imgsz: modelImgsz,
  }

  return {
    response,
    timing: {
      preprocessMs,
      inferenceMs,
      postprocessMs,
      totalMs,
    },
    diagnostics: {
      outputName,
      outputDims: decoded.outputDims,
      anchorCount: decoded.anchorCount,
      candidatesBeforeNms: decoded.candidatesBeforeNms,
      confThreshold: CONF_THRESHOLD,
      iouThreshold: NMS_IOU_THRESHOLD,
      letterbox: prepared.letterbox,
      preprocess: prepared.diagnostics,
      sessionIdentity: options.sessionIdentity,
    },
  }
}

/**
 * Run one static-image browser inference using the shared production
 * WASM InferenceSession (single thread). Static upload only.
 *
 * `modelImgsz` defaults to 1280. Production upload omits it.
 */
export async function runBrowserOnnxInference(
  source: ImageSource,
  options?: { modelImgsz?: number },
): Promise<OnnxInferenceResult> {
  const totalStarted = performance.now()
  const session = await initOnnxModel()
  const sessionIdentity = getOnnxSessionId()

  const result = await runIsolatedBrowserInference(source, {
    modelImgsz: options?.modelImgsz ?? MODEL_IMGSZ,
    session: session as unknown as OnnxSessionLike,
    sessionIdentity,
    createTensor: (data, dims) => new ort.Tensor('float32', data, dims),
  })

  return {
    ...result,
    timing: {
      ...result.timing,
      totalMs: performance.now() - totalStarted,
    },
  }
}

export async function runBrowserOnnxInferenceFromFile(
  file: File | Blob,
): Promise<OnnxInferenceResult> {
  const { image, objectUrl } = await loadImageFromBlob(file)
  try {
    return await runBrowserOnnxInference(image)
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

export async function runCaltechOnnxTest(): Promise<OnnxInferenceResult> {
  const image = await loadImageElement(TEST_IMAGE_URL)
  const result = await runBrowserOnnxInference(image)

  console.info('[onnx-test] Caltech static inference', {
    people: result.response.people,
    detections: result.response.detections,
    timing: result.timing,
    letterbox: result.diagnostics.letterbox,
    preprocess: result.diagnostics.preprocess,
    output: {
      name: result.diagnostics.outputName,
      dims: result.diagnostics.outputDims,
      anchors: result.diagnostics.anchorCount,
      candidatesBeforeNms: result.diagnostics.candidatesBeforeNms,
    },
    sessionIdentity: result.diagnostics.sessionIdentity,
  })

  return result
}
