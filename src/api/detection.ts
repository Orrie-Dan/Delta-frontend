import type { DetectionResponse, HealthResponse } from '../types/detection'
import type { RequestTimingMetrics } from '../lib/liveTypes'

const DEFAULT_API_URL = 'https://delta-o8cc.onrender.com'

export const API_BASE_URL =
  (import.meta.env.VITE_DETECTION_API_URL as string | undefined)?.replace(
    /\/$/,
    '',
  ) || DEFAULT_API_URL

const HEALTH_TIMEOUT_MS = 10_000
const DETECT_TIMEOUT_MS = 120_000

export class ApiError extends Error {
  status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)

  const onExternalAbort = () => controller.abort()
  if (options.signal) {
    if (options.signal.aborted) {
      window.clearTimeout(timer)
      throw new ApiError('Detection request was cancelled.')
    }
    options.signal.addEventListener('abort', onExternalAbort, { once: true })
  }

  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (options.signal?.aborted) {
        throw new ApiError('Detection request was cancelled.')
      }
      throw new ApiError(
        `Request timed out after ${Math.round(timeoutMs / 1000)}s. Please try again.`,
      )
    }
    throw new ApiError(
      'Unable to reach the detection API. Check your connection and try again.',
    )
  } finally {
    window.clearTimeout(timer)
    options.signal?.removeEventListener('abort', onExternalAbort)
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/health`,
    { method: 'GET' },
    HEALTH_TIMEOUT_MS,
  )

  if (!response.ok) {
    throw new ApiError(
      `Health check failed (${response.status}). The model service may be unavailable.`,
      response.status,
    )
  }

  const data = (await response.json()) as Partial<HealthResponse>

  if (!data || data.status !== 'ok') {
    throw new ApiError('Health check returned an unexpected response.')
  }

  return data as HealthResponse
}

function isValidDetectionResponse(data: unknown): data is DetectionResponse {
  if (!data || typeof data !== 'object') return false
  const obj = data as Record<string, unknown>

  if (typeof obj.people !== 'number') return false
  if (typeof obj.inference_ms !== 'number') return false
  if (!obj.image || typeof obj.image !== 'object') return false

  const image = obj.image as Record<string, unknown>
  if (typeof image.width !== 'number' || typeof image.height !== 'number') {
    return false
  }

  if (!Array.isArray(obj.detections)) return false

  return obj.detections.every((item) => {
    if (!item || typeof item !== 'object') return false
    const d = item as Record<string, unknown>
    return (
      typeof d.confidence === 'number' &&
      typeof d.x1 === 'number' &&
      typeof d.y1 === 'number' &&
      typeof d.x2 === 'number' &&
      typeof d.y2 === 'number'
    )
  })
}

export interface DetectPeopleOptions {
  filename?: string
  signal?: AbortSignal
  /** Only sent when backend supports it. Currently deployed API does not. */
  modelImgsz?: number
}

export interface DetectPeopleResult {
  response: DetectionResponse
  timing: RequestTimingMetrics
}

function buildDetectUrl(modelImgsz?: number): string {
  if (!modelImgsz) return `${API_BASE_URL}/detect`
  const url = new URL(`${API_BASE_URL}/detect`)
  url.searchParams.set('imgsz', String(modelImgsz))
  return url.toString()
}

export async function detectPeople(
  file: File | Blob,
  options?: DetectPeopleOptions,
): Promise<DetectionResponse> {
  const result = await detectPeopleInstrumented(file, options)
  return result.response
}

export async function detectPeopleInstrumented(
  file: File | Blob,
  options?: DetectPeopleOptions,
): Promise<DetectPeopleResult> {
  const formData = new FormData()
  const filename =
    options?.filename ?? (file instanceof File ? file.name : 'frame.jpg')
  formData.append('file', file, filename)

  const requestStarted = performance.now()
  const response = await fetchWithTimeout(
    buildDetectUrl(options?.modelImgsz),
    {
      method: 'POST',
      body: formData,
      signal: options?.signal,
    },
    DETECT_TIMEOUT_MS,
  )

  if (!response.ok) {
    let detail = `Detection failed (${response.status}).`
    try {
      const errorBody = (await response.json()) as {
        detail?: string
        message?: string
      }
      if (errorBody.detail) detail = String(errorBody.detail)
      else if (errorBody.message) detail = errorBody.message
    } catch {
      // ignore JSON parse failures on error bodies
    }
    throw new ApiError(detail, response.status)
  }

  const parseStarted = performance.now()
  let data: unknown
  try {
    data = await response.json()
  } catch {
    throw new ApiError('Detection API returned an invalid response.')
  }
  const parseMs = performance.now() - parseStarted
  const requestTotalMs = performance.now() - requestStarted

  if (!isValidDetectionResponse(data)) {
    throw new ApiError('Detection API returned an unexpected response format.')
  }

  const modelImgsz =
    typeof (data as DetectionResponse & { model_imgsz?: number }).model_imgsz ===
    'number'
      ? (data as DetectionResponse & { model_imgsz?: number }).model_imgsz!
      : null

  const serverInferenceMs = data.inference_ms
  const nonInferenceOverheadMs = Math.max(0, requestTotalMs - serverInferenceMs)

  return {
    response: data,
    timing: {
      requestTotalMs,
      parseMs,
      serverInferenceMs,
      nonInferenceOverheadMs,
      modelImgsz,
    },
  }
}
