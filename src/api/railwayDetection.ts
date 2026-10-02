import { getDetectEndpointUrl } from '../lib/detectionApiConfig'
import type { DetectionResponse } from '../types/detection'

const DETECT_TIMEOUT_MS = 120_000

export class RailwayDetectionError extends Error {
  status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'RailwayDetectionError'
    this.status = status
  }
}

export interface RailwayDetectOptions {
  filename?: string
  signal?: AbortSignal
}

export interface RailwayDetectTiming {
  requestTotalMs: number
  parseMs: number
  serverInferenceMs: number
  nonInferenceOverheadMs: number
  modelImgsz: number | null
}

export interface RailwayDetectResult {
  response: DetectionResponse
  timing: RailwayDetectTiming
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
      throw new RailwayDetectionError('Detection request was cancelled.')
    }
    options.signal.addEventListener('abort', onExternalAbort, { once: true })
  }

  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (options.signal?.aborted) {
        throw new RailwayDetectionError('Detection request was cancelled.')
      }
      throw new RailwayDetectionError(
        `Cloud detection timed out after ${Math.round(timeoutMs / 1000)}s.`,
      )
    }
    throw new RailwayDetectionError(
      'Unable to reach the cloud detection API. Check your connection and try again.',
    )
  } finally {
    window.clearTimeout(timer)
    options.signal?.removeEventListener('abort', onExternalAbort)
  }
}

/**
 * POST a captured camera frame to Railway `/detect`.
 * Does not set Content-Type — the browser supplies the multipart boundary.
 */
export async function detectPeopleOnRailway(
  file: File | Blob,
  options?: RailwayDetectOptions,
): Promise<RailwayDetectResult> {
  const endpoint = getDetectEndpointUrl()
  if (!endpoint) {
    throw new RailwayDetectionError(
      'Cloud detection URL is not configured (VITE_DETECTION_API_URL).',
    )
  }

  const formData = new FormData()
  const filename =
    options?.filename ?? (file instanceof File ? file.name : 'frame.jpg')
  formData.append('file', file, filename)

  const requestStarted = performance.now()
  const response = await fetchWithTimeout(
    endpoint,
    {
      method: 'POST',
      body: formData,
      signal: options?.signal,
    },
    DETECT_TIMEOUT_MS,
  )

  if (!response.ok) {
    let detail = `Cloud detection failed (${response.status}).`
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
    throw new RailwayDetectionError(detail, response.status)
  }

  const parseStarted = performance.now()
  let data: unknown
  try {
    data = await response.json()
  } catch {
    throw new RailwayDetectionError(
      'Cloud detection API returned an invalid response.',
    )
  }
  const parseMs = performance.now() - parseStarted
  const requestTotalMs = performance.now() - requestStarted

  if (!isValidDetectionResponse(data)) {
    throw new RailwayDetectionError(
      'Cloud detection API returned an unexpected response format.',
    )
  }

  const modelImgsz =
    typeof data.model_imgsz === 'number' ? data.model_imgsz : null
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
