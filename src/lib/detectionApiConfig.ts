/**
 * Railway cloud detection API base URL for live-camera fallback when WebGPU
 * is unavailable. Trailing slashes are stripped so `/detect` never becomes
 * `//detect`.
 */
export function getDetectionApiBaseUrl(): string | null {
  const raw = import.meta.env.VITE_DETECTION_API_URL
  if (typeof raw !== 'string') return null
  const trimmed = raw.trim().replace(/\/+$/, '')
  return trimmed.length > 0 ? trimmed : null
}

export function isDetectionApiConfigured(): boolean {
  return getDetectionApiBaseUrl() != null
}

export function getDetectEndpointUrl(): string | null {
  const base = getDetectionApiBaseUrl()
  return base ? `${base}/detect` : null
}
