import { isDetectionApiConfigured } from './detectionApiConfig'
import {
  getWebGpuInferenceOptions,
  getWebGpuRuntimeSnapshot,
  type WebGpuRuntimeStatus,
} from './onnxWebGpu'

export type SelectedLiveRuntime =
  | 'webgpu'
  | 'railway'
  | 'unavailable'
  | 'pending'

interface NavigatorUADataLike {
  mobile?: boolean
}

/**
 * Pragmatic mobile/tablet classification for live-runtime policy.
 * Treats User-Agent Client Hints `mobile === true` as authoritative when
 * present, then falls back to UA strings (needed for UA overrides / iPadOS).
 */
export function isMobileDevice(): boolean {
  if (typeof navigator === 'undefined') return false

  const uaData = (
    navigator as Navigator & { userAgentData?: NavigatorUADataLike }
  ).userAgentData
  // Only trust an affirmative Client Hint. A `false` value must not block UA
  // fallbacks (DevTools UA override leaves userAgentData.mobile=false).
  if (uaData?.mobile === true) return true

  const ua = navigator.userAgent || ''
  if (/Android|iPhone|iPod/i.test(ua)) return true
  // Classic iPad UA
  if (/iPad/i.test(ua)) return true
  // iPadOS 13+ often reports as Macintosh but exposes touch
  if (
    /Macintosh/i.test(ua) &&
    typeof navigator.maxTouchPoints === 'number' &&
    navigator.maxTouchPoints > 1
  ) {
    return true
  }

  return false
}

/**
 * Dev/debug force of Railway fallback without changing WebGPU feature detection.
 * Enabled via prop, `?forceLiveFallback=1`, or `window.__forceLiveRailwayFallback`.
 */
export function isForceLiveRailwayFallback(propForce?: boolean): boolean {
  if (propForce) return true
  if (typeof window === 'undefined') return false
  const params = new URLSearchParams(window.location.search)
  if (params.get('forceLiveFallback') === '1') return true
  return Boolean(
    (window as Window & { __forceLiveRailwayFallback?: boolean })
      .__forceLiveRailwayFallback,
  )
}

/** Production live policy: mobile/tablet never selects WebGPU. */
export function shouldInitializeWebGpuRuntime(): boolean {
  return !isMobileDevice()
}

export interface SelectLiveRuntimeOptions {
  forceRailway?: boolean
  webGpuStatus?: WebGpuRuntimeStatus
  /** True when a usable WebGPU InferenceSession exists. */
  webGpuSessionReady?: boolean
  cloudConfigured?: boolean
}

/**
 * Select the live inference runtime.
 *
 * Policy:
 * - Mobile/tablet → Railway (when configured)
 * - Desktop + ready WebGPU session → WebGPU
 * - Desktop without usable WebGPU → Railway
 * - Desktop WebGPU still loading → pending
 * - No Railway URL and no eligible WebGPU → unavailable
 */
export function selectLiveRuntime(
  options: SelectLiveRuntimeOptions = {},
): SelectedLiveRuntime {
  const forceRailway = isForceLiveRailwayFallback(options.forceRailway)
  const cloudOk =
    options.cloudConfigured ?? isDetectionApiConfigured()
  const webGpuStatus =
    options.webGpuStatus ?? getWebGpuRuntimeSnapshot().status
  const webGpuSessionReady =
    options.webGpuSessionReady ??
    (webGpuStatus === 'ready' && getWebGpuInferenceOptions() != null)

  if (forceRailway || isMobileDevice()) {
    return cloudOk ? 'railway' : 'unavailable'
  }

  if (webGpuSessionReady) return 'webgpu'
  if (webGpuStatus === 'loading') return 'pending'
  return cloudOk ? 'railway' : 'unavailable'
}

export function selectedRuntimeToLiveLabel(
  runtime: SelectedLiveRuntime,
): 'webgpu' | 'railway' | 'none' {
  if (runtime === 'webgpu') return 'webgpu'
  if (runtime === 'railway') return 'railway'
  return 'none'
}
