import { ONNX_MODEL_URL } from './onnxModel'
import type { IsolatedInferenceOptions, OnnxSessionLike } from './onnxInference'

/**
 * Production WebGPU session for live camera.
 * Initialization matches the WEB-ONNX-05C benchmark path:
 * requestAdapter() must succeed, then one onnxruntime-web/webgpu session.
 * Static upload keeps the separate WASM session in onnxModel.ts.
 */

export const LIVE_WEBGPU_IMGSZ = 960

export type WebGpuRuntimeStatus = 'loading' | 'ready' | 'unavailable' | 'error'

export interface WebGpuRuntimeSnapshot {
  status: WebGpuRuntimeStatus
  errorMessage: string | null
  adapterInfo: string | null
  sessionIdentity: number
  modelUrl: string
  /** InferenceSession.create calls for this runtime. Stays 1 after a successful init. */
  sessionCreations: number
  /** Model fetches performed by this runtime's session creation (not per frame). */
  modelFetchCount: number
}

interface RuntimeHandle {
  session: OnnxSessionLike
  sessionIdentity: number
  createTensor: IsolatedInferenceOptions['createTensor']
}

export interface WebGpuInitResult {
  snapshot: WebGpuRuntimeSnapshot
  createdThisCall: boolean
}

interface GpuAdapterInfoLike {
  vendor?: string
  architecture?: string
  device?: string
  description?: string
}

interface GpuAdapterLike {
  info?: GpuAdapterInfoLike
  requestAdapterInfo?: () => Promise<GpuAdapterInfoLike>
}

interface GpuLike {
  requestAdapter: () => Promise<GpuAdapterLike | null>
}

let status: WebGpuRuntimeStatus = 'loading'
let errorMessage: string | null = null
let adapterInfo: string | null = null
let runtime: RuntimeHandle | null = null
let sessionCreations = 0
let modelFetchCount = 0
let initPromise: Promise<WebGpuInitResult> | null = null
let webGpuInferenceSlots = 0

const listeners = new Set<(snapshot: WebGpuRuntimeSnapshot) => void>()

function getSnapshot(): WebGpuRuntimeSnapshot {
  return {
    status,
    errorMessage,
    adapterInfo,
    sessionIdentity: runtime?.sessionIdentity ?? 0,
    modelUrl: ONNX_MODEL_URL,
    sessionCreations,
    modelFetchCount,
  }
}

function notify(): void {
  const snapshot = getSnapshot()
  for (const listener of listeners) {
    listener(snapshot)
  }
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

function getNavigatorGpu(): GpuLike | null {
  const gpu = (navigator as Navigator & { gpu?: GpuLike }).gpu
  return gpu ?? null
}

export function hasNavigatorGpu(): boolean {
  return getNavigatorGpu() != null
}

async function readAdapterInfo(adapter: GpuAdapterLike): Promise<string> {
  try {
    const info =
      adapter.info ??
      (adapter.requestAdapterInfo
        ? await adapter.requestAdapterInfo()
        : undefined)
    if (!info) return 'adapter present, info unavailable'
    const parts = [info.vendor, info.architecture, info.device, info.description]
      .filter((part) => typeof part === 'string' && part.length > 0)
    return parts.length > 0 ? parts.join(' · ') : 'adapter present, info empty'
  } catch (error) {
    return `adapter present, info failed: ${errorText(error)}`
  }
}

async function withModelFetchCount<T>(
  work: () => Promise<T>,
): Promise<{ result: T; fetches: number }> {
  let fetches = 0
  const original = window.fetch.bind(window)
  const wrapped: typeof window.fetch = (input, init) => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : input.url
    if (url.includes('yolo26s_1280_mixed_v1.onnx')) fetches += 1
    return original(input, init)
  }
  window.fetch = wrapped
  try {
    const result = await work()
    return { result, fetches }
  } finally {
    if (window.fetch === wrapped) window.fetch = original
  }
}

export function getWebGpuRuntimeSnapshot(): WebGpuRuntimeSnapshot {
  return getSnapshot()
}

export function subscribeWebGpuRuntime(
  listener: (snapshot: WebGpuRuntimeSnapshot) => void,
): () => void {
  listeners.add(listener)
  listener(getSnapshot())
  return () => {
    listeners.delete(listener)
  }
}

export function getWebGpuInferenceOptions(): Omit<
  IsolatedInferenceOptions,
  'modelImgsz'
> | null {
  if (!runtime) return null
  return {
    session: runtime.session,
    sessionIdentity: runtime.sessionIdentity,
    createTensor: runtime.createTensor,
  }
}

export function getWebGpuSessionCreations(): number {
  return sessionCreations
}

export function getWebGpuModelFetchCount(): number {
  return modelFetchCount
}

/**
 * Non-blocking slot for the shared WebGPU session.
 * Callers drop the frame when this returns false. Do not queue.
 */
export function tryBeginWebGpuInference(): boolean {
  if (webGpuInferenceSlots >= 1) return false
  webGpuInferenceSlots += 1
  return true
}

export function endWebGpuInference(): void {
  webGpuInferenceSlots = Math.max(0, webGpuInferenceSlots - 1)
}

export function getWebGpuInferenceSlots(): number {
  return webGpuInferenceSlots
}

function failInit(
  nextStatus: 'unavailable' | 'error',
  reason: string,
  info: string | null,
): WebGpuInitResult {
  runtime = null
  status = nextStatus
  errorMessage = reason
  adapterInfo = info
  notify()
  return { snapshot: getSnapshot(), createdThisCall: false }
}

/**
 * Create the WebGPU InferenceSession once.
 * Concurrent callers share the same promise. A failed init is not retried
 * automatically.
 */
export function initWebGpuRuntime(): Promise<WebGpuInitResult> {
  if (runtime && status === 'ready') {
    return Promise.resolve({ snapshot: getSnapshot(), createdThisCall: false })
  }
  if (initPromise) return initPromise

  status = 'loading'
  errorMessage = null
  notify()

  initPromise = (async (): Promise<WebGpuInitResult> => {
    const gpu = getNavigatorGpu()
    if (!gpu) {
      return failInit(
        'unavailable',
        'navigator.gpu is not available in this browser.',
        null,
      )
    }

    let adapter: GpuAdapterLike | null
    try {
      adapter = await gpu.requestAdapter()
    } catch (error) {
      return failInit(
        'unavailable',
        `navigator.gpu.requestAdapter() threw: ${errorText(error)}`,
        null,
      )
    }
    if (!adapter) {
      return failInit(
        'unavailable',
        'navigator.gpu.requestAdapter() returned null.',
        null,
      )
    }

    const info = await readAdapterInfo(adapter)
    adapterInfo = info

    try {
      const webgpu = await import('onnxruntime-web/webgpu')
      // Same host settings as WEB-ONNX-05C. This does not enable WASM
      // multithreading and avoids a false WebGPU failure when COOP/COEP is absent.
      webgpu.env.wasm.numThreads = 1
      webgpu.env.wasm.proxy = false

      const created = await withModelFetchCount(() =>
        webgpu.InferenceSession.create(ONNX_MODEL_URL, {
          executionProviders: ['webgpu'],
        }),
      )
      const session = created.result
      modelFetchCount += created.fetches
      sessionCreations += 1
      runtime = {
        session: session as unknown as OnnxSessionLike,
        sessionIdentity: sessionCreations,
        createTensor: (data, dims) => new webgpu.Tensor('float32', data, dims),
      }
      status = 'ready'
      errorMessage = null
      adapterInfo = info
      notify()

      if (import.meta.env.DEV) {
        console.info('[onnx-webgpu] InferenceSession ready', {
          modelUrl: ONNX_MODEL_URL,
          executionProvider: 'webgpu',
          liveImgsz: LIVE_WEBGPU_IMGSZ,
          sessionIdentity: runtime.sessionIdentity,
          sessionCreations,
          modelFetchCount,
          adapterInfo: info,
          inputNames: session.inputNames,
          outputNames: session.outputNames,
        })
      }

      return { snapshot: getSnapshot(), createdThisCall: true }
    } catch (error) {
      return failInit('error', errorText(error), info)
    }
  })()

  return initPromise
}
