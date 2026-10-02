import * as ort from 'onnxruntime-web/wasm'

export type OnnxModelStatus = 'loading' | 'ready' | 'error'

export interface OnnxModelSnapshot {
  status: OnnxModelStatus
  errorMessage: string | null
  modelUrl: string
  inputNames: readonly string[] | null
  outputNames: readonly string[] | null
}

export const ONNX_MODEL_URL = '/models/yolo26s_1280_mixed_v1.onnx'

let status: OnnxModelStatus = 'loading'
let errorMessage: string | null = null
let session: ort.InferenceSession | null = null
let sessionId = 0
let inputNames: readonly string[] | null = null
let outputNames: readonly string[] | null = null
let initPromise: Promise<ort.InferenceSession> | null = null
let runtimeConfigured = false

const listeners = new Set<(snapshot: OnnxModelSnapshot) => void>()

function getSnapshot(): OnnxModelSnapshot {
  return {
    status,
    errorMessage,
    modelUrl: ONNX_MODEL_URL,
    inputNames,
    outputNames,
  }
}

function notify(): void {
  const snapshot = getSnapshot()
  for (const listener of listeners) {
    listener(snapshot)
  }
}

function configureWasmRuntime(): void {
  if (runtimeConfigured) return
  // Static-upload session stays single-thread WASM. Live camera uses onnxWebGpu.ts.
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  runtimeConfigured = true
}

export function getOnnxModelSnapshot(): OnnxModelSnapshot {
  return getSnapshot()
}

export function subscribeOnnxModel(
  listener: (snapshot: OnnxModelSnapshot) => void,
): () => void {
  listeners.add(listener)
  listener(getSnapshot())
  return () => {
    listeners.delete(listener)
  }
}

/**
 * Create the browser InferenceSession once. Concurrent callers and React
 * StrictMode remounts share the same Promise / session.
 *
 * Does not run inference.
 */
export function initOnnxModel(): Promise<ort.InferenceSession> {
  if (session) {
    return Promise.resolve(session)
  }
  if (initPromise) {
    return initPromise
  }

  status = 'loading'
  errorMessage = null
  inputNames = null
  outputNames = null
  notify()

  initPromise = (async () => {
    configureWasmRuntime()

    const created = await ort.InferenceSession.create(ONNX_MODEL_URL, {
      executionProviders: ['wasm'],
    })

    session = created
    sessionId += 1
    inputNames = created.inputNames
    outputNames = created.outputNames
    status = 'ready'
    errorMessage = null
    notify()

    if (import.meta.env.DEV) {
      console.info('[onnx] InferenceSession ready', {
        modelUrl: ONNX_MODEL_URL,
        executionProvider: 'wasm',
        numThreads: 1,
        sessionId,
        inputNames: created.inputNames,
        outputNames: created.outputNames,
      })
    }

    return created
  })().catch((error: unknown) => {
    session = null
    sessionId = 0
    inputNames = null
    outputNames = null
    status = 'error'
    errorMessage =
      error instanceof Error
        ? error.message
        : typeof error === 'string'
          ? error
          : 'Failed to initialize the browser ONNX model.'
    // Keep initPromise non-null after failure so StrictMode / subscribers
    // do not restart a second simultaneous download automatically.
    notify()
    throw error instanceof Error
      ? error
      : new Error(errorMessage ?? 'ONNX model initialization failed.')
  })

  return initPromise
}

/** Available after a successful init. Reused by browser inference. */
export function getOnnxSession(): ort.InferenceSession | null {
  return session
}

/** Stable id that increments only when a new session is created. */
export function getOnnxSessionId(): number {
  return sessionId
}
