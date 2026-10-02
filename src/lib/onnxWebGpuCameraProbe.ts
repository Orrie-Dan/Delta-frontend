import {
  ensureWebGpuInferenceRuntime,
  getCachedWebGpuAdapterInfo,
  getWebGpuInferenceRuntimeOptions,
} from './onnxBenchmark'
import { runIsolatedBrowserInference } from './onnxInference'
import {
  endWebGpuInference,
  tryBeginWebGpuInference,
} from './onnxWebGpu'
import { median, percentile95 } from './stats'

/** Debug-only WebGPU live-camera comparison (1280 then 960). */
export const WEBGPU_CAMERA_IMGSZ_SEQUENCE = [1280, 960] as const
export const DEFAULT_PHASE_DURATION_MS = 30_000
export const MIN_PHASE_DURATION_MS = 15_000
export const MAX_PHASE_DURATION_MS = 60_000

export type WebGpuCameraImgsz = (typeof WEBGPU_CAMERA_IMGSZ_SEQUENCE)[number]

export interface WebGpuCameraFrameSample {
  timestamp: number
  preprocessMs: number
  inferenceMs: number
  postprocessMs: number
  totalMs: number
  detections: number
}

export interface WebGpuCameraPhaseSummary {
  imgsz: WebGpuCameraImgsz
  plannedDurationMs: number
  actualDurationMs: number
  loopTicks: number
  framesDropped: number
  framesCompleted: number
  /** Completed inferences per second over the phase wall time. */
  endToEndFps: number
  /** 1000 / median(session.run ms) from completed frames. */
  inferenceFpsFromMedianRun: number
  /** Same as endToEndFps; explicit alias for reports. */
  inferenceFpsAchieved: number
  medianPreprocessMs: number
  medianInferenceMs: number
  medianPostprocessMs: number
  medianTotalMs: number
  p95TotalMs: number | null
  minTotalMs: number
  maxTotalMs: number
  detectionMin: number
  detectionMax: number
  detectionMedian: number
  detectionStdDev: number
  uniqueDetectionCounts: number
  detectionCounts: number[]
  cameraLoopFps: number
  longTaskCount: number
  longTaskTotalMs: number
  heapMbStart: number | null
  heapMbEnd: number | null
  heapMbPeak: number | null
  heapSamplesMb: number[]
  samples: WebGpuCameraFrameSample[]
}

export interface WebGpuCameraProbeReport {
  sessionIdentity: number
  adapterInfo: string | null
  webglRenderer: string | null
  hardwareConcurrency: number
  phaseDurationMs: number
  phases: WebGpuCameraPhaseSummary[]
  totalWallMs: number
  aborted: boolean
  error: string | null
}

export interface WebGpuCameraProbeProgress {
  phaseIndex: number
  phaseCount: number
  imgsz: WebGpuCameraImgsz
  elapsedMs: number
  phaseDurationMs: number
  loopTicks: number
  framesDropped: number
  framesCompleted: number
  lastDetections: number | null
  lastTotalMs: number | null
  message: string
}

export interface RunWebGpuCameraProbeOptions {
  video: HTMLVideoElement
  phaseDurationMs?: number
  imgszSequence?: readonly WebGpuCameraImgsz[]
  onProgress?: (progress: WebGpuCameraProbeProgress) => void
  signal?: AbortSignal
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

function stdDev(values: number[]): number {
  if (values.length === 0) return 0
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length
  const variance =
    values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

function attachLongTaskObserver(
  onLongTask: (durationMs: number) => void,
): PerformanceObserver | null {
  if (typeof PerformanceObserver === 'undefined') return null
  try {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        onLongTask(entry.duration)
      }
    })
    observer.observe({ entryTypes: ['longtask'] as string[] })
    return observer
  } catch {
    return null
  }
}

async function runPhase(
  video: HTMLVideoElement,
  imgsz: WebGpuCameraImgsz,
  phaseDurationMs: number,
  phaseIndex: number,
  phaseCount: number,
  runtime: NonNullable<ReturnType<typeof getWebGpuInferenceRuntimeOptions>>,
  onProgress: (progress: WebGpuCameraProbeProgress) => void,
  signal: AbortSignal,
): Promise<WebGpuCameraPhaseSummary> {
  let loopTicks = 0
  let framesDropped = 0
  let framesCompleted = 0
  let inFlight = false
  let lastDetections: number | null = null
  let lastTotalMs: number | null = null
  const samples: WebGpuCameraFrameSample[] = []
  const heapSamplesMb: number[] = []
  let heapPeak: number | null = null
  let longTaskCount = 0
  let longTaskTotalMs = 0

  const recordHeap = () => {
    const heap = usedHeapMb()
    if (heap == null) return
    heapSamplesMb.push(heap)
    if (heapPeak == null || heap > heapPeak) heapPeak = heap
  }

  const longTaskObserver = attachLongTaskObserver((durationMs) => {
    longTaskCount += 1
    longTaskTotalMs += durationMs
  })

  const phaseStart = performance.now()
  const phaseEndAt = phaseStart + phaseDurationMs
  recordHeap()
  const heapStart = heapSamplesMb[0] ?? null

  let heapIntervalId = window.setInterval(recordHeap, 2000)

  await new Promise<void>((resolve) => {
    let rafId = 0

    const finish = () => {
      window.clearInterval(heapIntervalId)
      longTaskObserver?.disconnect()
      cancelAnimationFrame(rafId)
      resolve()
    }

    const tick = () => {
      if (signal.aborted || performance.now() >= phaseEndAt) {
        finish()
        return
      }

      const now = performance.now()
      const elapsedMs = now - phaseStart

      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        rafId = requestAnimationFrame(tick)
        return
      }

      loopTicks += 1

      if (inFlight || !tryBeginWebGpuInference()) {
        framesDropped += 1
        onProgress({
          phaseIndex,
          phaseCount,
          imgsz,
          elapsedMs,
          phaseDurationMs,
          loopTicks,
          framesDropped,
          framesCompleted,
          lastDetections,
          lastTotalMs,
          message: `imgsz ${imgsz} · dropped frame (inference in flight)`,
        })
        rafId = requestAnimationFrame(tick)
        return
      }

      inFlight = true
      onProgress({
        phaseIndex,
        phaseCount,
        imgsz,
        elapsedMs,
        phaseDurationMs,
        loopTicks,
        framesDropped,
        framesCompleted,
        lastDetections,
        lastTotalMs,
        message: `imgsz ${imgsz} · running inference`,
      })

      void (async () => {
        try {
          const result = await runIsolatedBrowserInference(video, {
            ...runtime,
            modelImgsz: imgsz,
          })
          framesCompleted += 1
          lastDetections = result.response.people
          lastTotalMs = result.timing.totalMs
          samples.push({
            timestamp: performance.now(),
            preprocessMs: result.timing.preprocessMs,
            inferenceMs: result.timing.inferenceMs,
            postprocessMs: result.timing.postprocessMs,
            totalMs: result.timing.totalMs,
            detections: result.response.people,
          })
        } catch {
          // Keep the loop alive; do not queue a retry for this tick.
        } finally {
          inFlight = false
          endWebGpuInference()
        }
      })()

      rafId = requestAnimationFrame(tick)
    }

    signal.addEventListener('abort', finish, { once: true })
    rafId = requestAnimationFrame(tick)
  })

  recordHeap()
  const phaseEnd = performance.now()
  const actualDurationMs = phaseEnd - phaseStart
  const durationSec = actualDurationMs / 1000
  const detectionCounts = samples.map((sample) => sample.detections)
  const inferenceMs = samples.map((sample) => sample.inferenceMs)
  const totalMs = samples.map((sample) => sample.totalMs)
  const medianInference = inferenceMs.length > 0 ? median(inferenceMs) : 0
  const endToEndFps =
    durationSec > 0 ? framesCompleted / durationSec : 0

  return {
    imgsz,
    plannedDurationMs: phaseDurationMs,
    actualDurationMs,
    loopTicks,
    framesDropped,
    framesCompleted,
    endToEndFps,
    inferenceFpsFromMedianRun:
      medianInference > 0 ? 1000 / medianInference : 0,
    inferenceFpsAchieved: endToEndFps,
    medianPreprocessMs:
      samples.length > 0
        ? median(samples.map((sample) => sample.preprocessMs))
        : 0,
    medianInferenceMs: medianInference,
    medianPostprocessMs:
      samples.length > 0
        ? median(samples.map((sample) => sample.postprocessMs))
        : 0,
    medianTotalMs: totalMs.length > 0 ? median(totalMs) : 0,
    p95TotalMs: totalMs.length >= 5 ? percentile95(totalMs) : null,
    minTotalMs: totalMs.length > 0 ? Math.min(...totalMs) : 0,
    maxTotalMs: totalMs.length > 0 ? Math.max(...totalMs) : 0,
    detectionMin:
      detectionCounts.length > 0 ? Math.min(...detectionCounts) : 0,
    detectionMax:
      detectionCounts.length > 0 ? Math.max(...detectionCounts) : 0,
    detectionMedian:
      detectionCounts.length > 0 ? median(detectionCounts) : 0,
    detectionStdDev: stdDev(detectionCounts),
    uniqueDetectionCounts: new Set(detectionCounts).size,
    detectionCounts,
    cameraLoopFps: durationSec > 0 ? loopTicks / durationSec : 0,
    longTaskCount,
    longTaskTotalMs,
    heapMbStart: heapStart,
    heapMbEnd: heapSamplesMb[heapSamplesMb.length - 1] ?? null,
    heapMbPeak: heapPeak,
    heapSamplesMb,
    samples,
  }
}

export async function runWebGpuCameraProbe(
  options: RunWebGpuCameraProbeOptions,
): Promise<WebGpuCameraProbeReport> {
  const phaseDurationMs = Math.min(
    MAX_PHASE_DURATION_MS,
    Math.max(
      MIN_PHASE_DURATION_MS,
      options.phaseDurationMs ?? DEFAULT_PHASE_DURATION_MS,
    ),
  )
  const imgszSequence =
    options.imgszSequence ?? WEBGPU_CAMERA_IMGSZ_SEQUENCE
  const onProgress = options.onProgress ?? (() => {})
  const signal = options.signal ?? new AbortController().signal

  const wallStart = performance.now()
  let aborted = false
  let error: string | null = null
  const phases: WebGpuCameraPhaseSummary[] = []

  const init = await ensureWebGpuInferenceRuntime()
  if (!init.ok) {
    return {
      sessionIdentity: 0,
      adapterInfo: init.adapterInfo,
      webglRenderer: webGlRenderer(),
      hardwareConcurrency: navigator.hardwareConcurrency,
      phaseDurationMs,
      phases: [],
      totalWallMs: 0,
      aborted: false,
      error: init.reason,
    }
  }

  const runtime = getWebGpuInferenceRuntimeOptions()
  if (!runtime) {
    return {
      sessionIdentity: init.sessionIdentity,
      adapterInfo: init.adapterInfo,
      webglRenderer: webGlRenderer(),
      hardwareConcurrency: navigator.hardwareConcurrency,
      phaseDurationMs,
      phases: [],
      totalWallMs: 0,
      aborted: false,
      error: 'WebGPU session was not available after initialization.',
    }
  }

  try {
    for (let index = 0; index < imgszSequence.length; index += 1) {
      if (signal.aborted) {
        aborted = true
        break
      }
      const imgsz = imgszSequence[index]!
      onProgress({
        phaseIndex: index,
        phaseCount: imgszSequence.length,
        imgsz,
        elapsedMs: 0,
        phaseDurationMs,
        loopTicks: 0,
        framesDropped: 0,
        framesCompleted: 0,
        lastDetections: null,
        lastTotalMs: null,
        message: `Starting imgsz ${imgsz} phase (${Math.round(phaseDurationMs / 1000)}s)`,
      })
      phases.push(
        await runPhase(
          options.video,
          imgsz,
          phaseDurationMs,
          index,
          imgszSequence.length,
          runtime,
          onProgress,
          signal,
        ),
      )
    }
  } catch (err) {
    error =
      err instanceof Error ? err.message : 'WebGPU camera probe failed.'
    aborted = signal.aborted
  }

  const report: WebGpuCameraProbeReport = {
    sessionIdentity: init.sessionIdentity,
    adapterInfo: init.adapterInfo ?? getCachedWebGpuAdapterInfo(),
    webglRenderer: webGlRenderer(),
    hardwareConcurrency: navigator.hardwareConcurrency,
    phaseDurationMs,
    phases,
    totalWallMs: performance.now() - wallStart,
    aborted,
    error,
  }

  ;(
    window as Window & { __webGpuCameraProbeReport?: WebGpuCameraProbeReport }
  ).__webGpuCameraProbeReport = report

  return report
}
