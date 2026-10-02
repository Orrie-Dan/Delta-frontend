import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { ConfidenceFilter } from './components/ConfidenceFilter'
import { DetectionStats } from './components/DetectionStats'
import { DetectionViewer } from './components/DetectionViewer'
import { ErrorState } from './components/ErrorState'
import { Header } from './components/Header'
import { ImageUploader } from './components/ImageUploader'
import { LiveCameraPanel } from './components/LiveCameraPanel'
import { LoadingState } from './components/LoadingState'
import { ModeSwitcher } from './components/ModeSwitcher'
import { runBrowserOnnxInferenceFromFile } from './lib/onnxInference'
import {
  getOnnxModelSnapshot,
  initOnnxModel,
  subscribeOnnxModel,
  type OnnxModelSnapshot,
} from './lib/onnxModel'
import {
  getWebGpuRuntimeSnapshot,
  initWebGpuRuntime,
  subscribeWebGpuRuntime,
  type WebGpuRuntimeSnapshot,
} from './lib/onnxWebGpu'
import type {
  AppPhase,
  DetectionResponse,
  WorkspaceMode,
} from './types/detection'
import './styles.css'

const DEFAULT_CONFIDENCE = 0.25

const BrowserOnnxTestPanel = lazy(() =>
  import('./components/BrowserOnnxTestPanel').then((module) => ({
    default: module.BrowserOnnxTestPanel,
  })),
)

const WebGpuCameraDebugPanel = lazy(() =>
  import('./components/WebGpuCameraDebugPanel').then((module) => ({
    default: module.WebGpuCameraDebugPanel,
  })),
)

/** Debug ONNX / WebGPU panels; enable with ?debug=onnx */
function useOnnxDebugPanel(): boolean {
  return useMemo(() => {
    if (typeof window === 'undefined') return false
    return new URLSearchParams(window.location.search).get('debug') === 'onnx'
  }, [])
}

function App() {
  const showOnnxDebugPanel = useOnnxDebugPanel()
  const [browserModel, setBrowserModel] = useState<OnnxModelSnapshot>(() =>
    getOnnxModelSnapshot(),
  )
  const [webGpu, setWebGpu] = useState<WebGpuRuntimeSnapshot>(() =>
    getWebGpuRuntimeSnapshot(),
  )
  const [mode, setMode] = useState<WorkspaceMode>('upload')
  const [phase, setPhase] = useState<AppPhase>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] =
    useState(DEFAULT_CONFIDENCE)
  const previewUrlRef = useRef<string | null>(null)
  const detectingRef = useRef(false)

  const browserReady = browserModel.status === 'ready'
  const browserLoading = browserModel.status === 'loading'
  const browserError = browserModel.status === 'error'

  const revokePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }
  }, [])

  useEffect(() => {
    const unsubscribe = subscribeOnnxModel(setBrowserModel)
    void initOnnxModel().catch(() => {
      // Status/error are published via subscribeOnnxModel.
    })
    return unsubscribe
  }, [])

  useEffect(() => {
    const unsubscribe = subscribeWebGpuRuntime(setWebGpu)
    // Start WebGPU after the static WASM fetch so each runtime records its own
    // model load. Live inference still reuses the one WebGPU session.
    void initOnnxModel()
      .catch(() => undefined)
      .finally(() => {
        void initWebGpuRuntime().catch(() => {
          // Status/error are published via subscribeWebGpuRuntime.
        })
      })
    return unsubscribe
  }, [])

  useEffect(() => {
    return () => {
      revokePreview()
    }
  }, [revokePreview])

  const filteredDetections = useMemo(() => {
    if (!result) return []
    return result.detections.filter((d) => d.confidence >= confidenceThreshold)
  }, [result, confidenceThreshold])

  const handleModeChange = useCallback((nextMode: WorkspaceMode) => {
    setMode(nextMode)
  }, [])

  const handleImageSelected = useCallback(
    (selected: File) => {
      revokePreview()
      const url = URL.createObjectURL(selected)
      previewUrlRef.current = url
      setFile(selected)
      setPreviewUrl(url)
      setResult(null)
      setErrorMessage(null)
      setConfidenceThreshold(DEFAULT_CONFIDENCE)
      setPhase('preview')
    },
    [revokePreview],
  )

  const runDetection = useCallback(async () => {
    if (!file || detectingRef.current) return
    if (!browserReady) {
      setErrorMessage(
        browserError
          ? (browserModel.errorMessage ??
            'Browser ONNX model failed to load.')
          : 'Browser ONNX model is still loading. Please wait.',
      )
      setPhase('error')
      return
    }

    detectingRef.current = true
    setPhase('detecting')
    setErrorMessage(null)

    try {
      const { response, timing, diagnostics } =
        await runBrowserOnnxInferenceFromFile(file)

      console.info('[upload] browser ONNX detection', {
        people: response.people,
        detections: response.detections,
        timing,
        letterbox: diagnostics.letterbox,
        sessionIdentity: diagnostics.sessionIdentity,
      })

      setResult(response)
      setPhase('results')
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'An unexpected error occurred during browser detection.'
      setErrorMessage(message)
      setPhase('error')
    } finally {
      detectingRef.current = false
    }
  }, [browserError, browserModel.errorMessage, browserReady, file])

  const handleReset = useCallback(() => {
    revokePreview()
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setErrorMessage(null)
    setConfidenceThreshold(DEFAULT_CONFIDENCE)
    setPhase('upload')
  }, [revokePreview])

  return (
    <div className="app-shell">
      <Header browserModel={browserModel} webGpu={webGpu} />

      <main className="app-main">
        <section className="intro">
          <h2 className="intro__title">Detect People in Aerial Imagery</h2>
          <p className="intro__description">
            Upload a drone or aerial image for browser WASM detection, or open
            the live camera for WebGPU (or cloud CPU fallback when WebGPU is
            unavailable).
          </p>
          <div className="model-badge" aria-label="Model details">
            YOLO26s • VisDrone • Browser ONNX • WebGPU live camera
          </div>
        </section>

        <ModeSwitcher mode={mode} onChange={handleModeChange} />

        {showOnnxDebugPanel && (
          <Suspense
            fallback={
              <p className="inline-warning" role="status">
                Loading browser ONNX debug tools…
              </p>
            }
          >
            <BrowserOnnxTestPanel browserModel={browserModel} />
            <WebGpuCameraDebugPanel />
          </Suspense>
        )}

        {mode === 'live' && (
          <LiveCameraPanel
            active={mode === 'live'}
            webGpu={webGpu}
            confidenceThreshold={confidenceThreshold}
            onConfidenceChange={setConfidenceThreshold}
          />
        )}

        {mode === 'upload' && phase === 'upload' && (
          <section className="panel panel--upload" aria-label="Image upload">
            <ImageUploader onImageSelected={handleImageSelected} />
            {browserLoading && (
              <p className="inline-warning" role="status">
                Loading browser ONNX model… detection will unlock when ready.
              </p>
            )}
            {browserError && (
              <p className="inline-warning" role="alert">
                Browser model failed to load
                {browserModel.errorMessage
                  ? `: ${browserModel.errorMessage}`
                  : '.'}
              </p>
            )}
          </section>
        )}

        {mode === 'upload' && phase === 'preview' && previewUrl && (
          <section className="panel" aria-label="Image preview">
            <div className="preview-layout">
              <div className="preview-frame">
                <img
                  src={previewUrl}
                  alt="Selected aerial image preview"
                  className="preview-frame__image"
                />
              </div>

              <div className="action-bar">
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={() => void runDetection()}
                  disabled={!browserReady}
                >
                  {browserLoading ? 'Loading model…' : 'Detect People'}
                </button>
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={handleReset}
                >
                  Choose Another Image
                </button>
              </div>

              {!browserReady && (
                <p className="inline-warning" role="status">
                  {browserError
                    ? 'Browser ONNX model is unavailable. Fix model loading before detecting.'
                    : 'Waiting for the browser ONNX model.'}
                </p>
              )}
            </div>
          </section>
        )}

        {mode === 'upload' && phase === 'detecting' && previewUrl && (
          <section className="panel" aria-label="Detection in progress">
            <div className="preview-frame preview-frame--muted">
              <img
                src={previewUrl}
                alt="Aerial image being analyzed"
                className="preview-frame__image"
              />
              <div className="preview-frame__overlay">
                <LoadingState />
              </div>
            </div>
          </section>
        )}

        {mode === 'upload' && phase === 'error' && (
          <section className="panel" aria-label="Detection error">
            {previewUrl && (
              <div className="preview-frame preview-frame--muted">
                <img
                  src={previewUrl}
                  alt="Aerial image awaiting retry"
                  className="preview-frame__image"
                />
              </div>
            )}
            <ErrorState
              message={
                errorMessage ??
                'Detection failed. Please try again or choose another image.'
              }
              onRetry={() => void runDetection()}
              onReset={handleReset}
            />
          </section>
        )}

        {mode === 'upload' && phase === 'results' && previewUrl && result && (
          <section className="panel panel--results" aria-label="Detection results">
            <div className="results-toolbar">
              <ConfidenceFilter
                value={confidenceThreshold}
                onChange={setConfidenceThreshold}
              />
              <div className="action-bar action-bar--compact">
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={handleReset}
                >
                  Analyze Another Image
                </button>
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={() => void runDetection()}
                  disabled={!browserReady}
                >
                  Re-run Detection
                </button>
              </div>
            </div>

            <DetectionViewer
              imageUrl={previewUrl}
              result={result}
              filteredDetections={filteredDetections}
              confidenceThreshold={confidenceThreshold}
            />

            <DetectionStats
              result={result}
              filteredDetections={filteredDetections}
              modelName="YOLO26s (browser ONNX)"
            />
          </section>
        )}
      </main>

      <footer className="app-footer">
        <span>Person Detection MVP</span>
        <span>YOLO26s · Browser ONNX · WebGPU live camera</span>
      </footer>
    </div>
  )
}

export default App
