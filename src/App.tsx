import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, checkHealth, detectPeople } from './api/detection'
import { ConfidenceFilter } from './components/ConfidenceFilter'
import { DetectionStats } from './components/DetectionStats'
import { DetectionViewer } from './components/DetectionViewer'
import { ErrorState } from './components/ErrorState'
import { Header } from './components/Header'
import { ImageUploader } from './components/ImageUploader'
import { LiveCameraPanel } from './components/LiveCameraPanel'
import { LoadingState } from './components/LoadingState'
import { ModeSwitcher } from './components/ModeSwitcher'
import type {
  ApiStatus,
  AppPhase,
  DetectionResponse,
  HealthResponse,
  WorkspaceMode,
} from './types/detection'
import './styles.css'

const DEFAULT_CONFIDENCE = 0.25

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)
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

  const revokePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }
  }, [])

  const refreshHealth = useCallback(async () => {
    setApiStatus('checking')
    try {
      const response = await checkHealth()
      setHealth(response)
      setApiStatus('online')
    } catch {
      setHealth(null)
      setApiStatus('offline')
    }
  }, [])

  useEffect(() => {
    void refreshHealth()
  }, [refreshHealth])

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

    detectingRef.current = true
    setPhase('detecting')
    setErrorMessage(null)

    try {
      const response = await detectPeople(file)
      setResult(response)
      setPhase('results')
      if (apiStatus !== 'online') {
        void refreshHealth()
      }
    } catch (error) {
      const message =
        error instanceof ApiError
          ? error.message
          : 'An unexpected error occurred during detection.'
      setErrorMessage(message)
      setPhase('error')
    } finally {
      detectingRef.current = false
    }
  }, [apiStatus, file, refreshHealth])

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
      <Header
        apiStatus={apiStatus}
        health={health}
        onRetryHealth={() => void refreshHealth()}
      />

      <main className="app-main">
        <section className="intro">
          <h2 className="intro__title">Detect People in Aerial Imagery</h2>
          <p className="intro__description">
            Upload a drone or aerial image, or use live camera detection. The AI
            model identifies people and visualizes their locations.
          </p>
          <div className="model-badge" aria-label="Model details">
            YOLO26s • VisDrone • 1280px
          </div>
        </section>

        <ModeSwitcher mode={mode} onChange={handleModeChange} />

        {mode === 'live' && (
          <LiveCameraPanel
            active={mode === 'live'}
            apiStatus={apiStatus}
            confidenceThreshold={confidenceThreshold}
            onConfidenceChange={setConfidenceThreshold}
          />
        )}

        {mode === 'upload' && phase === 'upload' && (
          <section className="panel panel--upload" aria-label="Image upload">
            <ImageUploader onImageSelected={handleImageSelected} />
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
                  disabled={apiStatus === 'offline'}
                >
                  Detect People
                </button>
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={handleReset}
                >
                  Choose Another Image
                </button>
              </div>

              {apiStatus === 'offline' && (
                <p className="inline-warning" role="status">
                  Model appears offline. You can still select an image, then
                  retry when the service is available.
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
              modelName={health?.model ?? 'YOLO26s'}
            />
          </section>
        )}
      </main>

      <footer className="app-footer">
        <span>Person Detection MVP</span>
        <span>YOLO26s · Aerial person detection</span>
      </footer>
    </div>
  )
}

export default App
