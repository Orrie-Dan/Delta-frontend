import type { OnnxModelSnapshot } from '../lib/onnxModel'
import type { WebGpuRuntimeSnapshot } from '../lib/onnxWebGpu'

interface HeaderProps {
  browserModel: OnnxModelSnapshot
  webGpu: WebGpuRuntimeSnapshot
}

function webGpuLabel(status: WebGpuRuntimeSnapshot['status']): string {
  if (status === 'loading') return 'WebGPU loading…'
  if (status === 'ready') return 'WebGPU ready'
  if (status === 'unavailable') return 'WebGPU unavailable'
  return 'WebGPU error'
}

export function Header({ browserModel, webGpu }: HeaderProps) {
  const browserLabel =
    browserModel.status === 'loading'
      ? 'Browser model loading…'
      : browserModel.status === 'ready'
        ? 'Browser model ready'
        : 'Browser model unavailable'

  const browserTitle =
    browserModel.status === 'error'
      ? (browserModel.errorMessage ?? 'ONNX session failed to initialize')
      : browserModel.status === 'ready'
        ? `Static upload · ${browserModel.modelUrl} · WASM single-thread · imgsz 1280`
        : `Loading ${browserModel.modelUrl}`

  const webGpuTitle =
    webGpu.status === 'ready'
      ? `Live camera · WebGPU · imgsz 960 · ${webGpu.adapterInfo ?? 'adapter ready'} · session ${webGpu.sessionIdentity}`
      : (webGpu.errorMessage ?? 'Checking WebGPU live detection')

  return (
    <header className="app-header">
      <div className="app-header__brand">
        <h1 className="app-header__title">Person Detection</h1>
        <span className="app-header__badge">AI-Powered Aerial Analysis</span>
      </div>

      <div className="app-header__status">
        <div
          className={`status-pill status-pill--browser-${browserModel.status}`}
          role="status"
          aria-live="polite"
          title={browserTitle}
        >
          <span className="status-pill__dot" aria-hidden="true" />
          <span>{browserLabel}</span>
        </div>

        <div
          className={`status-pill status-pill--webgpu-${webGpu.status}`}
          role="status"
          aria-live="polite"
          title={webGpuTitle}
        >
          <span className="status-pill__dot" aria-hidden="true" />
          <span>{webGpuLabel(webGpu.status)}</span>
        </div>
      </div>
    </header>
  )
}
