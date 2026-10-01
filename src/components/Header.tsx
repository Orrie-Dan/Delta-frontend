import type { ApiStatus, HealthResponse } from '../types/detection'

interface HeaderProps {
  apiStatus: ApiStatus
  health: HealthResponse | null
  onRetryHealth: () => void
}

export function Header({ apiStatus, health, onRetryHealth }: HeaderProps) {
  const statusLabel =
    apiStatus === 'checking'
      ? 'Checking model…'
      : apiStatus === 'online'
        ? 'Model Online'
        : 'Model Unavailable'

  return (
    <header className="app-header">
      <div className="app-header__brand">
        <h1 className="app-header__title">Person Detection</h1>
        <span className="app-header__badge">AI-Powered Aerial Analysis</span>
      </div>

      <div className="app-header__status">
        <button
          type="button"
          className={`status-pill status-pill--${apiStatus}`}
          onClick={apiStatus === 'offline' ? onRetryHealth : undefined}
          disabled={apiStatus === 'checking'}
          aria-live="polite"
          title={
            apiStatus === 'offline'
              ? 'Click to retry health check'
              : health
                ? `${health.model} · ${health.image_size}px`
                : undefined
          }
        >
          <span className="status-pill__dot" aria-hidden="true" />
          <span>{statusLabel}</span>
        </button>
      </div>
    </header>
  )
}
