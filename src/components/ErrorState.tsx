interface ErrorStateProps {
  message: string
  onRetry?: () => void
  onReset?: () => void
}

export function ErrorState({ message, onRetry, onReset }: ErrorStateProps) {
  return (
    <div className="error-state" role="alert">
      <h2 className="error-state__title">Detection unavailable</h2>
      <p className="error-state__message">{message}</p>
      <div className="error-state__actions">
        {onRetry && (
          <button type="button" className="btn btn--primary" onClick={onRetry}>
            Retry
          </button>
        )}
        {onReset && (
          <button type="button" className="btn btn--ghost" onClick={onReset}>
            Choose Another Image
          </button>
        )}
      </div>
    </div>
  )
}
