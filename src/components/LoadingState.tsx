interface LoadingStateProps {
  message?: string
}

export function LoadingState({
  message = 'Analyzing aerial image…',
}: LoadingStateProps) {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-state__spinner" aria-hidden="true" />
      <p className="loading-state__message">{message}</p>
      <p className="loading-state__sub">
        Running YOLO26s browser ONNX detection on the uploaded image.
      </p>
    </div>
  )
}
