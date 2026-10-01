import { useMemo, useState } from 'react'
import type { Detection, DetectionResponse } from '../types/detection'

interface DetectionStatsProps {
  result: DetectionResponse
  filteredDetections: Detection[]
  modelName?: string
}

const DETAILS_PAGE_SIZE = 50

export function DetectionStats({
  result,
  filteredDetections,
  modelName = 'YOLO26s',
}: DetectionStatsProps) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [visibleCount, setVisibleCount] = useState(DETAILS_PAGE_SIZE)

  const averageConfidence = useMemo(() => {
    if (filteredDetections.length === 0) return null
    const sum = filteredDetections.reduce((acc, d) => acc + d.confidence, 0)
    return sum / filteredDetections.length
  }, [filteredDetections])

  const inferenceSeconds = (result.inference_ms / 1000).toFixed(2)

  const visibleDetails = filteredDetections.slice(0, visibleCount)

  return (
    <section className="detection-stats" aria-label="Detection summary">
      <div className="stat-grid">
        <article className="stat-card">
          <p className="stat-card__label">People Detected</p>
          <p className="stat-card__value">{filteredDetections.length}</p>
          {filteredDetections.length !== result.people && (
            <p className="stat-card__meta">of {result.people} returned</p>
          )}
        </article>

        <article className="stat-card">
          <p className="stat-card__label">Inference Time</p>
          <p className="stat-card__value">{inferenceSeconds} s</p>
        </article>

        <article className="stat-card">
          <p className="stat-card__label">Model</p>
          <p className="stat-card__value stat-card__value--text">{modelName}</p>
        </article>

        <article className="stat-card">
          <p className="stat-card__label">Input Resolution</p>
          <p className="stat-card__value stat-card__value--text">
            {result.image.width} × {result.image.height}
          </p>
        </article>

        <article className="stat-card">
          <p className="stat-card__label">Average Confidence</p>
          <p className="stat-card__value">
            {averageConfidence === null
              ? '—'
              : `${(averageConfidence * 100).toFixed(1)}%`}
          </p>
        </article>
      </div>

      <div className="detection-details">
        <button
          type="button"
          className="detection-details__toggle"
          aria-expanded={detailsOpen}
          onClick={() => setDetailsOpen((open) => !open)}
        >
          Detection Details
          <span aria-hidden="true">{detailsOpen ? '−' : '+'}</span>
        </button>

        {detailsOpen && (
          <div className="detection-details__panel">
            {filteredDetections.length === 0 ? (
              <p className="detection-details__empty">
                No detections above the current confidence threshold.
              </p>
            ) : (
              <>
                <ul className="detection-details__list">
                  {visibleDetails.map((detection, index) => (
                    <li key={`${detection.x1}-${detection.y1}-${index}`}>
                      <span className="detection-details__index">
                        #{index + 1}
                      </span>
                      <span>
                        {(detection.confidence * 100).toFixed(1)}%
                      </span>
                      <span className="detection-details__coords">
                        ({detection.x1.toFixed(1)}, {detection.y1.toFixed(1)}) → (
                        {detection.x2.toFixed(1)}, {detection.y2.toFixed(1)})
                      </span>
                    </li>
                  ))}
                </ul>

                {visibleCount < filteredDetections.length && (
                  <button
                    type="button"
                    className="btn btn--ghost btn--small"
                    onClick={() =>
                      setVisibleCount((count) => count + DETAILS_PAGE_SIZE)
                    }
                  >
                    Show more ({filteredDetections.length - visibleCount} remaining)
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
