import { useId, useMemo, useState } from 'react'
import type { Detection, DetectionResponse } from '../types/detection'

interface DetectionViewerProps {
  imageUrl: string
  result: DetectionResponse
  filteredDetections: Detection[]
  confidenceThreshold: number
}

export function DetectionViewer({
  imageUrl,
  result,
  filteredDetections,
  confidenceThreshold,
}: DetectionViewerProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const labelId = useId()

  const dense = filteredDetections.length > 40

  const strokeWidth = useMemo(() => {
    const minDim = Math.min(result.image.width, result.image.height)
    return Math.max(1.5, minDim * 0.0025)
  }, [result.image.width, result.image.height])

  const fontSize = useMemo(() => {
    const minDim = Math.min(result.image.width, result.image.height)
    return Math.max(10, minDim * 0.018)
  }, [result.image.width, result.image.height])

  return (
    <div className="detection-viewer">
      <div
        className="detection-viewer__frame"
        role="img"
        aria-labelledby={labelId}
      >
        <span id={labelId} className="visually-hidden">
          Aerial image with {filteredDetections.length} person detection
          {filteredDetections.length === 1 ? '' : 's'} at confidence{' '}
          {Math.round(confidenceThreshold * 100)}% or higher
        </span>

        <img
          src={imageUrl}
          alt="Uploaded aerial imagery with person detections"
          className="detection-viewer__image"
          draggable={false}
        />

        <svg
          className="detection-viewer__overlay"
          viewBox={`0 0 ${result.image.width} ${result.image.height}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {filteredDetections.map((detection, index) => {
            const width = Math.max(detection.x2 - detection.x1, 1)
            const height = Math.max(detection.y2 - detection.y1, 1)
            const showLabel = !dense || hoveredIndex === index
            const confidenceLabel = `${(detection.confidence * 100).toFixed(1)}%`
            const labelY = Math.max(detection.y1 - 2, fontSize + 2)

            return (
              <g
                key={`${detection.x1}-${detection.y1}-${index}`}
                className="detection-box"
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
                onFocus={() => setHoveredIndex(index)}
                onBlur={() => setHoveredIndex(null)}
              >
                <rect
                  x={detection.x1}
                  y={detection.y1}
                  width={width}
                  height={height}
                  fill="rgba(56, 189, 248, 0.12)"
                  stroke="#38bdf8"
                  strokeWidth={strokeWidth}
                  vectorEffect="non-scaling-stroke"
                  tabIndex={0}
                >
                  <title>{confidenceLabel}</title>
                </rect>

                {showLabel && (
                  <g className="detection-box__label">
                    <rect
                      x={detection.x1}
                      y={labelY - fontSize}
                      width={fontSize * 3.4}
                      height={fontSize + 4}
                      fill="rgba(8, 15, 28, 0.82)"
                      rx={2}
                    />
                    <text
                      x={detection.x1 + 3}
                      y={labelY - 2}
                      fill="#e0f2fe"
                      fontSize={fontSize}
                      fontFamily="IBM Plex Sans, system-ui, sans-serif"
                    >
                      {confidenceLabel}
                    </text>
                  </g>
                )}
              </g>
            )
          })}
        </svg>
      </div>

      {dense && (
        <p className="detection-viewer__hint">
          Dense scene — hover or focus a box to reveal its confidence label.
        </p>
      )}
    </div>
  )
}
