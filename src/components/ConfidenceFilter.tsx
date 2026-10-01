import { useId } from 'react'

interface ConfidenceFilterProps {
  value: number
  onChange: (value: number) => void
  disabled?: boolean
}

export function ConfidenceFilter({
  value,
  onChange,
  disabled = false,
}: ConfidenceFilterProps) {
  const id = useId()
  const percent = Math.round(value * 100)

  return (
    <div className="confidence-filter">
      <div className="confidence-filter__header">
        <label htmlFor={id} className="confidence-filter__label">
          Confidence
        </label>
        <span className="confidence-filter__value" aria-live="polite">
          {percent}%
        </span>
      </div>

      <input
        id={id}
        type="range"
        min={25}
        max={95}
        step={1}
        value={percent}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value) / 100)}
        className="confidence-filter__slider"
        aria-valuemin={25}
        aria-valuemax={95}
        aria-valuenow={percent}
        aria-valuetext={`${percent} percent`}
      />

      <p className="confidence-filter__hint">
        Filters detections already returned by the server. Backend minimum is
        25%.
      </p>
    </div>
  )
}
