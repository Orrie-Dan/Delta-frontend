import type { WorkspaceMode } from '../types/detection'

interface ModeSwitcherProps {
  mode: WorkspaceMode
  onChange: (mode: WorkspaceMode) => void
  disabled?: boolean
}

export function ModeSwitcher({ mode, onChange, disabled = false }: ModeSwitcherProps) {
  return (
    <div className="mode-switcher" role="tablist" aria-label="Detection mode">
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'upload'}
        className={`mode-switcher__btn${mode === 'upload' ? ' mode-switcher__btn--active' : ''}`}
        onClick={() => onChange('upload')}
        disabled={disabled}
      >
        Upload Image
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'live'}
        className={`mode-switcher__btn${mode === 'live' ? ' mode-switcher__btn--active' : ''}`}
        onClick={() => onChange('live')}
        disabled={disabled}
      >
        Live Camera
      </button>
    </div>
  )
}
