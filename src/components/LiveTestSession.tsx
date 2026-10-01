import {
  QUALITY_MARKERS,
  TEST_SCENARIOS,
  type QualityMarkerType,
  type TestScenario,
} from '../lib/liveConfig'
import type { QualityMarkerEvent, TestSessionExport } from '../lib/liveTypes'
import {
  exportTestSessionCsv,
  exportTestSessionJson,
} from '../lib/telemetryExport'

interface LiveTestSessionProps {
  scenario: TestScenario
  onScenarioChange: (scenario: TestScenario) => void
  recording: boolean
  onStart: () => void
  onStop: () => void
  recordCount: number
  markerCount: number
  markers: QualityMarkerEvent[]
  onMarker: (type: QualityMarkerType) => void
  onExport: () => TestSessionExport | null
  onClear: () => void
  cameraActive: boolean
}

const MARKER_LABELS: Record<QualityMarkerType, string> = {
  good: 'Good',
  missed_person: 'Missed Person',
  false_positive: 'False Positive',
  bad_box: 'Bad Box',
}

export function LiveTestSession({
  scenario,
  onScenarioChange,
  recording,
  onStart,
  onStop,
  recordCount,
  markerCount,
  markers,
  onMarker,
  onExport,
  onClear,
  cameraActive,
}: LiveTestSessionProps) {
  const handleExportJson = () => {
    const session = onExport()
    if (!session) return
    exportTestSessionJson(session)
  }

  const handleExportCsv = () => {
    const session = onExport()
    if (!session) return
    exportTestSessionCsv(session)
  }

  return (
    <div className="live-test-session">
      <h3 className="live-debug__heading">Test Session</h3>
      <p className="live-debug__hint">
        Records per-inference telemetry locally. Does not upload analytics or
        record video. Manual markers are qualitative only.
      </p>

      <label className="live-field">
        <span>Scenario</span>
        <select
          value={scenario}
          disabled={recording}
          onChange={(event) =>
            onScenarioChange(event.target.value as TestScenario)
          }
        >
          {TEST_SCENARIOS.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>

      <div className="action-bar">
        {!recording ? (
          <button
            type="button"
            className="btn btn--primary"
            disabled={!cameraActive}
            onClick={onStart}
          >
            Start Test Session
          </button>
        ) : (
          <button type="button" className="btn btn--primary" onClick={onStop}>
            Stop Test Session
          </button>
        )}
        <button
          type="button"
          className="btn btn--ghost"
          disabled={recordCount === 0 && markerCount === 0}
          onClick={handleExportJson}
        >
          Export JSON
        </button>
        <button
          type="button"
          className="btn btn--ghost"
          disabled={recordCount === 0 && markerCount === 0}
          onClick={handleExportCsv}
        >
          Export CSV
        </button>
        <button type="button" className="btn btn--ghost" onClick={onClear}>
          Clear
        </button>
      </div>

      <p className="live-debug__progress">
        {recording ? 'Recording…' : 'Idle'} · {recordCount} inferences ·{' '}
        {markerCount} markers
      </p>

      <div className="marker-row" role="group" aria-label="Quality markers">
        {QUALITY_MARKERS.map((type) => (
          <button
            key={type}
            type="button"
            className="btn btn--ghost btn--small"
            disabled={!recording}
            onClick={() => onMarker(type)}
          >
            {MARKER_LABELS[type]}
          </button>
        ))}
      </div>

      {markers.length > 0 && (
        <ul className="marker-list">
          {markers.slice(-8).map((marker, index) => (
            <li key={`${marker.timestamp}-${index}`}>
              {marker.type} · req {marker.requestId ?? '—'} · people{' '}
              {marker.peopleCount}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
