import type { TestSessionExport } from './liveTypes'

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function sessionStamp(iso: string): string {
  return iso.replace(/[:.]/g, '-').replace(/Z$/, '')
}

export function exportTestSessionJson(session: TestSessionExport) {
  const filename = `live-detection-test-${sessionStamp(session.stoppedAt)}.json`
  const blob = new Blob([JSON.stringify(session, null, 2)], {
    type: 'application/json',
  })
  downloadBlob(blob, filename)
}

function csvEscape(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return ''
  const text = String(value)
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

export function exportTestSessionCsv(session: TestSessionExport) {
  const filename = `live-detection-test-${sessionStamp(session.stoppedAt)}.csv`
  const header = [
    'kind',
    'timestamp',
    'requestId',
    'scenario',
    'captureWidth',
    'captureHeight',
    'jpegQuality',
    'payloadKb',
    'captureMs',
    'encodeMs',
    'requestTotalMs',
    'serverInferenceMs',
    'nonInferenceOverheadMs',
    'peopleReturned',
    'visibleDetections',
    'averageConfidence',
    'modelImgsz',
    'markerType',
  ]

  const rows: string[] = [header.join(',')]

  for (const record of session.records) {
    rows.push(
      [
        'inference',
        record.timestamp,
        record.requestId,
        session.scenario,
        record.captureWidth,
        record.captureHeight,
        record.jpegQuality,
        record.payloadKb.toFixed(2),
        record.captureMs.toFixed(2),
        record.encodeMs.toFixed(2),
        record.requestTotalMs.toFixed(2),
        record.serverInferenceMs.toFixed(2),
        record.nonInferenceOverheadMs.toFixed(2),
        record.peopleReturned,
        record.visibleDetections,
        record.averageConfidence?.toFixed(4) ?? '',
        record.modelImgsz ?? '',
        '',
      ]
        .map(csvEscape)
        .join(','),
    )
  }

  for (const marker of session.markers) {
    rows.push(
      [
        'marker',
        marker.timestamp,
        marker.requestId ?? '',
        marker.scenario,
        marker.captureWidth,
        '',
        marker.jpegQuality,
        '',
        '',
        '',
        '',
        '',
        '',
        marker.peopleCount,
        '',
        '',
        marker.modelImgsz ?? '',
        marker.type,
      ]
        .map(csvEscape)
        .join(','),
    )
  }

  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' })
  downloadBlob(blob, filename)
}
