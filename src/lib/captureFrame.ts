import type { FrameCaptureMetrics } from './liveTypes'

function targetSize(
  sourceWidth: number,
  sourceHeight: number,
  captureWidth: number,
): { width: number; height: number } {
  if (sourceWidth <= captureWidth) {
    return { width: sourceWidth, height: sourceHeight }
  }
  const scale = captureWidth / sourceWidth
  return {
    width: captureWidth,
    height: Math.max(1, Math.round(sourceHeight * scale)),
  }
}

/**
 * Draw the newest video frame into an offscreen canvas, optionally downscaled
 * to `captureWidth` while preserving aspect ratio (no crop / no stretch).
 * Encodes as JPEG Blob (never base64).
 */
export async function captureVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  options: {
    captureWidth: number
    jpegQuality: number
  },
): Promise<FrameCaptureMetrics> {
  const sourceWidth = video.videoWidth
  const sourceHeight = video.videoHeight

  if (!sourceWidth || !sourceHeight) {
    throw new Error('Camera frame is not ready yet.')
  }

  const { width: captureWidth, height: captureHeight } = targetSize(
    sourceWidth,
    sourceHeight,
    options.captureWidth,
  )

  if (canvas.width !== captureWidth) canvas.width = captureWidth
  if (canvas.height !== captureHeight) canvas.height = captureHeight

  const context = canvas.getContext('2d', { alpha: false })
  if (!context) {
    throw new Error('Unable to capture camera frame.')
  }

  const captureStarted = performance.now()
  context.drawImage(video, 0, 0, captureWidth, captureHeight)
  const captureMs = performance.now() - captureStarted

  const encodeStarted = performance.now()
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (result) => {
        if (!result) {
          reject(new Error('Failed to encode camera frame as JPEG.'))
          return
        }
        resolve(result)
      },
      'image/jpeg',
      options.jpegQuality,
    )
  })
  const encodeMs = performance.now() - encodeStarted

  return {
    blob,
    sourceWidth,
    sourceHeight,
    captureWidth,
    captureHeight,
    captureMs,
    encodeMs,
    payloadBytes: blob.size,
  }
}
