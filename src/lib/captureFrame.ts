const JPEG_QUALITY = 0.8

/**
 * Capture the newest available video frame as a JPEG Blob.
 * Reuses a single canvas to avoid allocating per frame.
 */
export function captureVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  quality = JPEG_QUALITY,
): Promise<Blob> {
  const width = video.videoWidth
  const height = video.videoHeight

  if (!width || !height) {
    return Promise.reject(new Error('Camera frame is not ready yet.'))
  }

  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height

  const context = canvas.getContext('2d', { alpha: false })
  if (!context) {
    return Promise.reject(new Error('Unable to capture camera frame.'))
  }

  context.drawImage(video, 0, 0, width, height)

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Failed to encode camera frame as JPEG.'))
          return
        }
        resolve(blob)
      },
      'image/jpeg',
      quality,
    )
  })
}
