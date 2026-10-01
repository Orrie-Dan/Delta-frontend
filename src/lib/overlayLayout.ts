/**
 * Compute the content rectangle for object-fit: contain.
 * Coordinates are relative to the element's client box.
 */
export function getObjectFitContainRect(
  sourceWidth: number,
  sourceHeight: number,
  clientWidth: number,
  clientHeight: number,
): { x: number; y: number; width: number; height: number } {
  if (!sourceWidth || !sourceHeight || !clientWidth || !clientHeight) {
    return { x: 0, y: 0, width: clientWidth, height: clientHeight }
  }

  const sourceAspect = sourceWidth / sourceHeight
  const clientAspect = clientWidth / clientHeight

  if (clientAspect > sourceAspect) {
    const width = clientHeight * sourceAspect
    const height = clientHeight
    return {
      x: (clientWidth - width) / 2,
      y: 0,
      width,
      height,
    }
  }

  const width = clientWidth
  const height = clientWidth / sourceAspect
  return {
    x: 0,
    y: (clientHeight - height) / 2,
    width,
    height,
  }
}
