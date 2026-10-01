export function average(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

export function median(values: number[]): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1]! + sorted[mid]!) / 2
  }
  return sorted[mid]!
}

/** Requires reasonably large sample size; returns null when n < 5. */
export function percentile95(values: number[]): number | null {
  if (values.length < 5) return null
  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(0.95 * sorted.length) - 1),
  )
  return sorted[index]!
}

export function averageNullable(values: Array<number | null>): number | null {
  const nums = values.filter((value): value is number => value !== null)
  if (nums.length === 0) return null
  return average(nums)
}
