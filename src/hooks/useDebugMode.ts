import { useMemo } from 'react'

/** Developer diagnostics mode via ?debug=true */
export function useDebugMode(): boolean {
  return useMemo(() => {
    if (typeof window === 'undefined') return false
    return new URLSearchParams(window.location.search).get('debug') === 'true'
  }, [])
}
