import type { ReplayAnalysis } from "./types/analysis"

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

export async function analyzeReplay(
  logs: unknown[],
  observer: number,
): Promise<ReplayAnalysis> {
  const response = await fetch(`${API_BASE}/api/replay-analysis`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ logs, observer }),
  })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(body?.detail ?? `Analysis failed (${response.status})`)
  }
  return body as ReplayAnalysis
}
