import type { ReplayAnalysis } from "./types/analysis"

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

export async function analyzeReplay(
  input: unknown,
  observer: number,
): Promise<ReplayAnalysis> {
  const payload = replayPayload(input, observer)
  const response = await fetch(`${API_BASE}/api/replay-analysis`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(body?.detail ?? `Analysis failed (${response.status})`)
  }
  return body as ReplayAnalysis
}

export async function loadExtensionImport(token: string): Promise<ReplayAnalysis> {
  if (!/^[A-Za-z0-9_-]{16,64}$/.test(token)) {
    throw new Error("拡張機能のimport tokenが不正です。")
  }
  const response = await fetch(`${API_BASE}/api/extension-import/${token}`)
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(body?.detail ?? `Import failed (${response.status})`)
  }
  return body as ReplayAnalysis
}

function replayPayload(input: unknown, observer: number): Record<string, unknown> {
  if (typeof input === "string" && /^https:\/\//i.test(input.trim())) {
    return { url: input.trim(), observer }
  }
  if (Array.isArray(input)) return { logs: input, observer }
  if (!input || typeof input !== "object") {
    throw new Error("牌譜入力はJSONオブジェクトまたは配列である必要があります。")
  }
  const value = input as Record<string, unknown>
  if (value.format === "mahjong-ai-replay-bundle/v1") {
    return { bundle: value, observer }
  }
  if (value.format === "mahjong-soul-decoded-replay/v1") {
    return { decoded_mahjong_soul: value, observer }
  }
  if (value.format === "mahjong-ai-event-log/v1") {
    return { log: value, observer }
  }
  if (Array.isArray(value.logs)) {
    return { logs: value.logs, observer }
  }
  throw new Error("対応していない牌譜形式です。")
}
