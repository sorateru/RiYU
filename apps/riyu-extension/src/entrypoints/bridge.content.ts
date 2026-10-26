/*
 * Modified for RiYU from MahjongSoul-review-supporter (Apache-2.0).
 * The original decoded to Tenhou/6; this version emits an anonymized local
 * interchange document to the extension popup only after a user click.
 */
import { decodeGameRecord } from "@/lib/record-decode"
import { buildDecodedReplay } from "@/lib/record-export"
import { RSR } from "@/lib/messages"

type ExtractionResult =
  | { ok: true; decoded: Record<string, unknown>; shareUrl: string }
  | { ok: false; error: string }

export default defineContentScript({
  matches: [
    "https://game.mahjongsoul.com/*",
    "https://mahjongsoul.game.yo-star.com/*",
    "https://game.maj-soul.net/*",
    "https://game.maj-soul.com/*",
  ],
  main() {
    let pending: ((result: ExtractionResult) => void) | null = null

    chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
      if (request?.type !== "riyu:extract-replay") return false
      if (pending !== null) {
        sendResponse({ ok: false, error: "牌譜の取得処理は既に実行中です。" })
        return false
      }
      pending = sendResponse
      window.postMessage({ direction: RSR.GET_RECORD }, window.location.origin)
      window.setTimeout(() => {
        if (pending !== sendResponse) return
        pending = null
        sendResponse({
          ok: false,
          error: "牌譜データを取得できません。牌譜を開いてからページを再読み込みしてください。",
        })
      }, 5000)
      return true
    })

    window.addEventListener("message", (event) => {
      if (event.source !== window || event.origin !== location.origin) return
      if (!event.data || event.data.direction !== RSR.RECORD || pending === null) return
      const respond = pending
      pending = null
      try {
        const record = decodeGameRecord(new Uint8Array(event.data.bytes as ArrayBuffer))
        respond({
          ok: true,
          decoded: buildDecodedReplay(record),
          shareUrl: window.location.href,
        })
      } catch (error) {
        respond({
          ok: false,
          error: error instanceof Error ? error.message : "牌譜のdecodeに失敗しました。",
        })
      }
    })
  },
})
