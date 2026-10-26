import "./style.css"

type ExtractionResult =
  | { ok: true; decoded: Record<string, unknown>; shareUrl: string }
  | { ok: false; error: string }

const serverInput = document.querySelector<HTMLInputElement>("#server")!
const observerInput = document.querySelector<HTMLSelectElement>("#observer")!
const sendButton = document.querySelector<HTMLButtonElement>("#send")!
const status = document.querySelector<HTMLParagraphElement>("#status")!

chrome.storage.local.get(["riyuServer", "riyuObserver"], (saved) => {
  if (typeof saved.riyuServer === "string") serverInput.value = saved.riyuServer
  if (typeof saved.riyuObserver === "number") {
    observerInput.value = String(saved.riyuObserver)
  }
})

sendButton.addEventListener("click", async () => {
  sendButton.disabled = true
  status.textContent = "牌譜を取得中…"
  try {
    const server = validatedLoopbackURL(serverInput.value)
    const observer = Number(observerInput.value)
    await chrome.storage.local.set({ riyuServer: server, riyuObserver: observer })
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
    if (!tab.id) throw new Error("アクティブな牌譜タブがありません。")
    const extracted = await chrome.tabs.sendMessage<unknown, ExtractionResult>(tab.id, {
      type: "riyu:extract-replay",
    })
    if (!extracted?.ok) {
      throw new Error(extracted?.error ?? "牌譜を取得できませんでした。")
    }
    status.textContent = "RiYUへ送信中…"
    const response = await fetch(`${server}/api/extension-import`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        decoded_mahjong_soul: extracted.decoded,
        observer,
        share_url: extracted.shareUrl,
      }),
    })
    const body = await response.json().catch(() => null)
    if (!response.ok) throw new Error(body?.detail ?? `RiYU API error (${response.status})`)
    if (typeof body?.review_url !== "string") throw new Error("RiYU APIの応答が不正です。")
    await chrome.tabs.create({ url: body.review_url })
    window.close()
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "送信に失敗しました。"
  } finally {
    sendButton.disabled = false
  }
})

function validatedLoopbackURL(value: string): string {
  const url = new URL(value.trim())
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname)) {
    throw new Error("RiYU URLはhttp://127.0.0.1 または http://localhostを指定してください。")
  }
  if (url.username || url.password || (url.pathname !== "/" && url.pathname !== "")) {
    throw new Error("RiYU URLに認証情報やパスを含めないでください。")
  }
  return url.origin
}
