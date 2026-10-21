import { useEffect, useRef, useState } from "react"

type HomePageProps = {
  onAnalyze: (documents: unknown[], observer: number) => Promise<void>
  loading: boolean
  error: string
}

export default function HomePage({
  onAnalyze,
  loading,
  error,
}: HomePageProps) {
  const [source, setSource] = useState("")
  const [observer, setObserver] = useState(0)
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [inputError, setInputError] = useState("")
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const draft = localStorage.getItem("riyu.replayDraft")
    if (draft) setSource(draft)
  }, [])

  function decodeDocuments(): unknown[] {
    const parsed = JSON.parse(source) as unknown
    if (Array.isArray(parsed)) return parsed
    if (parsed && typeof parsed === "object" && "logs" in parsed) {
      const logs = (parsed as { logs?: unknown }).logs
      if (!Array.isArray(logs)) throw new Error("logs は配列で指定してください。")
      return logs
    }
    return [parsed]
  }

  function handleSubmit(
    event: React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault()

    const normalized = source.trim()

    if (!normalized || loading) {
      return
    }

    try {
      const documents = decodeDocuments()
      if (documents.length === 0) throw new Error("解析する局がありません。")
      setInputError("")
      localStorage.setItem("riyu.replayDraft", normalized)
      void onAnalyze(documents, observer)
    } catch (caught) {
      setInputError(caught instanceof Error ? caught.message : "JSONを解析できません。")
    }
  }

  async function loadFiles(files: File[]) {
    if (!files.length) return
    try {
      const documents = await Promise.all(files.map(async (file) => JSON.parse(await file.text()) as unknown))
      setSource(JSON.stringify(documents.length === 1 ? documents[0] : documents, null, 2))
      setSelectedFiles(files.map((file) => file.name))
      setInputError("")
    } catch {
      setInputError("選択したファイルに正しいJSONではないものがあります。")
    }
  }

  return (
    <div className="landing-page">
      <main className="landing-center">
        <div className="landing-brand">
          <h1>RiYU</h1>

          <p>
            Riichi Yield Utility
          </p>
        </div>

        <form
          className="landing-form"
          onSubmit={handleSubmit}
        >
          <p className="landing-description">
            正規化済み <code>mahjong-ai-event-log/v1</code> を読み込みます。
            複数の局JSONをまとめて選択できます。
          </p>

          <div
            className={`landing-input-wrapper ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              void loadFiles(Array.from(event.dataTransfer.files))
            }}
          >
            <textarea
              value={source}
              onChange={(event) => {
                setSource(
                  event.target.value,
                )
              }}
              placeholder="正規化済み牌譜JSONを貼り付けるか、ファイルをドロップ"
              className="landing-input"
              autoComplete="off"
            />
          </div>
          <div className="landing-actions">
            <button type="button" className="landing-file" onClick={() => fileInput.current?.click()}>
              JSONを選択
            </button>
            <input ref={fileInput} className="visually-hidden" type="file" multiple accept="application/json,.json" onChange={(event) => void loadFiles(Array.from(event.target.files ?? []))} />
            <label className="landing-observer">
              SEAT
              <select value={observer} onChange={(event) => setObserver(Number(event.target.value))}>
                <option value={0}>0 / East</option><option value={1}>1 / South</option>
                <option value={2}>2 / West</option><option value={3}>3 / North</option>
              </select>
            </label>
            <button type="submit" className="landing-submit" disabled={!source.trim() || loading}>
              {loading ? "解析中…" : "解析する"}
            </button>
          </div>
          {selectedFiles.length > 0 && <p className="selected-files">{selectedFiles.length}ファイル: {selectedFiles.join(", ")}</p>}
          {(inputError || error) && <p className="landing-error" role="alert">{inputError || error}</p>}
        </form>
      </main>
    </div>
  )
}
