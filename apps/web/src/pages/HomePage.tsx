import { useRef, useState } from "react"

const MAX_UPLOAD_BYTES = 16 * 1024 * 1024
const MAX_UPLOAD_FILES = 64

type HomePageProps = {
  onAnalyze: (input: unknown, observer: number) => Promise<void>
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

  function decodeInput(): unknown {
    if (/^https:\/\//i.test(source.trim())) {
      return source.trim()
    }
    if (/^http:\/\//i.test(source.trim())) {
      throw new Error("共有リンクはHTTPS形式で入力してください。")
    }
    const parsed = JSON.parse(source) as unknown
    if (Array.isArray(parsed) && parsed.length === 0) {
      throw new Error("解析する局がありません。")
    }
    return parsed
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
      const input = decodeInput()
      setInputError("")
      void onAnalyze(input, observer)
    } catch (caught) {
      setInputError(caught instanceof Error ? caught.message : "JSONを解析できません。")
    }
  }

  async function loadFiles(files: File[]) {
    if (!files.length) return
    if (files.length > MAX_UPLOAD_FILES) {
      setInputError(`一度に選択できるファイルは${MAX_UPLOAD_FILES}件までです。`)
      return
    }
    const totalBytes = files.reduce((total, file) => total + file.size, 0)
    if (totalBytes > MAX_UPLOAD_BYTES) {
      setInputError("合計ファイルサイズは16 MiB以下にしてください。")
      return
    }
    try {
      const documents = await Promise.all(files.map(async (file) => JSON.parse(await file.text()) as unknown))
      if (
        documents.length > 1 &&
        documents.some((document) =>
          document != null &&
          typeof document === "object" &&
          [
            "mahjong-ai-replay-bundle/v1",
            "mahjong-soul-decoded-replay/v1",
          ].includes(String((document as { format?: unknown }).format))
        )
      ) {
        throw new Error("Replay Bundleとdecode済み雀魂JSONは単独で選択してください。")
      }
      setSource(JSON.stringify(documents.length === 1 ? documents[0] : documents, null, 2))
      setSelectedFiles(files.map((file) => file.name))
      setInputError("")
    } catch (caught) {
      setInputError(
        caught instanceof Error
          ? caught.message
          : "選択したファイルに正しいJSONではないものがあります。",
      )
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
            雀魂共有URL、EventLog、Replay Bundle、またはローカルdecode済みJSONを読み込みます。
            複数の局JSONもまとめて選択できます。
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
              placeholder="牌譜JSON / Replay Bundle / decode済み雀魂JSONを貼り付けるか、ファイルをドロップ"
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
