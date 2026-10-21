import { useState } from "react"

type HomePageProps = {
  onAnalyze: (replayUrl: string) => void
}

export default function HomePage({
  onAnalyze,
}: HomePageProps) {
  const [replayUrl, setReplayUrl] = useState("")

  function handleSubmit(
    event: React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault()

    const trimmedUrl = replayUrl.trim()

    if (!trimmedUrl) {
      return
    }

    onAnalyze(trimmedUrl)
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
          <div className="landing-input-wrapper">
            <input
              type="text"
              value={replayUrl}
              onChange={(event) => {
                setReplayUrl(
                  event.target.value,
                )
              }}
              placeholder="Paste Mahjong Soul replay URL"
              className="landing-input"
              autoComplete="off"
            />

            <button
              type="submit"
              className="landing-submit"
              disabled={!replayUrl.trim()}
            >
              ANALYZE
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}