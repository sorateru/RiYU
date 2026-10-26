import { useEffect, useState } from "react"

import { analyzeReplay, loadExtensionImport } from "./api"
import AnalysisPage from "./pages/AnalysisPage"
import HomePage from "./pages/HomePage"
import RealtimePage from "./pages/RealtimePage"
import type { ReplayAnalysis } from "./types/analysis"

type Page =
  | {
      name: "home"
    }
  | {
      name: "analysis"
      analysis: ReplayAnalysis
    }
  | {
      name: "realtime"
    }

function initialPage(): Page {
  return window.location.hash === "#realtime" ? { name: "realtime" } : { name: "home" }
}

export default function App() {
  const [page, setPage] = useState<Page>(initialPage)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    const token = new URLSearchParams(window.location.search).get("import")
    if (!token) return
    setLoading(true)
    loadExtensionImport(token)
      .then((analysis) => {
        setPage({ name: "analysis", analysis })
        window.history.replaceState({}, "", window.location.pathname)
      })
      .catch((caught: unknown) => {
        setError(caught instanceof Error ? caught.message : "Extension import failed")
      })
      .finally(() => setLoading(false))
  }, [])

  function showHome() {
    window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}`)
    setPage({ name: "home" })
  }

  function showRealtime() {
    window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}#realtime`)
    setPage({ name: "realtime" })
  }

  async function handleAnalyze(input: unknown, observer: number) {
    setLoading(true)
    setError("")
    try {
      const analysis = await analyzeReplay(input, observer)
      setPage({ name: "analysis", analysis })
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis failed")
    } finally {
      setLoading(false)
    }
  }

  if (page.name === "analysis") {
    return (
      <AnalysisPage
        analysis={page.analysis}
        onBack={showHome}
        onOpenRealtime={showRealtime}
      />
    )
  }
  if (page.name === "realtime") {
    return <RealtimePage onBack={showHome} />
  }
  return (
    <HomePage
      onAnalyze={handleAnalyze}
      onOpenRealtime={showRealtime}
      loading={loading}
      error={error}
    />
  )
}
