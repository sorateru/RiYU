import { useEffect, useState } from "react"

import { analyzeReplay, loadExtensionImport } from "./api"
import AnalysisPage from "./pages/AnalysisPage"
import HomePage from "./pages/HomePage"
import type { ReplayAnalysis } from "./types/analysis"

type Page =
  | {
      name: "home"
    }
  | {
      name: "analysis"
      analysis: ReplayAnalysis
    }

export default function App() {
  const [page, setPage] = useState<Page>({
    name: "home",
  })
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
        onBack={() => {
          setPage({
            name: "home",
          })
        }}
      />
    )
  }
  return (
    <HomePage
      onAnalyze={handleAnalyze}
      loading={loading}
      error={error}
    />
  )
}
