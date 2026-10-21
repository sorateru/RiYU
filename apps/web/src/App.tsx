import { useState } from "react"

import { analyzeReplay } from "./api"
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
  async function handleAnalyze(documents: unknown[], observer: number) {
    setLoading(true)
    setError("")
    try {
      const analysis = await analyzeReplay(documents, observer)
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
