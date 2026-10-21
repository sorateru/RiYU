import { useState } from "react"

import AnalysisPage from "./pages/AnalysisPage"
import HomePage from "./pages/HomePage"

type Page =
  | {
      name: "home"
    }
  | {
      name: "analysis"
      replayUrl: string
    }

export default function App() {
  const [page, setPage] = useState<Page>({
    name: "home",
  })

  if (page.name === "analysis") {
    return (
      <AnalysisPage
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
      onAnalyze={(replayUrl) => {
        setPage({
          name: "analysis",
          replayUrl,
        })
      }}
    />
  )
}