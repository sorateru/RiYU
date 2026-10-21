export type CandidateAnalysis = {
  tile: string
  score: number
  shanten: number
  ukeire: number
  dealInRisk: number | null
  expectedValue: number | null
}

export type HandTileAnalysis = {
  tile: string
  recommendation: number
  drawn?: boolean
}

export type TurnAnalysis = {
  turn: number

  handTiles: HandTileAnalysis[]

  candidates: CandidateAnalysis[]

  scores: {
    self: number
    right: number
    opposite: number
    left: number
  }

  discards: {
    self: string[]
    right: string[]
    opposite: string[]
    left: string[]
  }

  remainingTiles: number

  dora: string

  eventSequence?: number
}

export type RoundAnalysis = {
  id: string

  label: string

  turns: TurnAnalysis[]
}

export type ReplayAnalysis = {
  format?: "riyu-replay-analysis/v1"

  sourceLogId?: string

  sourceLogIds?: string[]

  observer?: number

  analysisMode?:
    | "exact_tile_efficiency"
    | "model_estimate"

  rounds: RoundAnalysis[]
}
