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

  honba?: number

  riichiSticks?: number

  decisionCount?: number

  turns: TurnAnalysis[]
}

export type ReplayAnalysis = {
  format?: "riyu-replay-analysis/v1"

  sourceLogId?: string

  sourceLogIds?: string[]

  source?: {
    platform: "mahjong_soul" | "tenhou" | "local" | "other"
    acquisition: string
    authorization: string
    converterName: string
    converterVersion: string
    reference: string | null
  }

  observer?: number

  analysisMode?:
    | "exact_tile_efficiency"
    | "model_estimate"

  decisionCount?: number

  rounds: RoundAnalysis[]
}
