export type RealtimeSession = {
  session_id: string
  observer: number
  source_kind: string
  source_service: string | null
  source_profile: string | null
  source_acquisition: string | null
  authorization_recorded: boolean
  connected_at_unix: number
  last_sequence: number | null
  viewer_count: number
  has_analysis: boolean
  latest_type: string | null
}

export type RealtimeStreamState = {
  round_id: string | null
  round: {
    wind?: string
    hand_number?: number
    honba?: number
    riichi_sticks?: number
  } | null
  turn: number | null
  hand: Array<{ kind?: string; red?: boolean } | string>
  scores: number[] | null
  remaining_tiles: number | null
  current_actor: number | null
  decision_required: boolean
  decision_kind: string | null
  legal_action_count: number
  next_source_sequence: number
}

export type RealtimeMessage = {
  type: string
  session_id?: string
  sequence?: number
  source_sequence?: number | null
  latest?: RealtimeMessage | null
  state?: RealtimeStreamState
  result?: Record<string, any>
  message?: string
  status?: PerceptionStatus
}

export type PerceptionStatus = {
  reason: "waiting_for_frame" | "unknown_slots" | "low_confidence" | "stabilizing" | "stable" | "stable_unchanged" | "midround_waiting_for_self_turn" | "midround_unsupported_melds" | "desynchronized_waiting_for_round"
  candidate_frames: number
  required_frames: number
  minimum_confidence?: number
  unknown_slots: string[]
  recognized_counts: Record<string, number>
  desynchronized: boolean
  round_initialized: boolean
}
