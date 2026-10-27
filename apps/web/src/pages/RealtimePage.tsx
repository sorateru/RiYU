import { useEffect, useMemo, useRef, useState } from "react"

import { loadScreenCaptureSessions, realtimeViewerURL } from "../api"
import type {
  RealtimeMessage,
  PerceptionStatus,
  RealtimeSession,
  RealtimeStreamState,
} from "../types/realtime"

type ConnectionStatus = "waiting" | "connecting" | "connected" | "offline"

type RealtimePageProps = {
  onBack: () => void
}

const WIND_LABELS: Record<string, string> = {
  east: "東",
  south: "南",
  west: "西",
  north: "北",
}

function tileLabel(tile: RealtimeStreamState["hand"][number] | undefined): string {
  if (!tile) return "?"
  if (typeof tile === "string") return tile
  const kind = tile.kind ?? "?"
  const honors: Record<string, string> = {
    east: "東", south: "南", west: "西", north: "北",
    white: "白", green: "發", red: "中",
  }
  if (honors[kind]) return honors[kind]
  if (/^[mps][1-9]$/.test(kind)) {
    const suffix = kind[0] === "m" ? "萬" : kind[0] === "p" ? "筒" : "索"
    return `${tile.red ? "赤" : ""}${kind[1]}${suffix}`
  }
  return kind
}

function actionLabel(action: Record<string, any> | undefined): string {
  if (!action) return "候補手"
  const labels: Record<string, string> = {
    discard: "打牌", riichi: "リーチ", chi: "チー", pon: "ポン",
    kan: "カン", ron: "ロン", tsumo: "ツモ", pass: "パス",
  }
  const tile = action.discard ?? action.tile ?? action.called_tile
  return `${labels[action.type] ?? action.type ?? "候補手"}${tile ? ` ${tileLabel(tile)}` : ""}`
}

function formatConnectedAt(value: number): string {
  return new Intl.DateTimeFormat("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value * 1_000))
}

export default function RealtimePage({ onBack }: RealtimePageProps) {
  const [sessions, setSessions] = useState<RealtimeSession[]>([])
  const [status, setStatus] = useState<ConnectionStatus>("waiting")
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [message, setMessage] = useState<RealtimeMessage | null>(null)
  const [streamState, setStreamState] = useState<RealtimeStreamState | null>(null)
  const [perceptionStatus, setPerceptionStatus] = useState<PerceptionStatus | null>(null)
  const [error, setError] = useState("")
  const socketRef = useRef<WebSocket | null>(null)
  const activeSessionRef = useRef<string | null>(null)

  function acceptMessage(payload: RealtimeMessage) {
    if (payload.type === "analysis") {
      setMessage(payload)
      if (payload.state) setStreamState(payload.state)
    } else if (payload.type === "state_update" && payload.state) {
      setStreamState(payload.state)
    } else if (payload.type === "perception_status" && payload.status) {
      setPerceptionStatus(payload.status)
    }
  }

  function disconnect(nextStatus: ConnectionStatus = "waiting") {
    const socket = socketRef.current
    socketRef.current = null
    socket?.close()
    activeSessionRef.current = null
    setActiveSession(null)
    setStatus(nextStatus)
  }

  function connect(session: RealtimeSession) {
    if (activeSessionRef.current === session.session_id && socketRef.current) return
    disconnect("connecting")
    setError("")
    setMessage(null)
    setStreamState(null)
    setPerceptionStatus(null)
    setActiveSession(session.session_id)
    activeSessionRef.current = session.session_id
    const socket = new WebSocket(realtimeViewerURL(session.session_id))
    socketRef.current = socket
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as RealtimeMessage
      if (payload.type === "subscribed") {
        setStatus("connected")
        if (payload.latest) acceptMessage(payload.latest)
      } else if (payload.type === "analysis" || payload.type === "state_update" || payload.type === "perception_status") {
        setStatus("connected")
        acceptMessage(payload)
      } else if (payload.type === "source_offline") {
        setStatus("offline")
      } else if (payload.type === "error") {
        setError(payload.message ?? "画面キャプチャ解析へ接続できませんでした。")
      }
    }
    socket.onerror = () => setError("画面キャプチャ解析のWebSocketへ接続できませんでした。")
    socket.onclose = () => {
      if (socketRef.current === socket) {
        socketRef.current = null
        activeSessionRef.current = null
        setStatus((current) => current === "offline" ? current : "waiting")
      }
    }
  }

  async function refreshSessions(autoConnect = false) {
    try {
      const available = await loadScreenCaptureSessions()
      setSessions(available)
      setError("")
      if (autoConnect && available.length > 0 && socketRef.current == null) {
        connect(available[0])
      }
      if (
        activeSessionRef.current &&
        !available.some((session) => session.session_id === activeSessionRef.current)
      ) {
        disconnect("offline")
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "画面キャプチャ入力を確認できませんでした。")
    }
  }

  useEffect(() => {
    void refreshSessions(true)
    const timer = window.setInterval(() => void refreshSessions(true), 2_000)
    return () => {
      window.clearInterval(timer)
      socketRef.current?.close()
    }
  }, [])

  const analysis = message?.result
  const realtime = analysis?.realtime as Record<string, any> | undefined
  const candidates = useMemo(() => {
    const policy = analysis?.model_policy?.value
    if (Array.isArray(policy) && policy.length > 0) return policy.slice(0, 6)
    const efficiency = analysis?.discard_efficiency?.value
    return Array.isArray(efficiency) ? efficiency.slice(0, 6) : []
  }, [analysis])
  const currentSession = sessions.find((session) => session.session_id === activeSession)
  const round = streamState?.round
  const roundLabel = round
    ? `${WIND_LABELS[round.wind ?? ""] ?? round.wind ?? ""}${round.hand_number ?? "?"}局`
    : "対局待機中"
  const statusLabel = {
    waiting: "キャプチャ待機",
    connecting: "接続中",
    connected: "画面解析中",
    offline: "キャプチャ終了",
  }[status]
  const perceptionLabel = !perceptionStatus ? "認識結果待ち"
    : perceptionStatus.desynchronized ? "局面同期を再待機中"
    : ({
        waiting_for_frame: "最初の画面を待機中",
        unknown_slots: "未認識の領域があります",
        low_confidence: "認識信頼度が不足しています",
        stabilizing: `画面を確認中 ${perceptionStatus.candidate_frames}/${perceptionStatus.required_frames}`,
        stable: "局面を認識しました",
        stable_unchanged: "同じ局面を追跡中",
        midround_waiting_for_self_turn: "途中局：自分のツモ番を待っています",
        midround_unsupported_melds: "途中局：副露のない次局を待っています",
        desynchronized_waiting_for_round: "同期ずれ：次局開始を待っています",
      } as const)[perceptionStatus.reason]

  return (
    <div className="realtime-page">
      <header className="realtime-header">
        <button className="brand-button" onClick={onBack} aria-label="牌譜解析トップへ戻る">
          <strong>RiYU</strong>
          <span>Riichi Yield Utility</span>
        </button>
        <nav aria-label="解析モード">
          <button onClick={onBack}>牌譜解析</button>
          <button className="active" aria-current="page">リアルタイム</button>
        </nav>
      </header>

      <main className="realtime-shell">
        <section className="realtime-hero">
          <div>
            <p className="realtime-eyebrow">MAHJONG SOUL · SCREEN CAPTURE</p>
            <h1>{status === "connected" ? "Live analysis" : "Waiting for capture"}</h1>
            <p>Safari上の雀魂画面を読み取り専用で取得し、安定化した局面だけをリアルタイム解析します。</p>
          </div>
          <span className={`realtime-status status-${status}`}><i />{statusLabel}</span>
        </section>

        <section className="capture-pipeline" aria-label="画面キャプチャ解析フロー">
          <div><b>01</b><span>Safari</span><small>許可済み対局画面</small></div>
          <i>→</i>
          <div><b>02</b><span>Vision Adapter</span><small>画面認識・連続frame確認</small></div>
          <i>→</i>
          <div><b>03</b><span>RiYU</span><small>合法性検証・AI解析</small></div>
        </section>

        <div className="realtime-grid">
          <aside className="capture-sidebar">
            <section className="capture-card">
              <div className="capture-card-heading">
                <div><span>CAPTURE SOURCE</span><h2>画面キャプチャ</h2></div>
                <button onClick={() => void refreshSessions(true)}>更新</button>
              </div>
              {sessions.length === 0 ? (
                <div className="capture-empty">
                  <span className="capture-ring" />
                  <strong>Vision Adapterを待っています</strong>
                  <p>Terminalから画面キャプチャを起動すると、自動的に接続します。</p>
                  <code>tools/run_majsoul_webapp.py local-data/majsoul-vision.json</code>
                </div>
              ) : (
                <div className="capture-session-list">
                  {sessions.map((session) => (
                    <button
                      className={session.session_id === activeSession ? "active" : ""}
                      key={session.session_id}
                      onClick={() => connect(session)}
                    >
                      <span><i />{session.session_id}</span>
                      <small>画面収録 · {formatConnectedAt(session.connected_at_unix)}</small>
                      <small>observer {session.observer} · seq {session.last_sequence ?? "—"}</small>
                    </button>
                  ))}
                </div>
              )}
              {error && <p className="capture-error" role="alert">{error}</p>}
              {currentSession && (
                <div className="capture-diagnostic" aria-live="polite">
                  <strong>{perceptionLabel}</strong>
                  {perceptionStatus?.minimum_confidence != null && (
                    <small>最低信頼度 {(perceptionStatus.minimum_confidence * 100).toFixed(1)}%</small>
                  )}
                  {perceptionStatus && (
                    <small>
                      手牌 {perceptionStatus.recognized_counts.hand ?? 0} · ツモ牌 {perceptionStatus.recognized_counts.drawn_tile ?? 0} · ドラ {perceptionStatus.recognized_counts.dora_indicators ?? 0}
                    </small>
                  )}
                  {perceptionStatus?.unknown_slots.length ? (
                    <details><summary>未認識 {perceptionStatus.unknown_slots.length}箇所</summary>
                      <code>{perceptionStatus.unknown_slots.join("\n")}</code>
                    </details>
                  ) : null}
                </div>
              )}
              <p className="capture-scope">通信傍受、ブラウザ注入、非公開API、対局操作は使用しません。</p>
            </section>
          </aside>

          <section className="live-workspace">
            <div className="live-overview">
              <article><span>局</span><strong>{roundLabel}</strong><small>{round?.honba ?? 0}本場</small></article>
              <article><span>巡目</span><strong>{streamState?.turn ?? "—"}</strong><small>残り {streamState?.remaining_tiles ?? "—"}枚</small></article>
              <article><span>状態</span><strong>{streamState?.decision_required ? "判断待ち" : streamState ? "同期中" : "待機"}</strong><small>{streamState?.decision_kind ?? "screen capture"}</small></article>
              <article><span>解析</span><strong>{realtime?.analysis_ms == null ? "—" : `${Math.round(realtime.analysis_ms)}ms`}</strong><small>seq {message?.source_sequence ?? message?.sequence ?? "—"}</small></article>
            </div>

            <section className={`live-hand-card ${streamState?.decision_required ? "decision" : ""}`}>
              <div className="live-section-heading">
                <div><span>OBSERVED HAND</span><h2>認識した手牌</h2></div>
                <small>{currentSession ? `capture: ${currentSession.session_id}` : "画面キャプチャ未接続"}</small>
              </div>
              {streamState?.hand?.length ? (
                <div className="live-tiles">
                  {streamState.hand.map((tile, index) => <span key={`${tileLabel(tile)}-${index}`}>{tileLabel(tile)}</span>)}
                </div>
              ) : (
                <div className="live-placeholder">雀魂の対局画面が認識されると、ここに手牌が表示されます。</div>
              )}
            </section>

            <section className="live-candidates-card">
              <div className="live-section-heading">
                <div><span>RIYU RECOMMEND</span><h2>候補手</h2></div>
                <small>{analysis ? "公開情報のみで解析" : "意思決定局面を待機中"}</small>
              </div>
              {candidates.length > 0 ? (
                <ol className="live-candidate-list">
                  {candidates.map((candidate: Record<string, any>, index: number) => {
                    const action = candidate.action ?? { type: "discard", tile: candidate.discard }
                    const policy = candidate.policy_score
                    return <li key={`${index}-${actionLabel(action)}`}>
                      <b>{String(index + 1).padStart(2, "0")}</b>
                      <strong>{actionLabel(action)}</strong>
                      <span>{policy == null ? `${candidate.shanten?.minimum ?? candidate.shanten ?? "—"}シャンテン` : `Policy ${(policy * 100).toFixed(1)}%`}</span>
                      <span>受け入れ {candidate.ukeire_count ?? "—"}枚</span>
                      <span>放銃損失 {candidate.expected_deal_in_loss == null ? "N/A" : `${Math.round(candidate.expected_deal_in_loss).toLocaleString()}点`}</span>
                    </li>
                  })}
                </ol>
              ) : (
                <div className="live-placeholder">手番または鳴き判断が発生すると、候補手とリスクを表示します。</div>
              )}
            </section>
          </section>
        </div>
      </main>
    </div>
  )
}
