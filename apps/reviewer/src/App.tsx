import { useEffect, useMemo, useRef, useState } from "react";

type Json = Record<string, any>;
type Mode = "play" | "review" | "live";

const API = import.meta.env.VITE_REVIEWER_API ?? "http://127.0.0.1:8000";
const WS_API = API.replace(/^http/, "ws");
const WINDS: Record<string, string> = { east: "東", south: "南", west: "西", north: "北" };
const HONORS: Record<string, string> = {
  east: "東", south: "南", west: "西", north: "北", white: "白", green: "發", red: "中",
};
const TILE_ORDER = [
  ...Array.from({ length: 9 }, (_, index) => `m${index + 1}`),
  ...Array.from({ length: 9 }, (_, index) => `p${index + 1}`),
  ...Array.from({ length: 9 }, (_, index) => `s${index + 1}`),
  "east", "south", "west", "north", "white", "green", "red",
];

function tileLabel(tile: Json | undefined): string {
  if (!tile) return "?";
  if (HONORS[tile.kind]) return HONORS[tile.kind];
  const suit = tile.kind?.[0];
  const number = tile.kind?.slice(1);
  const suffix = suit === "m" ? "萬" : suit === "p" ? "筒" : "索";
  return `${tile.red ? "赤" : ""}${number}${suffix}`;
}

function Tile({ tile, small = false }: { tile: Json; small?: boolean }) {
  return <span className={`tile ${tile.red ? "red-tile" : ""} ${small ? "small-tile" : ""}`}>{tileLabel(tile)}</span>;
}

function statusLabel(value: Json | undefined): string {
  if (!value || value.status === "unavailable") return "N/A";
  return value.status === "exact" ? "厳密計算" : "モデル推定";
}

function actionLabel(action: Json): string {
  const tile = action.discard ?? action.tile ?? action.called_tile ?? action.action?.tile ?? action.action?.called_tile;
  const labels: Record<string, string> = {
    discard: "打牌", riichi: "リーチ", chi: "チー", pon: "ポン", kan: "カン",
    ron: "ロン", tsumo: "ツモ", pass: "パス", abortive_draw: "九種九牌",
  };
  return `${labels[action.type] ?? action.type}${tile ? ` ${tileLabel(tile)}` : ""}`;
}

function ReviewPanel() {
  const [input, setInput] = useState("{}");
  const [analysis, setAnalysis] = useState<Json | null>(null);
  const [error, setError] = useState("");
  const state = useMemo(() => {
    try { return JSON.parse(input); } catch { return null; }
  }, [input]);

  async function analyze() {
    setError("");
    setAnalysis(null);
    if (!state) { setError("JSONを解析できません。"); return; }
    const response = await fetch(`${API}/analyze`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ state }),
    });
    const body = await response.json();
    if (!response.ok) { setError(body.detail ?? "解析に失敗しました。"); return; }
    setAnalysis(body);
  }

  const shanten = analysis?.shanten?.value?.minimum;
  const efficiency = analysis?.discard_efficiency?.value ?? [];
  const modelPolicy = analysis?.model_policy?.value ?? [];

  return <div className="mode-panel">
    <section className="intro-panel">
      <p className="eyebrow">OFFLINE REVIEW LAB</p>
      <h1>局面解析</h1>
      <p>保存牌譜または自作シミュレーターの局面を検討します。AI対局とは独立した機能です。</p>
    </section>
    <section className="input-panel">
      <label htmlFor="state">GameState JSON</label>
      <textarea id="state" value={input} onChange={(event) => setInput(event.target.value)} spellCheck={false} />
      <button onClick={analyze}>局面を解析</button>
      {error && <p className="error">{error}</p>}
    </section>
    {analysis && <div className="results">
      <section className="summary">
        <article><span>シャンテン</span><strong>{shanten ?? "N/A"}</strong><small>{statusLabel(analysis.shanten)}</small></article>
        <article><span>残り牌</span><strong>{state?.remaining_tiles ?? "—"}</strong><small>公開局面</small></article>
        <article><span>合法手</span><strong>{analysis.legal_actions?.value?.length ?? 0}</strong><small>{statusLabel(analysis.legal_actions)}</small></article>
      </section>
      <section>
        <h2>手牌</h2>
        <div className="tiles">{state?.hand?.map((tile: Json, index: number) => <Tile tile={tile} key={index} />)}</div>
      </section>
      <section>
        <div className="section-heading"><h2>候補打牌</h2><em>シャンテン・受け入れは厳密計算</em></div>
        <ol className="ranking">{efficiency.map((candidate: Json) => <li key={JSON.stringify(candidate.discard)}>
          <strong>{tileLabel(candidate.discard)}</strong><span>{candidate.shanten.minimum}シャンテン</span><span>受け入れ {candidate.ukeire_count}枚</span>
        </li>)}</ol>
      </section>
      <section>
        <div className="section-heading"><h2>AI期待値順位</h2><em>{statusLabel(analysis.model_policy)}</em></div>
        {modelPolicy.length ? <ol className="ranking">{modelPolicy.map((item: Json, index: number) => <li key={index}>
          <strong>{actionLabel(item.action)}</strong>
          {item.estimated_action_value != null && <span>評価値 {Math.round(item.estimated_action_value).toLocaleString()}</span>}
          {item.shanten != null && <span>{item.shanten}シャンテン・受け入れ {item.ukeire_count ?? "—"}枚</span>}
          {item.rollout_expected_points != null && <span>探索期待点 {Math.round(item.rollout_expected_points).toLocaleString()}</span>}
          {item.win_probability != null && <span>
            {item.rollout_depth}巡内 和了 {(item.win_probability * 100).toFixed(1)}%・放銃 {((item.deal_in_probability ?? 0) * 100).toFixed(1)}%・他家和了 {((item.opponent_win_probability ?? 0) * 100).toFixed(1)}%
          </span>}
          <small>Policy {item.policy_score.toFixed(4)}</small>
        </li>)}</ol> : <p className="na">{analysis.model_policy?.reason ?? "評価済みモデルがないためN/Aです。"}</p>}
      </section>
    </div>}
  </div>;
}

function LiveAnalysisPanel() {
  const [sessionId, setSessionId] = useState("majsoul-offline-replay");
  const [status, setStatus] = useState<"disconnected" | "connecting" | "connected" | "offline">("disconnected");
  const [message, setMessage] = useState<Json | null>(null);
  const [streamState, setStreamState] = useState<Json | null>(null);
  const [sessions, setSessions] = useState<Json[]>([]);
  const [error, setError] = useState("");
  const socketRef = useRef<WebSocket | null>(null);
  const autoConnectRef = useRef(true);

  useEffect(() => {
    void refreshSessions(true);
    const timer = window.setInterval(() => void refreshSessions(true), 2_000);
    return () => {
      window.clearInterval(timer);
      socketRef.current?.close();
    };
  }, []);

  async function refreshSessions(autoConnect = false) {
    setError("");
    try {
      const response = await fetch(`${API}/realtime/sessions`);
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "セッション一覧を取得できません。");
      const available = body.sessions ?? [];
      setSessions(available);
      if (autoConnect && autoConnectRef.current && socketRef.current == null && available.length > 0) {
        const eventSource = available.find((item: Json) => item.source_profile === "majsoul-normalized-event/v1");
        connect((eventSource ?? available[0]).session_id);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "セッション一覧を取得できません。");
    }
  }

  function disconnect(disableAutoConnect = true) {
    if (disableAutoConnect) autoConnectRef.current = false;
    const socket = socketRef.current;
    socketRef.current = null;
    socket?.close();
    setStatus("disconnected");
  }

  function connect(selected?: string) {
    const target = (selected ?? sessionId).trim();
    if (!target) { setError("セッションIDを入力してください。"); return; }
    autoConnectRef.current = true;
    disconnect(false);
    setSessionId(target);
    setMessage(null);
    setStreamState(null);
    setError("");
    setStatus("connecting");
    const socket = new WebSocket(`${WS_API}/ws/realtime/view/${encodeURIComponent(target)}`);
    socketRef.current = socket;
    const acceptPayload = (payload: Json) => {
      if (payload.type === "analysis") {
        setMessage(payload);
        if (payload.state) setStreamState(payload.state);
      } else if (payload.type === "state_update") {
        setStreamState(payload.state ?? null);
      }
    };
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as Json;
      if (payload.type === "subscribed") {
        setStatus("connected");
        if (payload.latest) acceptPayload(payload.latest);
      } else if (payload.type === "analysis" || payload.type === "state_update") {
        setStatus("connected");
        acceptPayload(payload);
      } else if (payload.type === "source_offline") {
        setStatus("offline");
      } else if (payload.type === "error") {
        setError(payload.message ?? "リアルタイム接続に失敗しました。");
      }
    };
    socket.onerror = () => setError("リアルタイムWebSocketへ接続できません。");
    socket.onclose = () => {
      if (socketRef.current === socket) {
        socketRef.current = null;
        setStatus((current) => current === "offline" ? current : "disconnected");
      }
    };
  }

  const analysis = message?.result;
  const realtime = analysis?.realtime;
  const candidates = analysis?.model_policy?.value ?? [];
  const exactCandidates = analysis?.discard_efficiency?.value ?? [];
  const round = streamState?.round;
  const roundLabel = round ? `${WINDS[round.wind] ?? round.wind}${round.hand_number}局 ${round.honba}本場` : "対局待機中";
  const liveHeadline = status === "disconnected" || status === "connecting"
    ? "Waiting for Mahjong Soul"
    : streamState?.decision_required ? "Your turn" : streamState ? "Game detected" : "Waiting for game";
  const connectionLabels = {
    disconnected: "未接続", connecting: "接続中", connected: "受信中", offline: "入力終了",
  };

  return <div className="mode-panel live-mode">
    <section className="intro-panel live-intro">
      <div>
        <p className="eyebrow">AUTHORIZED LIVE REVIEW</p>
        <h1>{liveHeadline}</h1>
        <p>許可済みの正規化観測ストリームを閲覧します。画面取得、通信傍受、非公開API、対局操作は行いません。</p>
      </div>
      <span className={`connection-badge status-${status}`}>{connectionLabels[status]}</span>
    </section>
    <section className="live-connect">
      <div className="session-entry">
        <label>セッションID<input value={sessionId} onChange={(event) => setSessionId(event.target.value)} /></label>
        <button onClick={() => connect()} disabled={status === "connecting"}>閲覧接続</button>
        <button className="secondary-button" onClick={() => disconnect(true)}>切断</button>
        <button className="secondary-button" onClick={() => refreshSessions(false)}>入力一覧を更新</button>
      </div>
      {sessions.length > 0 && <div className="live-sessions">{sessions.map((session) => <button
        key={session.session_id} onClick={() => connect(session.session_id)}
      ><strong>{session.session_id}</strong><small>{session.source_service ?? session.source_kind}・seq {session.last_sequence ?? "—"}</small></button>)}</div>}
      {error && <p className="error">{error}</p>}
      <p className="scope-note">雀魂用sourceは、運営者の許可記録を持つ外部adapterか、保存済みGameStateのoffline replayだけを受理します。</p>
    </section>
    {streamState && <section className={`live-game-state ${streamState.decision_required ? "decision-required" : ""}`}>
      <div><span>現在局</span><strong>{roundLabel}</strong></div>
      <div><span>巡目</span><strong>{streamState.turn ?? "—"}</strong></div>
      <div><span>状態</span><strong>{streamState.decision_required ? "意思決定が必要" : "局面同期中"}</strong></div>
      <div className="live-hand"><span>手牌</span><div className="tiles">{(streamState.hand ?? []).map((tile: Json, index: number) => <Tile tile={tile} small key={index} />)}</div></div>
    </section>}
    {analysis && <div className="results live-results">
      <section className="summary live-summary">
        <article><span>Sequence</span><strong>{message.sequence}</strong><small>{message.session_id}</small></article>
        <article><span>解析時間</span><strong>{realtime?.analysis_ms?.toFixed(0) ?? "—"}<i>ms</i></strong><small>{realtime?.source_profile ?? "normalized stream"}</small></article>
        <article><span>シャンテン</span><strong>{analysis.shanten?.value?.minimum ?? "N/A"}</strong><small>{statusLabel(analysis.shanten)}</small></article>
        <article><span>候補</span><strong>{analysis.legal_actions?.value?.length ?? 0}</strong><small>{realtime?.state_changed ? "局面更新" : "同一局面"}</small></article>
      </section>
      <section>
        <div className="section-heading"><h2>AI候補</h2><em>Policy・Value・Riskは個別表示</em></div>
        {candidates.length ? <ol className="ranking live-ranking">{candidates.map((item: Json, index: number) => <li key={index}>
          <strong>{actionLabel(item.action)}</strong>
          <span className="policy-meter"><i style={{ width: `${Math.max(0, Math.min(1, item.policy_score ?? 0)) * 100}%` }} /><b>{item.policy_score == null ? "N/A" : `${(item.policy_score * 100).toFixed(1)}%`}</b></span>
          <span>局収支 {item.action_round_value == null ? "N/A" : `${Math.round(item.action_round_value).toLocaleString()}点`}</span>
          <span>半荘価値 {item.action_hanchan_value?.toFixed(3) ?? "N/A"}</span>
          <span>放銃損失 {Math.round(item.expected_deal_in_loss ?? 0).toLocaleString()}点</span>
        </li>)}</ol> : <p className="na">{analysis.model_policy?.reason ?? "評価済みモデルがないためN/Aです。"}</p>}
      </section>
      <section>
        <div className="section-heading"><h2>厳密牌効率</h2><em>公開情報のみ</em></div>
        <ol className="ranking">{exactCandidates.map((candidate: Json) => <li key={JSON.stringify(candidate.discard)}>
          <strong>{tileLabel(candidate.discard)}</strong><span>{candidate.shanten.minimum}シャンテン</span><span>受け入れ {candidate.ukeire_count}枚</span>
        </li>)}</ol>
      </section>
    </div>}
  </div>;
}

function PlayerArea({ seat, game }: { seat: number; game: Json }) {
  const observation = game.observation;
  const player = observation.players[seat];
  const isHuman = seat === game.human_seat;
  const riverDiscards = player.discards.filter((discard: Json) => discard.called_by == null);
  return <article className={`player-area ${isHuman ? "human-area" : ""}`}>
    <div className="player-heading">
      <strong>{isHuman ? "あなた" : `AI ${seat + 1}`}</strong>
      <span>{game.scores[seat].toLocaleString()}点</span>
      {player.riichi !== "none" && <b>立直</b>}
    </div>
    {player.melds.length > 0 && <div className="player-row">
      <span className="row-label">副露</span>
      <div className="melds">{player.melds.map((meld: Json, index: number) => <div className="meld" key={index}>
        {meld.tiles.map((tile: Json, tileIndex: number) => <Tile tile={tile} small key={tileIndex} />)}
      </div>)}</div>
    </div>}
    <div className="player-row river-row">
      <span className="row-label">河 <small>{riverDiscards.length}枚</small></span>
      <div className="river">{riverDiscards.map((discard: Json, index: number) => <Tile tile={discard.tile} small key={index} />)}</div>
    </div>
  </article>;
}

function PlayPanel() {
  const [game, setGame] = useState<Json | null>(null);
  const [seat, setSeat] = useState(0);
  const [seed, setSeed] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function request(path: string, options: RequestInit = {}) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}${path}`, options);
      const body = response.status === 204 ? null : await response.json();
      if (!response.ok) throw new Error(body?.detail ?? "対局処理に失敗しました。");
      return body;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "対局処理に失敗しました。");
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function startGame() {
    if (game?.session_id) {
      await request(`/local-play/sessions/${game.session_id}`, { method: "DELETE" });
    }
    const numericSeed = seed.trim() ? Number(seed) : undefined;
    if (numericSeed !== undefined && (!Number.isSafeInteger(numericSeed) || numericSeed < 0)) {
      setError("シードは0以上の整数にしてください。");
      return;
    }
    const body = await request("/local-play/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ human_seat: seat, seed: numericSeed }),
    });
    if (body) setGame(body);
  }

  async function playAction(action: Json) {
    if (!game || busy) return;
    const body = await request(`/local-play/sessions/${game.session_id}/actions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (body) setGame(body);
  }

  const observation = game?.observation;
  const round = game?.round;
  const concealed = observation ? [...observation.hand].sort((left: Json, right: Json) => TILE_ORDER.indexOf(left.kind) - TILE_ORDER.indexOf(right.kind)) : [];

  return <div className="mode-panel play-mode">
    <section className="intro-panel play-intro">
      <div>
        <p className="eyebrow">LOCAL HANCHAN</p>
        <h1>AI対局</h1>
        <p>あなた1人とAI3人のローカル東南戦です。局面解析・推奨表示とは接続していません。</p>
      </div>
      <div className="start-controls">
        <label>開始席
          <select value={seat} onChange={(event) => setSeat(Number(event.target.value))} disabled={busy}>
            <option value={0}>東家</option><option value={1}>南家</option><option value={2}>西家</option><option value={3}>北家</option>
          </select>
        </label>
        <label>シード（任意）<input value={seed} onChange={(event) => setSeed(event.target.value)} inputMode="numeric" placeholder="ランダム" /></label>
        <button onClick={startGame} disabled={busy}>{game ? "新しい半荘" : "半荘を開始"}</button>
      </div>
      {error && <p className="error">{error}</p>}
    </section>

    {game && observation && <>
      <section className="match-strip">
        <div><small>局</small><strong>{WINDS[round.wind]}{round.hand_number}局</strong></div>
        <div><small>本場</small><strong>{round.honba}</strong></div>
        <div><small>供託</small><strong>{round.riichi_sticks}</strong></div>
        <div><small>残り</small><strong>{observation.remaining_tiles}</strong></div>
        <div><small>モデル</small><strong className="model-name">{game.opponent_model_id}</strong></div>
      </section>

      <section className="table-shell">
        <div className="opponents">
          {[0, 1, 2, 3].filter((player) => player !== game.human_seat).map((player) => <PlayerArea seat={player} game={game} key={player} />)}
        </div>
        <PlayerArea seat={game.human_seat} game={game} />
        <div className="hand-zone">
          <div className="section-heading"><h2>あなたの手牌</h2><em>対局中は候補手を表示しません</em></div>
          <div className="tiles hand-tiles">
            {concealed.map((tile: Json, index: number) => <Tile tile={tile} key={index} />)}
            {observation.drawn_tile && <span className="drawn-tile"><Tile tile={observation.drawn_tile} /></span>}
          </div>
        </div>
      </section>

      {game.status === "match_ended" ? <section className="result-panel">
        <p className="eyebrow">MATCH COMPLETE</p><h2>半荘終了</h2>
        <div className="final-ranking">{game.ranks.map((rank: number, player: number) => <div key={player}>
          <strong>{rank}位</strong><span>{player === game.human_seat ? "あなた" : `AI ${player + 1}`}</span><b>{game.scores[player].toLocaleString()}点</b>
        </div>)}</div>
      </section> : <section className="action-panel">
        <div className="section-heading"><h2>あなたの操作</h2><em>{busy ? "AIが思考中…" : `${game.legal_actions.length}個の合法手`}</em></div>
        <div className="action-buttons">{game.legal_actions.map((action: Json, index: number) => <button
          className={`action-${action.type}`} disabled={busy} onClick={() => playAction(action)} key={`${index}-${JSON.stringify(action)}`}
        >{actionLabel(action)}</button>)}</div>
      </section>}
    </>}
  </div>;
}

export function App() {
  const [mode, setMode] = useState<Mode>("live");
  return <main>
    <nav className="mode-tabs" aria-label="機能選択">
      <button className={mode === "play" ? "active" : ""} onClick={() => setMode("play")}>AI対局</button>
      <button className={mode === "review" ? "active" : ""} onClick={() => setMode("review")}>局面解析</button>
      <button className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}>リアルタイム</button>
    </nav>
    <div hidden={mode !== "play"}><PlayPanel /></div>
    <div hidden={mode !== "review"}><ReviewPanel /></div>
    <div hidden={mode !== "live"}><LiveAnalysisPanel /></div>
  </main>;
}
