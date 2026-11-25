import { useState } from "react"
import { Tile } from "../../../shared/mahjong-tiles/Tile"

import type { ReplayAnalysis } from "../types/analysis"

import "../style.css"

type AnalysisPageProps = {
  onBack: () => void
  onOpenRealtime: () => void
  analysis: ReplayAnalysis
}

export default function AnalysisPage({
  onBack,
  onOpenRealtime,
  analysis,
}: AnalysisPageProps) {
  const [roundIndex, setRoundIndex] = useState(0)
  const [turnIndex, setTurnIndex] = useState(0)

  const currentRound =
    analysis.rounds[roundIndex]

  const currentTurn =
    currentRound.turns[turnIndex]

  const bestCandidate =
    currentTurn.candidates[0]

  function selectRound(index: number) {
    setRoundIndex(index)

    setTurnIndex(0)
  }

  function selectTurn(index: number) {
    if (
      index < 0 ||
      index >= currentRound.turns.length
    ) {
      return
    }

    setTurnIndex(index)
  }

  function previousTurn() {
    if (turnIndex > 0) {
      setTurnIndex(
        (current) => current - 1,
      )

      return
    }

    if (roundIndex > 0) {
      const previousRoundIndex =
        roundIndex - 1

      const previousRound =
        analysis.rounds[
          previousRoundIndex
        ]

      setRoundIndex(
        previousRoundIndex,
      )

      setTurnIndex(
        previousRound.turns.length - 1,
      )
    }
  }

  function nextTurn() {
    if (
      turnIndex <
      currentRound.turns.length - 1
    ) {
      setTurnIndex(
        (current) => current + 1,
      )

      return
    }

    if (
      roundIndex <
      analysis.rounds.length - 1
    ) {
      setRoundIndex(
        (current) => current + 1,
      )

      setTurnIndex(0)
    }
  }

  const isFirstTurn =
    roundIndex === 0 &&
    turnIndex === 0

  const isLastTurn =
    roundIndex ===
      analysis.rounds.length - 1 &&
    turnIndex ===
      currentRound.turns.length - 1

  return (
    <div className="analysis-page">
      <header className="analysis-header">
        <div>
          <h1>RiYU</h1>

          <p>
            Riichi Yield Utility
          </p>
        </div>

        <div className="analysis-header-meta">
          {analysis.source && (
            <span>{analysis.source.platform === "mahjong_soul" ? "雀魂" : analysis.source.platform}</span>
          )}
          <span>{analysis.rounds.length}局</span>
          <span>{analysis.decisionCount ?? analysis.rounds.reduce((total, round) => total + round.turns.length, 0)}判断</span>
          <span>{analysis.analysisMode === "model_estimate" ? "モデル推定" : "厳密牌効率"}</span>
        </div>

        <div className="analysis-header-actions">
          <button className="header-button" onClick={onOpenRealtime}>Live</button>
          <button className="header-button" onClick={onBack}>New Replay</button>
        </div>
      </header>

      <div className="analysis-layout">
        <aside className="round-sidebar">
          <h2>ROUND</h2>

          {analysis.rounds.map(
            (round, index) => (
              <button
                key={round.id}
                className={`round-item ${
                  index === roundIndex
                    ? "active"
                    : ""
                }`}
                onClick={() => {
                  selectRound(index)
                }}
              >
                {round.label}
              </button>
            ),
          )}
        </aside>

        <main className="board-area">
          <section className="board-card">
            <div className="board-header">
              <div>
                <span className="round-label">
                  {currentRound.label}
                </span>

                <span className="turn-label">
                  {currentTurn.turn}巡目
                </span>
              </div>

              <div className="score-summary">
                <span>
                  自家{" "}
                  {currentTurn.scores.self}
                </span>

                <span>
                  下家{" "}
                  {currentTurn.scores.right}
                </span>

                <span>
                  対面{" "}
                  {
                    currentTurn.scores
                      .opposite
                  }
                </span>

                <span>
                  上家{" "}
                  {currentTurn.scores.left}
                </span>
              </div>
            </div>

            <div className="mahjong-board">
              <div className="opponent top">
                <div className="player-name">
                  対面
                </div>

                <div className="discard-row">
                  {currentTurn.discards.opposite.map(
                    (tile, index) => (
                      <Tile tile={tile} small key={index} />
                    ),
                  )}
                </div>
              </div>

              <div className="opponent left">
                <div className="player-name">
                  上家
                </div>

                <div className="discard-row vertical">
                  {currentTurn.discards.left.map(
                    (tile, index) => (
                      <Tile tile={tile} small key={index} />
                    ),
                  )}
                </div>
              </div>

              <div className="center-info">
                <div>
                  {currentRound.label}
                </div>

                <div>
                  残り{" "}
                  {
                    currentTurn.remainingTiles
                  }
                  枚
                </div>

                <div className="replay-dora">
                  <span>ドラ表示牌</span>
                  {currentTurn.dora.trim().split(/\s+/).filter(Boolean).map((tile, index) => (
                    <Tile tile={tile} small key={index} />
                  ))}
                </div>
              </div>

              <div className="opponent right">
                <div className="player-name">
                  下家
                </div>

                <div className="discard-row vertical">
                  {currentTurn.discards.right.map(
                    (tile, index) => (
                      <Tile tile={tile} small key={index} />
                    ),
                  )}
                </div>
              </div>

              <div className="self-player">
                <div className="player-name">
                  自家
                </div>

                <div className="hand-with-recommendations">
                  {currentTurn.handTiles.map(
                    (item, index) => (
                      <div
                        className={`hand-tile-column ${
                          item.drawn
                            ? "drawn-column"
                            : ""
                        }`}
                        key={`${item.tile}-${index}`}
                      >
                        <div className="recommendation-gauge">
                          <div
                            className={`recommendation-gauge-fill ${
                              item.recommendation ===
                              bestCandidate.score
                                ? "high"
                                : ""
                            }`}
                            style={{
                              height: `${Math.max(
                                item.recommendation,
                                4,
                              )}%`,
                            }}
                          />
                        </div>

                        <Tile tile={item.tile} className={item.drawn ? "hand-tile drawn-tile" : "hand-tile"} />
                      </div>
                    ),
                  )}
                </div>
              </div>
            </div>
          </section>

          <section className="timeline-card">
            <button
              type="button"
              onClick={previousTurn}
              disabled={isFirstTurn}
            >
              ←
            </button>

            <div className="timeline">
              {currentRound.turns.map(
                (turn, index) => (
                  <span
                    key={turn.turn}
                    className={
                      index === turnIndex
                        ? "active"
                        : ""
                    }
                    onClick={() => {
                      selectTurn(index)
                    }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" ||
                        event.key === " "
                      ) {
                        selectTurn(index)
                      }
                    }}
                  >
                    {turn.turn}
                  </span>
                ),
              )}
            </div>

            <button
              type="button"
              onClick={nextTurn}
              disabled={isLastTurn}
            >
              →
            </button>
          </section>
        </main>

        <aside className="analysis-panel">
          <div className="recommendation-card">
            <span className="section-label">
              RiYU RECOMMEND
            </span>

            <Tile tile={bestCandidate.tile} className="recommended-tile" />

            <div className="recommended-score">
              <span>{analysis.analysisMode === "model_estimate" ? "Recommendation" : "Tile efficiency"}</span>

              <strong>
                {bestCandidate.score}%
              </strong>
            </div>

            <div className="main-confidence-bar">
              <div
                className="main-confidence-fill"
                style={{
                  width: `${bestCandidate.score}%`,
                }}
              />
            </div>
          </div>

          <div className="candidate-card">
            <h2>
              CANDIDATES
            </h2>

            <div className="candidate-list">
              {currentTurn.candidates.map(
                (
                  candidate,
                  index,
                ) => (
                  <div
                    className={`candidate-row ${
                      index === 0
                        ? "recommended"
                        : ""
                    }`}
                    key={candidate.tile}
                  >
                    <div className="candidate-title">
                      <div className="candidate-title-left">
                        <Tile tile={candidate.tile} small />

                        {index === 0 && (
                          <span className="candidate-badge">
                            BEST
                          </span>
                        )}
                      </div>

                      <span className="candidate-score">
                        {
                          candidate.score
                        }
                        %
                      </span>
                    </div>

                    <div className="candidate-score-bar">
                      <div
                        className="candidate-score-fill"
                        style={{
                          width: `${candidate.score}%`,
                        }}
                      />
                    </div>

                    <div className="candidate-metrics">
                      <div>
                        <span>
                          Shanten
                        </span>

                        <strong>
                          {
                            candidate.shanten
                          }
                        </strong>
                      </div>

                      <div>
                        <span>
                          Ukeire
                        </span>

                        <strong>
                          {
                            candidate.ukeire
                          }
                        </strong>
                      </div>

                      <div>
                        <span>
                          Deal-in
                        </span>

                        <strong>
                          {candidate.dealInRisk == null ? "N/A" : `${candidate.dealInRisk}%`}
                        </strong>
                      </div>

                      <div>
                        <span>
                          EV
                        </span>

                        <strong>
                          {candidate.expectedValue == null
                            ? "N/A"
                            : `${candidate.expectedValue >= 0 ? "+" : ""}${candidate.expectedValue}`}
                        </strong>
                      </div>
                    </div>
                  </div>
                ),
              )}
            </div>
          </div>

        </aside>
      </div>
    </div>
  )
}
