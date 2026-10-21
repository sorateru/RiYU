import "../style.css"

const candidates = [
  {
    tile: "5m",
    score: 72,
    shanten: 1,
    ukeire: 28,
    dealInRisk: 3.2,
    expectedValue: 820,
  },
  {
    tile: "8p",
    score: 18,
    shanten: 1,
    ukeire: 24,
    dealInRisk: 7.8,
    expectedValue: 510,
  },
  {
    tile: "1s",
    score: 7,
    shanten: 2,
    ukeire: 41,
    dealInRisk: 1.1,
    expectedValue: 120,
  },
]

const handTiles = [
  { tile: "1m", recommendation: 4 },
  { tile: "2m", recommendation: 3 },
  { tile: "3m", recommendation: 2 },
  { tile: "5m", recommendation: 72 },
  { tile: "5m", recommendation: 4 },
  { tile: "6m", recommendation: 3 },
  { tile: "7m", recommendation: 2 },
  { tile: "3p", recommendation: 1 },
  { tile: "4p", recommendation: 1 },
  { tile: "5p", recommendation: 2 },
  { tile: "6s", recommendation: 1 },
  { tile: "7s", recommendation: 1 },
  { tile: "8s", recommendation: 1 },
  { tile: "9p", recommendation: 3, drawn: true },
]

export default function AnalysisPage() {
  return (
    <div className="analysis-page">
      <header className="analysis-header">
        <div>
          <h1>RiYU</h1>
          <p>Riichi Mahjong Analysis</p>
        </div>

        <button className="header-button">
          新しい牌譜を解析
        </button>
      </header>

      <div className="analysis-layout">
        <aside className="round-sidebar">
          <h2>局一覧</h2>

          <button className="round-item active">
            東1局
          </button>

          <button className="round-item">
            東2局
          </button>

          <button className="round-item">
            東3局
          </button>

          <button className="round-item">
            東4局
          </button>

          <button className="round-item">
            南1局
          </button>

          <button className="round-item">
            南2局
          </button>
        </aside>

        <main className="board-area">
          <section className="board-card">
            <div className="board-header">
              <div>
                <span className="round-label">
                  東1局
                </span>

                <span className="turn-label">
                  8巡目
                </span>
              </div>

              <div className="score-summary">
                <span>自家 25000</span>
                <span>下家 25000</span>
                <span>対面 25000</span>
                <span>上家 25000</span>
              </div>
            </div>

            <div className="mahjong-board">
              <div className="opponent top">
                <div className="player-name">
                  対面
                </div>

                <div className="discard-row">
                  <span>1m</span>
                  <span>9m</span>
                  <span>3p</span>
                  <span>東</span>
                  <span>7s</span>
                </div>
              </div>

              <div className="opponent left">
                <div className="player-name">
                  上家
                </div>

                <div className="discard-row vertical">
                  <span>2p</span>
                  <span>白</span>
                  <span>6m</span>
                </div>
              </div>

              <div className="center-info">
                <div>東1局</div>
                <div>残り 46枚</div>
                <div>ドラ 4p</div>
              </div>

              <div className="opponent right">
                <div className="player-name">
                  下家
                </div>

                <div className="discard-row vertical">
                  <span>9s</span>
                  <span>中</span>
                  <span>3m</span>
                </div>
              </div>

              <div className="self-player">
                <div className="player-name">
                  自家
                </div>

                <div className="hand-with-recommendations">
                  {handTiles.map((item, index) => (
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
                            item.recommendation >= 50
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

                      <span
                        className={`hand-tile ${
                          item.drawn
                            ? "drawn-tile"
                            : ""
                        }`}
                      >
                        {item.tile}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section className="timeline-card">
            <button>←</button>

            <div className="timeline">
              <span>1</span>
              <span>2</span>
              <span>3</span>
              <span>4</span>
              <span>5</span>
              <span>6</span>
              <span>7</span>
              <span className="active">8</span>
              <span>9</span>
              <span>10</span>
            </div>

            <button>→</button>
          </section>
        </main>

        <aside className="analysis-panel">
          <div className="recommendation-card">
            <span className="section-label">
              RiYU推奨
            </span>

            <div className="recommended-tile">
              5m
            </div>

            <div className="recommended-score">
              <span>AI評価</span>
              <strong>72%</strong>
            </div>

            <div className="main-confidence-bar">
              <div
                className="main-confidence-fill"
                style={{
                  width: "72%",
                }}
              />
            </div>
          </div>

          <div className="candidate-card">
            <h2>候補比較</h2>

            <div className="candidate-list">
              {candidates.map(
                (candidate, index) => (
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
                        <span className="candidate-tile">
                          {candidate.tile}
                        </span>

                        {index === 0 && (
                          <span className="candidate-badge">
                            推奨
                          </span>
                        )}
                      </div>

                      <span className="candidate-score">
                        {candidate.score}%
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
                          シャンテン
                        </span>

                        <strong>
                          {candidate.shanten}
                        </strong>
                      </div>

                      <div>
                        <span>
                          受入枚数
                        </span>

                        <strong>
                          {candidate.ukeire}
                        </strong>
                      </div>

                      <div>
                        <span>
                          放銃リスク
                        </span>

                        <strong>
                          {candidate.dealInRisk}%
                        </strong>
                      </div>

                      <div>
                        <span>
                          期待局収支
                        </span>

                        <strong>
                          {candidate.expectedValue >= 0
                            ? "+"
                            : ""}
                          {candidate.expectedValue}
                        </strong>
                      </div>
                    </div>
                  </div>
                ),
              )}
            </div>
          </div>

          <button className="detail-button">
            詳細解析を表示
          </button>
        </aside>
      </div>
    </div>
  )
}