import { useRef, useState } from "react";

const METRICS = [
  ["average_rank", "平均順位", "rank"],
  ["average_score_delta", "平均半荘収支", "points"],
  ["first_rate", "トップ率", "rate"],
  ["fourth_rate", "ラス率", "rate"],
  ["win_rate_per_seat_hand", "和了率", "rate"],
  ["deal_in_rate_per_seat_hand", "放銃率", "rate"],
  ["riichi_rate_per_seat_hand", "リーチ率", "rate"],
  ["furo_rate_per_seat_hand", "副露率", "rate"],
] as const;

const POLICIES: Record<string, string> = {
  tile_efficiency: "牌効率ベースライン",
  imitation: "模倣学習",
  hybrid_no_search: "期待値（探索なし）",
  hybrid_search: "期待値（探索あり）",
  hybrid_search_high_accuracy: "期待値（高精度探索）",
};
const COUNT = new Intl.NumberFormat("ja-JP");
const SIGNED_POINTS = new Intl.NumberFormat("ja-JP", { maximumFractionDigits: 1, signDisplay: "exceptZero" });

function object(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) return value as Record<string, unknown>;
  throw new Error("Arena結果の形式を確認してください。");
}

function finite(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  throw new Error("Arena結果に数値が不足しています。");
}

function parseReport(value: unknown) {
  const root = object(value);
  if (root.format !== "mahjong-ai-arena/v2") {
    throw new Error("evaluate_arena.py が出力した Arena v2 のJSONを選択してください。");
  }
  const config = object(root.config);
  const policies = object(config.policies);
  const models = object(root.models);
  const summary = object(root.summary);
  const paired = object(root.paired);
  const sides = (["a", "b"] as const).map((side) => {
    const policy = policies[side];
    const modelId = object(object(models[side]).metadata).model_id;
    const results = object(summary[side]);
    return {
      title: side.toUpperCase(),
      model: policy === "tile_efficiency" ? "牌効率ベースライン" :
        typeof modelId === "string" ? modelId : "モデル名不明",
      policy: typeof policy === "string" ? (POLICIES[policy] ?? policy) : "方策不明",
      metrics: METRICS.map(([key]) => finite(results[key])),
    };
  });
  const blocks = finite(config.seed_blocks);
  const halfGames = finite(config.matches);
  const hands = finite(config.hands);
  if ([blocks, halfGames, hands].some((count) => !Number.isSafeInteger(count) || count < 1)) {
    throw new Error("対戦数または局数が正しくありません。");
  }
  const rankCI = paired.average_rank_difference_a_minus_b_95ci;
  if (!Array.isArray(rankCI) || rankCI.length !== 2) {
    throw new Error("平均順位差の信頼区間が不足しています。");
  }
  const interval = rankCI.map(finite);
  return {
    sides, blocks, halfGames, hands, interval,
    rankDifference: finite(paired.average_rank_difference_a_minus_b),
    pointDifference: finite(paired.average_score_delta_difference_a_minus_b),
    verdict: blocks > 1 && root.verdict === "a_better" ? "Aが優勢" :
      blocks > 1 && root.verdict === "b_better" ? "Bが優勢" : "優劣は未確定",
  };
}

function formatMetric(value: number, unit: (typeof METRICS)[number][2]) {
  if (unit === "rate") return `${(value * 100).toFixed(1)}%`;
  if (unit === "points") return `${SIGNED_POINTS.format(value)}点`;
  return value.toFixed(3);
}

export function ArenaPanel() {
  const [loaded, setLoaded] = useState<{ name: string; report: ReturnType<typeof parseReport> }>();
  const [error, setError] = useState("");
  const latestLoad = useRef(0);

  async function loadReport(file: File) {
    const request = ++latestLoad.current;
    setError("");
    setLoaded(undefined);
    try {
      const report = parseReport(JSON.parse(await file.text()));
      if (request === latestLoad.current) setLoaded({ name: file.name, report });
    } catch (caught) {
      if (request === latestLoad.current) {
        setError(caught instanceof SyntaxError ? "JSONを読み取れません。Arenaの出力ファイルを選択してください。" :
          caught instanceof Error ? caught.message : "結果の読み込みに失敗しました。");
      }
    }
  }

  const report = loaded?.report;
  return <div className="mode-panel arena-mode">
    <section className="intro-panel">
      <h1>AI対戦結果</h1>
      <p>座席を入れ替えて行った東南戦の結果から、2つのAI方策を比較します。</p>
      <label htmlFor="arena-report">Arena結果のJSONを選択</label>
      <input id="arena-report" type="file" accept=".json,application/json" onChange={(event) => {
        const file = event.currentTarget.files?.[0];
        event.currentTarget.value = "";
        if (file) void loadReport(file);
      }} />
      {error && <p role="alert" className="error">{error}</p>}
    </section>
    {report && <section aria-labelledby="arena-verdict">
      <h2 id="arena-verdict">{report.verdict}</h2>
      <p>{COUNT.format(report.halfGames)}半荘 · {COUNT.format(report.hands)}局 · {COUNT.format(report.blocks)}シード</p>
      <p className="arena-source">{loaded.name}</p>
      <div className="arena-scroll">
        <table className="arena-metrics">
          <thead><tr><th scope="col">指標</th>{report.sides.map((side) =>
            <th scope="col" key={side.title}>{side.title} · {side.model}
              {side.model !== side.policy && <small>{side.policy}</small>}
            </th>)}</tr></thead>
          <tbody>{METRICS.map(([key, label, unit], index) => <tr key={key}>
            <th scope="row">{label}</th>{report.sides.map((side) =>
              <td key={side.title}>{formatMetric(side.metrics[index], unit)}</td>)}
          </tr>)}</tbody>
        </table>
      </div>
      <p>平均順位差（A−B） {report.rankDifference.toFixed(3)} · 得点差（A−B） {SIGNED_POINTS.format(report.pointDifference)}点</p>
      <p>{report.blocks > 1
        ? `平均順位差の95%信頼区間 ${report.interval.map((value) => value.toFixed(3)).join(" ～ ")}（負ならAが上位）`
        : "1シードのみの試行です。順位差の信頼区間と優劣の判定には複数シードが必要です。"}</p>
    </section>}
  </div>;
}
