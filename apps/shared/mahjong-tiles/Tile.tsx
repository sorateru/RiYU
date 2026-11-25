import "./tile.css";

type TileValue = string | { kind?: string; red?: boolean };

const honors = [
  ["east", "東"], ["south", "南"], ["west", "西"], ["north", "北"],
  ["white", "白"], ["green", "發"], ["red", "中"],
] as const;
const suits: Record<string, string> = { m: "萬", p: "筒", s: "索" };
const svgUrls = import.meta.glob<string>("./svg/*.svg", {
  eager: true, query: "?url&no-inline", import: "default",
});

// GameState uses suit-first kinds; replay uses rank-first labels (赤5p, 東).
export function tileCode(tile: TileValue | null | undefined): string | undefined {
  const kind = (typeof tile === "string" ? tile : tile?.kind)?.trim();
  if (kind === "back") return kind;
  const honor = honors.findIndex(([id, label]) => kind === id || kind === label);
  if (honor >= 0) return `${honor + 1}z`;
  if (!kind) return undefined;
  if (/^[1-7]z$/.test(kind)) return kind;

  const red = typeof tile === "string" ? kind.startsWith("赤") : tile?.red === true;
  const value = kind.replace(/^赤/, "");
  const suitFirst = /^([mps])([0-9])$/.exec(value);
  const rankFirst = /^([0-9])([mps])$/.exec(value);
  const suit = suitFirst?.[1] ?? rankFirst?.[2];
  const rank = suitFirst?.[2] ?? rankFirst?.[1];
  if (!suit || !rank) return undefined;
  return `${red && rank === "5" ? "0" : rank}${suit}`;
}

export function tileLabel(tile: TileValue | null | undefined): string {
  const code = tileCode(tile);
  if (!code) return typeof tile === "string" && tile ? tile : "?";
  if (code === "back") return "裏向きの牌";
  const [rank, suit] = code;
  if (suit === "z") return honors[Number(rank) - 1][1];
  return `${rank === "0" ? "赤5" : rank}${suits[suit]}`;
}

export function Tile({ tile, className = "", small = false }: {
  tile: TileValue | null | undefined;
  className?: string;
  small?: boolean;
}) {
  const label = tileLabel(tile);
  const src = svgUrls[`./svg/${tileCode(tile)}.svg`];
  const classes = `mahjong-tile ${small ? "mahjong-tile-small" : ""} ${className}`;
  return src ? <img className={classes} src={src} alt={label} title={label}
    width={128} height={176} draggable={false} />
    : <span className={`${classes} mahjong-tile-unknown`} title={label}>{label}</span>;
}
