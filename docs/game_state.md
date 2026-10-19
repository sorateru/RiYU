# GameState JSON契約 v1

## 基本原則

`GameState` は一人のプレイヤーから観測可能な局面であり、通常モデルへの唯一の入力元となる。`schema_version` は現在 `1`。JSONの列挙値とフィールド名は `snake_case` である。

Rust型が正本であり、この文書は意味上の契約を説明する。完全な機械検証は `game_state_from_json` またはPythonの `validate_state_json` を使用する。

## 主なフィールド

| フィールド | 意味 |
|---|---|
| `rules` | 独立したルール設定。現在は四人麻雀のみ |
| `information.observer` | この観測を行うプレイヤー（0〜3） |
| `round` | 場風、局数、本場、供託、親。未開始時は `null` |
| `scores` | プレイヤー0〜3の点数 |
| `hand` | 観測者の手牌。ツモ牌は含めない |
| `drawn_tile` | 観測者のツモ牌。なければ `null` |
| `dora_indicators` | 公開済みドラ表示牌 |
| `players` | 4人分の河、副露、リーチ状態 |
| `ippatsu_eligible` | 4人の一発成立可能状態 |
| `remaining_tiles` | ライブ山の残り牌数 |
| `turn` | 現局のDRAWイベント数 |
| `current_actor` | 現在打牌または嶺上ツモを行うプレイヤー。応答待ちは `null` |
| `phase` | 状態機械の現在フェーズ |
| `pending_kan` | 搶槓応答中の未確定暗槓・加槓。それ以外は `null` |
| `event_history` | 対局全体で連番になった観測イベント |

`last_event` は `event_history` の末尾と同一でなければならない。`next_event_sequence` は履歴長と一致し、次に適用できるイベント番号を示す。

## 牌

牌は次の形を取る。

```json
{"kind": "m5", "red": true}
```

`kind` は `m1`〜`m9`、`p1`〜`p9`、`s1`〜`s9`、`east`、`south`、`west`、`north`、`white`、`green`、`red` の34種。`red: true` は `m5`、`p5`、`s5` だけに許可される。

## イベント

すべてのイベントは `sequence` と `type` を持つ。主な例:

```json
{
  "sequence": 1,
  "type": "draw",
  "player": 0,
  "tile": {"kind": "p9", "red": false},
  "rinshan": false
}
```

他家のツモは情報を漏らさないよう `tile: null` とする。観測者自身のツモは牌が必須である。

対応イベント:

- `round_start`
- `draw`
- `discard`
- `chi`
- `pon`
- `kan`
- `riichi`
- `dora_reveal`
- `ron`
- `tsumo`
- `draw_game`
- `round_end`
- `pass`（ロン可能時の見逃しフリテンを記録）

イベント列は `round_start` から始まる。`round_end` は `ron`、`tsumo`、`draw_game` の結果イベント後だけ適用できる。打牌後に鳴きがなければ、次プレイヤーの `draw` が暗黙に応答窓を閉じる。暗槓・加槓は搶槓応答中だけ `pending_kan` に置き、搶槓がなければ `rinshan: true` のDRAW時に副露として確定する。

## 公開情報と完全情報

通常の `GameState` JSONには以下のキーが存在してはならない。

- `concealed_hands`
- `live_wall`
- `dead_wall`

これらはSimulator/Oracle専用の `CompleteGameState.private` にだけ存在する。通常推論のシリアライズ関数は `GameState` しか受け取らない。

## 現在保証する整合性

- プレイヤーID、局番号、ルール設定の範囲
- イベント連番と履歴の一致
- 状態フェーズと手番の一致
- 観測者/他家のDRAW情報公開範囲
- 赤牌を含む既知牌の上限枚数
- 副露の枚数・同種牌・順子構造
- 最新打牌と鳴き元の一致
- リーチ宣言と成立、1000点供託
- 捨て牌・一時・リーチ後フリテン
- 終局結果と `round_end` の一致
- シャンテン、役、フリテン、待ち、符・翻・点数、動的合法手

厳密計算の対応範囲と残るローカル差分は `docs/rules_phase2.md` を正本とする。未学習モデルに依存する和了率、放銃率、順位期待値は `unavailable` として返す。
