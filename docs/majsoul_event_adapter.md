# 雀魂イベントAdapter契約 v1

## 実装範囲

RiYU本体は、運営者の許可を得たcollectorまたは保存イベントから、正規化済みの公開観測を
受け取る。雀魂クライアントの通信傍受、非公開API呼び出し、ブラウザ注入、画面取得、入力操作は
このadapterに含めない。

```text
許可済みcollector / 保存イベント
              │ majsoul-normalized-event/v1
              ▼
      MahjongSoulEvent adapter
              │ Rust GameEvent JSON
              ▼
        Rust GameState::apply_event
              │
       legal_actions / Decision V3+
              │ 意思決定時のみ推論
              ▼
       realtime viewer WebSocket
```

Rustの`GameEvent`と`GameState`が正本であり、雀魂固有の値は
`python/mahjong_ai/adapters/majsoul.py`より内側へ持ち込まない。

## 接続

WebSocket endpointは`/ws/realtime/events/source`。Handshakeでは次を指定する。

```json
{
  "type": "hello",
  "protocol": "mahjong-ai-realtime/v1",
  "session_id": "authorized-majsoul-session",
  "observer": 0,
  "source_kind": "authorized_external_adapter",
  "authorization": "source_operator_confirms_authorized",
  "authorization_reference": "operator-permission-record",
  "source_service": "majsoul",
  "source_profile": "majsoul-normalized-event/v1"
}
```

保存イベントの再生では`source_kind=offline_replay`を使い、許可表明と許可参照は不要。

## Event envelope

すべてのイベントは次の包絡形式を取る。

```json
{
  "type": "event",
  "event": {
    "event_id": "east-1:draw:14",
    "sequence": 14,
    "round_id": "east-1-0",
    "name": "draw",
    "data": {
      "player": 0,
      "tile": "6s",
      "rinshan": false
    }
  }
}
```

- `event_id`: 再送時にも不変な一意ID
- `sequence`: 半荘内で0から連続するsource連番
- `round_id`: 局を一意に識別するID
- `name`: 下表のイベント名
- `data`: イベント固有payload

同じ`event_id`と同じ内容の再送は`duplicate`として無視する。同じIDで内容が異なる場合、
連番が欠落した場合、古い未知イベントが到着した場合は状態を変更せずエラーにする。

対応する`name`:

- `round_start`
- `draw`
- `discard`
- `chi`, `pon`, `kan`
- `riichi`
- `dora_reveal`
- `pass`
- `ron`, `tsumo`
- `draw_game`
- `round_end`

牌はcore JSONの`{"kind":"s6","red":false}`に加えて、`6s`、赤五の`0m`、
字牌の`1z`〜`7z`をadapter境界で受け付ける。他家ツモの`tile`は必ず`null`とする。
`discard.data.tsumogiri_known`は省略時`true`である。画面上で他家ツモ切りを確定できない場合は
`tsumogiri=false, tsumogiri_known=false`とし、公開状態とモデル入力で不明値として保持する。

## 意思決定と配信

全イベントはGameStateへ適用するが、AI推論は観測者に`pass`以外の合法手が存在するときだけ
実行する。自摸後の打牌・リーチ・暗槓・加槓・ツモ、他家打牌へのチー・ポン・大明槓・ロン、
搶槓ロンが同じ合法手生成を利用する。

意思決定がないイベントはviewerへ`state_update`、意思決定があるイベントは`analysis`を配信する。
どちらにも`state`要約が付き、局、巡目、手牌、残り牌、フェーズ、意思決定状態をUIが表示する。

## 再接続と整合性確認

途中局からの復旧は、Rustで検証可能なGameState snapshotと、次に期待するsource連番を送る。

```json
{
  "type": "snapshot",
  "round_id": "east-2-1",
  "next_source_sequence": 83,
  "state": {"schema_version": 1}
}
```

将来のVision Adapterまたはcollector監視値は、状態を変更しない`state_check`で比較できる。

```json
{
  "type": "state_check",
  "assertion": {
    "round_id": "east-2-1",
    "turn": 32,
    "hand_count": 14,
    "dora_count": 1,
    "current_actor": 0,
    "phase": "awaiting_discard"
  }
}
```

不一致では`valid=false`とfield別の期待値・観測値を返し、GameStateは変更しない。

## 保存イベントによるテスト

```bash
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py --port 8787

PYTHONPATH=python .venv/bin/python tools/replay_majsoul_events.py \
  docs/examples/majsoul-events.jsonl \
  --linger 30
```

React UIは2秒間隔でsourceを検出し、雀魂イベントsourceへ自動接続する。
