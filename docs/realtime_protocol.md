# ローカル・リアルタイム解析プロトコル

## 境界

`/ws/lab` は、別プロセスが生成した正規化済み観測 `GameState` を解析するloopback限定WebSocketである。画面取得、OCR、通信傍受、非公開API、ブラウザ拡張、対局操作は行わない。

外部サービスとの接続可否は、このAPIの存在からは判断できない。`authorized_external_adapter` は接続元の責任者が必要な許可を確認したという表明であって、サービス運営者の許可を代替しない。参加者の同意だけでは利用規約上の許可にならない場合がある。

## 起動

```bash
python tools/serve_analysis.py --port 8787
```

サーバーは `127.0.0.1` にだけbindする。評価済みcheckpointを使う場合は `--checkpoint model.pt` を追加する。checkpointがなければ厳密シャンテン・受け入れ・合法手だけを返し、モデル値は `N/A` になる。

## Handshake

接続直後の最初のメッセージは次の形式とする。

```json
{
  "type": "hello",
  "protocol": "mahjong-ai-realtime/v1",
  "session_id": "local-research-001",
  "observer": 0,
  "source_kind": "local_simulator"
}
```

`source_kind` は次のいずれか。

- `local_simulator`: 自作Simulator
- `offline_replay`: 保存牌譜の再生
- `authorized_external_adapter`: 運営者の許可を別途確認したadapter

外部adapterでは、さらに次の固定値が必要になる。

```json
"authorization": "source_operator_confirms_authorized"
```

成功時は `type: ready` と、サーバーが受理したobserver・sourceを返す。

## 解析要求

```json
{
  "type": "analyze",
  "sequence": 42,
  "state": { "schema_version": 1 }
}
```

`state` は `docs/game_state.md` の完全な観測状態とする。`sequence` はセッション内で必ず増加させる。observerはHandshakeと一致し、`observer_hand_visible=true`、`opponent_hands_visible=false`、`wall_order_visible=false` でなければならない。

成功responseは次の包絡形式になる。

```json
{
  "type": "analysis",
  "protocol": "mahjong-ai-realtime/v1",
  "session_id": "local-research-001",
  "sequence": 42,
  "result": {
    "shanten": {"status": "exact"},
    "discard_efficiency": {"status": "exact"},
    "model_policy": {"status": "unavailable"},
    "explanations": []
  }
}
```

エラーは `{"type":"error","code":"...","message":"..."}` で返す。不正な要求ではsequenceを消費しない。1メッセージは1 MBまでで、ブラウザ接続はReviewer UIのloopback Originだけを許可する。
