# ローカル・リアルタイム解析プロトコル

## 境界

`/ws/lab` は、別プロセスが生成した正規化済み観測 `GameState` を解析するloopback限定WebSocketである。画面取得、OCR、通信傍受、非公開API、ブラウザ拡張、対局操作は行わない。

連続的なsourceと閲覧UIを分離する場合は次を使う。

- `/ws/realtime/source`: 正規化観測を送信する単一source
- `/ws/realtime/view/{session_id}`: 解析済み結果だけを受け取るread-only viewer
- `/realtime/sessions`: 現在のsource一覧。許可記録の内容自体は返さない

sourceの解析待ちとviewerの配信待ちはどちらも最大1件で、入力またはUIが遅れた場合は
古い未処理・未表示結果を破棄して最新局面を優先する。sourceには破棄したsequenceと
置換先sequenceを`type=superseded`で通知する。実行中の1件は中断せず、無制限に推論threadを増やさない。

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

## 雀魂プロファイル

雀魂由来のsourceを名乗る場合、Handshakeに次を追加する。

```json
{
  "source_service": "majsoul",
  "source_profile": "majsoul-normalized-observation/v1"
}
```

`source_kind=authorized_external_adapter`では、さらに運営者の許可を識別する
`authorization_reference`が必須である。この値はセッション一覧や解析responseには返さない。

```json
{
  "type": "hello",
  "protocol": "mahjong-ai-realtime/v1",
  "session_id": "authorized-majsoul-research",
  "observer": 0,
  "source_kind": "authorized_external_adapter",
  "authorization": "source_operator_confirms_authorized",
  "authorization_reference": "permission-record-id",
  "source_service": "majsoul",
  "source_profile": "majsoul-normalized-observation/v1"
}
```

実装されているのは「許可済みsourceから正規化観測を受け取る契約」までである。
雀魂クライアントの通信取得、画面取得、ブラウザへの注入は含まない。

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

`/ws/realtime/source`の結果には`result.realtime`が追加され、解析時間、
状態fingerprint、前回からのシャンテン・最良候補・合法手数の変化を返す。fingerprintは
状態同一性確認用のSHA-256で、元の非公開データは含まない。

## 保存状態による結合試験

雀魂との実接続を行わず、保存済みの観測`GameState` JSONLをsourceとして再生できる。

```bash
PYTHONPATH=python .venv/bin/python tools/replay_realtime_states.py \
  saved-observations.jsonl \
  --session-id majsoul-offline-replay \
  --interval 0.25
```

CLIはloopback WebSocket以外への送信を拒否する。

エラーは `{"type":"error","code":"...","message":"..."}` で返す。不正な要求ではsequenceを消費しない。1メッセージは1 MBまでで、ブラウザ接続はReviewer UIのloopback Originだけを許可する。
