# アーキテクチャ

## 目的と境界

`mahjong-ai` は、局面の事実を扱う決定論的な麻雀エンジンと、事実から評価値を推定するAIを分離する。AIは状態を直接変更せず、候補の `Action` を返す。ルールエンジンだけが合法性を判定し、確定した `GameEvent` を `GameState` に適用する。

```text
牌譜 / Simulator / 保存画像 / 許諾済みAdapter
                   │ 観測イベントまたは正規化GameState
                   ▼
         Rust event/state engine ── JSON/PyO3 ── Python encoder/model
                   │                                  │
                   │ 合法手                           │ logits/value
                   └──────── policy mask ─────────────┘
```

## 正本と依存方向

Rustの `mahjong-core` を局面データ契約の正本とする。Python、UI、牌譜パーサー、画像認識が独自の状態表現を正本として持つことは禁止する。それぞれの入力はイベント列またはバージョン付きJSONへ変換する。

依存方向は次の通り。

1. `tile` / `player` / `rules`: 値オブジェクト
2. `action` / `event`: 意思決定と確定事実
3. `state`: イベント適用、状態不変条件
4. `bindings/python`: Rust APIの薄い公開層
5. `data` / `ai` / `analysis` / `perception`: バインディングの利用側

AI、OpenCV、WebSocket、React/Tauriをコアへ依存させない。これによりルールテストはGPUやUIなしで高速に実行できる。

## 不完全情報の分離

通常推論が受け取る `GameState` は、観測者の手牌、公開された河・副露・ドラ、点数・局情報だけを持つ。他家の手牌、山順、王牌順を表すフィールド自体がない。

完全情報は `CompleteGameState { public, private }` でのみ表す。`PrivateGameState` を受け取る関数を通常推論APIに追加しない。将来のOracle学習は別パッケージでこの型を明示的に使用し、蒸留先には公開情報エンコーダーの出力だけを渡す。

## 状態更新

`GameState::apply_event` は次を順に実行する。

1. イベント連番を検査する。
2. 状態を複製し、複製側へイベントを適用する。
3. 公開情報、局面フェーズ、牌枚数、履歴を再検証する。
4. 成功時だけ元状態を置き換える。

これにより、不正イベントが途中まで状態を書き換えることを防ぐ。状態はイベント履歴を保持し、同じ観測者・ルール・イベント列から再現できる。

`Draw.tile` は観測者自身なら必須、他家なら必ず `null` とする。同じ完全情報イベントを各プレイヤー向けの観測イベントへ投影する処理は `FourPlayerSimulator` 境界に置き、4つの `GameState` へ自家ツモだけを開示する。

## ActionとEvent

`Action` はAIや人間の「意図」であり、まだ事実ではない。`EventKind` は検証後に確定した事実である。たとえばリーチは宣言と1000点供託の成立を分離し、`declared` と `accepted` の2イベントで表現する。

Phase 2では `legal_actions(state)` が動的な `Vec<Action>` を生成する。Policy Headは固定IDを直接実行せず、この配列に対するlogitを返す。選択後もルールエンジンで再検証する。

## 実装済み主要コンポーネント

- `engine/core`: 厳密ルール、点数、観測投影、四人Simulator
- `data`: 観測者別牌譜を正規化しParquetへ保存
- `ai`: Feature Encoder、Residual Network、動的Policy/Value/Auxiliary Head
- `training`: 教師あり、MLflow、自己対戦、Replay Buffer、PPO部品、Oracle蒸留
- `evaluation`: 固定条件Arena、信頼区間、回帰ベンチマーク
- `analysis`: 厳密値とモデル推定値を区別した構造化JSON、順序付きリアルタイムセッション
- `perception`: 保存画像から候補観測を作り、確信度と不明値を保持する
- `apps/reviewer`: FastAPI HTTP/WebSocket + React/Tauri。未学習値や算出不能値は `N/A` とする

リアルタイムWebSocket境界は取得元から独立している。入力元は観測者向けに正規化された `GameState`、またはadapter境界の公開観測eventを送信する。eventはサービス別mapperで共通`GameEvent`へ変換し、Rustの同じ状態機械へ適用する。サービス側は観測者番号、可視性フラグ、source連番、event ID、局IDを再検証する。ライブ取得は、ユーザーが明示的に許可した画面を読み取るread-onlyのVision Adapterに限定し、通信傍受、非公開API、入力操作は行わない。保存牌譜については、別配布のChrome拡張が牌譜閲覧画面で既に受信済みの記録をdecodeし、オフラインEventLog境界へ渡す。詳細は `docs/realtime_protocol.md`、`docs/majsoul_vision_adapter.md`、`docs/browser_extension.md` に定める。

## バージョニング

JSONには `schema_version` を必須とする。破壊的変更では値を増やし、読み込み時の移行関数とfixtureテストを用意する。Feature EncoderはJSONとは独立した `encoder_version` と特徴量仕様を持たせ、チェックポイントに必ず記録する。

## 性能方針

Phase 1は正確性を優先してイベント適用時に状態を複製する。自己対戦のプロファイルで支配的と判明した場合だけ、検証済みトランザクションや差分更新へ置換する。計測なしに安全性を落とす最適化は行わない。
