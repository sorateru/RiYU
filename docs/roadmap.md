# ロードマップ

## Phase 1 — 基盤（今回）

- [x] Rust Workspace、Python/maturin設定
- [x] Tile、Player、RuleConfig
- [x] 観測可能 `GameState` と完全情報型の分離
- [x] Action、Event、イベント再生
- [x] JSON v1とPyO3境界
- [x] 不変条件・原子的更新の単体テスト
- [x] アーキテクチャ・状態契約・セットアップ文書

完了判定は `fmt`、`clippy -D warnings`、Workspaceテスト、Pythonバインディングのビルドとスモークテストが通ること。実行環境にツールチェーンがない場合は未検証として明記する。

## Phase 2 — ルールエンジン

1. 通常形、七対子、国士無双のシャンテン数
2. 有効牌と既知牌を差し引いた受け入れ枚数
3. 和了形、役、翻、符、点数（責任払い等のルール差を含む）
4. 打牌・リーチ・チー・ポン・各カン・ロン・ツモ・パスの合法手生成
5. フリテン（同巡、捨て牌、リーチ後見逃し）
6. 流局、本場、供託、連荘、親移動
7. 完全情報イベントから4人それぞれの観測イベントへの安全な投影
8. 公開ルールfixtureと他実装とのクロスチェック

精度基準として、既知の牌姿fixture、境界点数、赤牌を含む副露、複数ロン、各種流局を回帰テスト化する。未実装ルールは無視せず、明示的な `UnsupportedRule` と失敗テストで可視化する。

## Phase 3 — 牌譜とデータ

- 牌譜形式ごとのアダプターから共通イベントへ変換
- 全巡目の再生検証と不正牌譜の診断
- observation / legal_actions / expert_action / final result のParquet化
- 元牌譜、変換器版、ルール版、スキーマ版の来歴記録

## Phase 4 — Encoderとモデル

- バージョン付き34種特徴量Encoder
- 合法手マスクと動的Action集合
- Residual backbone、Policy/Value Head
- ラベルが存在するAuxiliary Headだけを有効化
- 未学習チェックポイントを解析推奨として返さない推論API

## Phase 5 — 教師あり学習

- Dataset/DataLoader、再現可能なsplit
- Policy/Value loss、validation、checkpoint
- MLflowによる設定・データ・コード・指標の記録
- calibration、Policy Accuracy、Value Error

## Phase 6 — 牌譜解析UI

- 局面、厳密シャンテン・受け入れ・合法手
- 学習済みモデルの候補順位とモデル出力
- 厳密計算、推定値、N/Aを視覚的に区別

## Phase 7〜9

- 高速Simulator、並列自己対戦、Replay Buffer、PPO/Actor-Critic比較
- Opponent Model、Oracle Training、知識蒸留と情報漏洩テスト
- オフライン画像認識、確信度付きGameState変換、練習Simulator連携

「NAGA級」は名称や単一accuracyでは判定せず、固定ルール・固定相手・十分な対局数で平均順位、順位率、和了/放銃、順位点と信頼区間を比較して判断する。

