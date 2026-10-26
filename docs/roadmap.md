# ロードマップ

「基盤実装済み」は学習・評価パイプラインが実際に一周することを意味し、強い学習済みモデルやNAGA級性能を意味しない。

## Phase 1 — 基盤（完了）

- [x] Rust Workspace、Python/maturin、GameState、Action、Event
- [x] 観測状態と完全情報型の分離、JSON v1、原子的イベント再生
- [x] Rust/Pythonテスト、アーキテクチャと状態契約

## Phase 2 — ルールエンジン（標準ルール基盤実装済み）

- [x] 通常形・七対子・国士のシャンテン、受け入れ、和了分解
- [x] 主要役・役満、符、翻、点数、ドラ、合法手、各種フリテン
- [x] リーチ、一発、搶槓、流局、複数ロン、責任払い点数移動
- [x] 連荘、本場、供託、親移動、東風・半荘・延長戦
- [x] 5000牌姿の参照実装クロスチェック
- [ ] 流し満貫、責任者の副露履歴からの自動認定、全ローカル差分
- [ ] 配布条件を確認済みの大規模牌譜fixtureによる全役クロスチェック

## Phase 3 — 牌譜とデータ（基盤実装済み）

- [x] 観測者別共通イベントログv1
- [x] 全巡目再生、秘密牌漏洩・不正牌譜の診断
- [x] observation / legal_actions / expert_action / final result のParquet化
- [x] 元牌譜、変換器版、状態schema版の来歴
- [ ] 利用・配布条件を確認したサービス別オフラインアダプター

## Phase 4 — Encoderとモデル（基盤実装済み）

- [x] `observable-planes-v1` 49×34特徴量
- [x] 動的Action特徴、合法手マスク
- [x] Residual backbone、Policy/Rank/Score/Auxiliary Head
- [x] 未学習・未承認checkpointを推奨として返さない推論API

## Phase 5 — 教師あり学習（基盤実装済み）

- [x] Parquet Dataset/DataLoader、牌譜ID単位split
- [x] Policy/Value loss、validation、checkpoint、MLflow
- [x] Policy Accuracy、Rank MAE、Score MSE
- [x] 人間牌譜の権利manifest、SHA-256検証、模倣学習→PPOカリキュラム
- [x] Decision V4軸・ModelOutput V2・階層鳴き/リーチPolicy・候補別放銃/Action Value Head
- [x] factual/counterfactual分離、Opponent Action、Belief consistency、Rule/long-memory、局収支分布
- [x] 全Head masked loss、gradient norm、Cosine scheduler、再開可能checkpoint
- [x] mask-aware評価、統一 `evaluate_state`、全Head教師coverage付きoverfit gate
- [ ] 実Decision V4の100〜1000件overfit gate合格
- [ ] 10k〜50k短期学習でmulti-task干渉・calibration確認
- [ ] 大規模学習、calibration、承認済みモデルカード

## Phase 6 — 牌譜解析UI（実装済み）

- [x] FastAPI + React/Tauriの局面解析
- [x] 手牌、厳密シャンテン・受け入れ・合法手・候補順位
- [x] 厳密値、モデル推定、N/Aの表示区別

## Phase 7 — Simulatorと自己対戦（基盤実装済み）

- [x] Rust四人Simulator、4観測への秘密情報投影
- [x] Self Play Worker、Replay Buffer
- [x] GAE、clipped PPO loss、Arena統計・信頼区間
- [x] CPUプロセス並列worker、決定論的seed、Parquet shard、hash付きmanifest
- [x] Model Policy自己対戦、旧Policy/Value収集、PPO反復更新、再開可能checkpoint
- [x] Model snapshot固定のプロセス並列収集、制限環境のthread fallback
- [ ] 分散GPU学習、長期リーグ評価

## Phase 8 — Opponent / Oracle（基盤実装済み）

- [x] 観測情報だけのテンパイ・待ち・打点Head
- [x] Brier Score / Log Loss
- [x] Oracle専用batch型、Teacher、知識蒸留loss
- [x] 通常Encoderの秘密情報拒否テスト
- [ ] 大規模教師生成、calibration、対戦性能への寄与評価

## Phase 9 — オフライン画像認識（基盤実装済み）

- [x] レイアウト設定、OpenCVテンプレート分類、確信度・不明値
- [x] 保存PNGディレクトリ監視、自作Simulator連携インターフェース
- [x] 合成画像による認識テスト
- [ ] 実画像の注釈、分類モデル学習、複数解像度fixture

## Phase 10 — 許諾済みリアルタイム入力境界（基盤実装済み）

- [x] loopback限定WebSocket、バージョン付きhandshake、メッセージ上限
- [x] 観測者一致、秘密情報不可視、単調増加sequenceの検証
- [x] 厳密計算、評価済みモデル推定、根拠文を同一responseで返却
- [x] 外部adapterの明示的な許諾表明
- [x] 雀魂用許可済みsource profile、source/viewer分離、最新解析優先配信
- [x] read-onlyリアルタイムUIと保存GameStateリプレイCLI
- [x] 雀魂正規化event adapter、共通GameEvent同期、意思決定検出、復旧snapshot
- [x] event重複・欠落・局ID・State assertion検証と自動接続UI
- [ ] 対象サービス運営者から書面で許可された場合のみ、別リポジトリのadapterと結合試験

## 性能目標

NAGA級はコード量や単一accuracyでは宣言しない。固定ルール・固定相手・十分な対局数で平均順位、順位率、和了・放銃、順位点、calibrationと信頼区間を比較し、モデルカードで再現可能にする。

## Phase 11 — 重み付きBeliefと押し引き（着手）

- [x] 候補間で共通の粒子集合と乱数を共有
- [x] 模倣Policyによる最新観測打牌の尤度と粒子重要度重み
- [x] 相手別放銃確率、放銃時期待損失、和了/他家ツモ/横移動/流局/未解決の分離
- [x] Policy/Valueの反事実秘密情報非依存CIテスト
- [x] 公開イベントごとに遷移するstateful Particle FilterとESS resampling
- [x] Rust厳密役・点数を各粒子へ接続
- [x] 相手別危険度・打点分布Head、validation較正ツール、Brier/Log Loss/ECE評価
- [x] train split限定の自動不均衡重み、合法手限定label smoothing、Optimizer付き継続学習
- [x] 公開イベントを相手別attention poolingする時系列latentと聴牌・待ち・危険度Head共有
- [x] Decision V4 / ModelOutput V2 / masked multi-task Loss / checkpoint完成契約
- [x] 階層型response・call type・riichi Headと候補×相手別放銃打点Head
- [ ] 100〜1000 decisionの全Head overfit完成ゲート
- [ ] 大規模学習済みcheckpointでの最終holdout calibration合格
- [x] 1〜4着確率を用いた半荘順位Utilityへの接続
- [x] リーチ/ダマ、鳴き/パスを同一粒子で比較する候補beam
- [x] Arenaの牌効率・模倣・探索あり/なしmodeと必要統計

詳細な監査、依存関係、昇格指標は [NAGA級以降へ向けた実装監査](naga_development_audit.md) を参照する。
