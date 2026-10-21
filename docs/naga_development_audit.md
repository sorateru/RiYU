# NAGA級以降へ向けた実装監査とロードマップ

この文書は2026-10-20時点のコード監査結果である。「基盤がある」と「学習・較正済みで強い」を区別する。現時点でNAGA級性能は未検証であり、主評価は独立Arenaとする。

## 実装監査

| 領域 | 状態 | 再利用する実装 | 主要な不足 |
|---|---|---|---|
| ルール・合法手・点数 | 実装済み（一部ローカル差分を除く） | Rust `engine/core` | 流し満貫等と大規模fixture照合 |
| 観測情報分離 | 実装済み | Rust観測投影、`FeatureEncoder`、公開イベント列 | CIでPolicy/Valueそのものの反事実秘密情報テストが不足していた |
| 49×34 Policy/Value | 基盤実装済み、性能未承認 | `MahjongNetwork` | 順位分布の意思決定Utilityへの直接接続、独立Encoder化 |
| 公開時系列Transformer | 基盤実装済み、性能未承認 | `SequenceMahjongNetwork` | 相手別latent、打点分布、危険度較正 |
| 相手状態推定 | 不完全 | 聴牌・待ち・シャンテンHead、条件付き手牌生成 | 一貫した `P(H|O)`、時系列Bayes更新、ESS監視、resampling |
| 不完全情報探索 | 不完全 | 共通乱数による候補間共通世界、短期rollout | belief重み、完全な鳴き・役・点数、順位Utility |
| 押し引き | 不完全 | rolloutの和了・放銃・他家ツモ | 相手別較正、放銃時点数分布、横移動・流局の十分な長さでの推定 |
| リーチ・鳴き | 不完全 | 動的合法手、successor比較、教師Policy | リーチ/ダマと鳴く/見送るの局終了までの反実仮想比較 |
| 半荘Value | 不完全 | 4順位logit、最終点、半荘Simulator | 順位確率をUtilityに変換した候補比較、オーラス条件の較正 |
| データ分割 | 不完全 | 半荘ID単位train/validation | 開発中に見ない固定holdout、時期・対局者分離、20万→100万の学習曲線 |
| Arena | 基盤実装済み | 同一seed、4座席配置、平均順位CI | AI種別拡張、全要求統計、十分な対局数と昇格判定 |
| 自己対戦/PPO | 基盤実装済み、後順位 | BC初期値、PPO部品 | belief/value完成後のBC KL付き移行 |

## 今回追加した第一段階

`BeliefRolloutSearch` に重み付き粒子posteriorを追加した。粒子集合は候補行動ごとに作り直さず、一度だけ生成して全候補で共有する。

相手の最新打牌 `a` を観測したとき、各粒子 `H_i` の重みを概念的に次で更新する。

```text
w_i' = w_i * max(P_policy(a | H_i, O), epsilon)
w_i  = w_i' / sum_j(w_j')
```

`ImitationDiscardLikelihood` は既存Policyを尤度モデルとして使う。そのPolicyへ渡すのは、行動した相手から見た公開情報と、その粒子における当該相手自身の手牌だけである。他家2人の仮想手牌と仮想牌山は入力しない。

現在のstateless推論では、現在の粒子手牌と矛盾しやすい全過去打牌を重複して評価せず、相手ごとの最新打牌だけを尤度に使う。次段階ではイベント到着ごとにdraw/discard遷移を行うstateful Particle Filterへ置き換える。

rollout出力には次を追加した。

- 自分の和了、放銃、他家ツモ、他家間横移動、流局、探索打切りの確率
- 和了時期待得点と放銃時期待失点
- 相手別放銃確率と相手別放銃時期待失点
- 粒子数とEffective Sample Size (ESS)

短期探索の打切りは流局と混ぜず `unresolved_probability` とする。これらはすべて未較正のモデル推定であり、厳密値ではない。

## 依存関係順ロードマップ

### M1 — Stateful opponent belief

1. 公開イベントごとに粒子のdraw/discard/callを遷移する。
2. BC Policy尤度で逐次重み更新し、ESS低下時だけ系統resamplingする。
3. 粒子ごとにRustの厳密シャンテン・待ち・役・点数を計算する。
4. 実手牌はラベルにだけ使い、hand negative log-likelihood、tile marginal Brier/Log Loss、ESS、coverageをholdoutで測る。

合格条件は、単純な残り牌一様priorよりholdoutの尤度と下流の相手別放銃Brierが改善し、秘密情報反事実テストを通ること。

### M2 — 相手別押し引きと打点

1. 候補打牌×相手ごとの放銃hazardを学習する。
2. 放銃時点数を分布Headとして学習し、危険度と条件付き損失を分離する。
3. `P(deal-in) * E(loss | deal-in)` を自分の和了、他家ツモ、横移動、流局と統合する。
4. 筋、壁、ワンチャンス、見え枚数、手出し/ツモ切り、リーチ前後巡目は固定判定でなく特徴にする。

合格条件は相手別Brier、Log Loss、ECE、点数MAE/CRPS、およびArena放銃率・平均放銃点の改善。

### M3 — 半荘順位Utility

1. 1〜4着確率を明示的に出し、確率和と較正を検査する。
2. 現在順位、点差、局、親、本場、供託を条件に点棒変化を順位Utilityへ写像する。
3. 局収支EVと半荘Utilityを併記し、南場・オーラスでは後者を意思決定に使用する。

合格条件は順位分布Log Loss/Brier/ECEと、南場・オーラスsubset Arenaの平均順位改善。

### M4 — リーチ・鳴き反実仮想探索

同一belief粒子と同一乱数列上で、リーチ/ダマ、鳴く/見送るを局終了まで比較する。役、ドラ、副露、相手の降り、手牌固定を含むRust精算へ接続する。

### M5 — データとArena昇格ゲート

1. train/validation/final-holdoutを半荘単位で固定し、可能なら時期と対局者も分離する。
2. 20万、50万、100万判断のvalidationと同一条件Arenaを記録する。
3. 旧モデル、牌効率、模倣のみ、探索あり/なしを同じ配牌・牌山と座席交換で比較する。
4. 95%信頼区間で旧モデルを上回るまでcheckpointを未承認にする。

### M6 — 自己対戦

M1〜M5の完了後、強いBCを初期値としてKL制約付きPPOへ移る。KL係数はArenaと方策崩壊監視に基づいて段階的に弱める。

## 推定値の表示契約

- `exact`: Rustルールエンジンが観測局面から一意に計算した値。
- `model_estimate`: 学習モデルまたは重み付きrolloutの推定。モデルID、較正状態、サンプル数、ESSを付ける。
- `unvalidated_estimate`: 配線・研究用だが独立holdout/Arena未合格の値。推奨根拠には使わない。
- `unavailable`: 必要な入力、checkpoint、検証がない値。

実際の他家手牌、実際の牌山、将来自摸は教師ラベルまたはSimulator内部だけに置き、通常Policy/Valueの入力契約には追加しない。
