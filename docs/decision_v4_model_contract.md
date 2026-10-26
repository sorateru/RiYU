# Decision V4 / Mahjong ModelOutput V2 固定契約

この文書は大規模学習前に凍結する入力・教師・Head・Loss・checkpoint・推論APIの契約である。

- architecture: `sequence-multitask-v2`
- model output: `mahjong-model-output/v2`
- decision schema: `mahjong-ai-sequence-decision/v4`
- observable sequence feature: `observable-events-rules-memory-v3`
- label: `sequence-labels-v2`

## 入力と軸

- action軸`A`は`legal_actions`の順序と完全に一致する。
- 相手軸は観測者から見た`[下家, 対面, 上家]`である。
- モデル入力は公開局面、合法手、公開event、rule featuresだけである。
- Transformerは直近96 eventを読み、それより前のeventは固定長summaryに圧縮する。
- 相手の実手牌、非公開自摸、牌山順、局後の結果、future-draw教師、Simulatorのhidden stateは入力に入れない。
- rule featuresはルール種別、赤牌、東風/半荘、西入、複数ロン、ヤメ・トビ等を表現し、旧データはunknown/defaultで読む。

## Decision V4の教師

V3の列に加え、次を保存する。

- `factual_round_result_by_action[A]`, `factual_round_result_mask[A]`
- `counterfactual_round_result_by_action[A]`, `counterfactual_round_result_mask[A]`
- `factual_action_round_delta[A]`, `factual_action_round_delta_mask[A]`
- `counterfactual_action_round_delta[A]`, `counterfactual_action_round_delta_mask[A]`
- `factual_action_final_rank[A]`, `factual_action_final_score[A]`, `factual_action_hanchan_mask[A]`
- `counterfactual_action_final_rank[A]`, `counterfactual_action_final_score[A]`, `counterfactual_action_hanchan_mask[A]`
- `opponent_next_discard[3]` + mask
- `opponent_next_response[3]` + mask。classは`pass/call/riichi/win`
- `opponent_next_call_type[3]` + mask。classは`chi/pon/kan`
- `opponent_tile_count[3][34]` + opponent mask

実牌譜で得られる候補別教師は選択された1候補だけをfactual mask=trueにする。
未選択候補は0ラベルではなくmask=falseである。Simulator教師はcounterfactual列だけへ保存し、
factualとcounterfactualのmaskの重複はvalidatorとDataLoaderの両方で拒否する。

## MahjongModelOutput V2

- Policy: `policy_logits[B,A]`, `response_logits[B,3]`, `call_type_logits[B,3]`, `riichi_logits[B,A]`
- state value: `state_round_value[B]`, `state_hanchan_value[B]`
- action value: `action_round_value[B,A]`, `action_hanchan_value[B,A]`
- advantage: `action_round_advantage[B,A]`, `action_hanchan_advantage[B,A]`
- 候補別局収支分布: `action_round_distribution_logits[B,A,13]`
- 従来value/result: `rank_logits[B,4]`, `final_score[B]`, `round_delta[B,4]`, `round_result_logits[B,A,5]`
- belief: `tenpai_logits[B,3]`, `wait_logits[B,3,34]`, `shanten_logits[B,3,8]`, `future_draw_logits[B,34]`
- risk: `deal_in_logits[B,A,3]`, `deal_in_point_logits[B,A,3,8]`
- opponent action: `opponent_next_discard_logits[B,3,34]`, `opponent_response_logits[B,3,4]`, `opponent_call_type_logits[B,3,3]`
- hidden-belief拡張: `opponent_tile_count_logits[B,3,34,5]`, `opponent_latents[B,3,D]`
- conditioning diagnostics: `memory_summary[B,E]`, `rule_embedding[B,E]`

`Q(s,a)-V(s)`はAdvantageとして明示的に返す。Policyは強者模倣priorとして保持し、
`evaluate_state`は`policy_score`, `value_score`, `risk_score`, `search_score`を別々返す。

局収支13 binの境界は`-18000, -12000, -8000, -5200, -3000, -1000, 1000, 3000, 5200, 8000, 12000, 18000`点で、回帰Headと並行して学習する。

## Loss V2

従来の`policy/rank/score/round/tenpai/wait/shanten/draw/deal/dealpoint/result/response/call_type/riichi`に、次を追加する。

- `state_round_value`, `state_hanchan_value`: masked Huber
- `action_round_value`, `action_hanchan_value`: 候補別masked Huber
- `action_round_distribution`: 候補別masked CE
- `opponent_discard`, `opponent_response`, `opponent_call_type`: 相手別masked CE
- `opponent_tile_count`: 相手×牌種別masked CE
- `belief_consistency`: tenpai確率、シャンテン0以下確率、待ち存在確率間の軟い二乗誤差

全Lossは個別重みとvalid maskを持つ。有効教師0件は微分可能な0とし、NaNや擬似負例を作らない。
半荘Utilityは`HanchanUtility` protocolで差し替え可能で、既定は`RankScoreUtility`である。

## Checkpointと互換性

checkpointにはmodel/optimizer/scheduler/RNG/global stepに加え、上記5 version、git commit、seed、
hyperparameters、dataset fingerprint/manifest hash、Utility定義を保存する。V1 checkpointからは
CNN、Transformer、Action Encoder、Policy、従来Value、opponent latent/riskをwarm-startし、V2追加Headだけを
初期化する。Head構成が同一のV2 checkpointはOptimizerとschedulerを含め完全resumeできる。

## 現時点の境界

Offline RLは未学習だが、`OfflineRLSignals`でstate/action value、advantage、behavior actionをAWR/IQL Trainerへ渡せる。
Simulatorによるcounterfactualラベル生成、完全なhidden-hand生成モデル、rolloutとPolicy/Value/Riskの最終統合はこの契約の次段階である。
