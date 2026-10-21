# Decision V3 / Mahjong Model V1 固定契約

この文書は大規模学習前に凍結する教師・Head・Loss・checkpoint・推論契約である。
`decision/v3` と `mahjong-model-output/v1` の意味を変更するときは既存名を再利用せず、
必ず新しいversionを発行する。

## 座席・action軸

- すべての座席配列は観測者相対順 `[self, shimocha, toimen, kamicha]` とする。
- 相手だけの配列は `[shimocha, toimen, kamicha]` の長さ3とする。
- action軸は `legal_actions` の順番と完全一致する。
- `chosen_action_index` は `legal_actions` 内の唯一の正解indexとする。
- tile軸は34種、待ち牌はmulti-hot、点数分布は固定8 binとする。

## 1 decisionの必須列

- `observation`, `public_events`
- `legal_actions`, `chosen_action`, `chosen_action_index`
- `opponent_tenpai[3]`, `opponent_wait[3][34]`, `opponent_shanten[3]`
- `round_result`（`self_win`, `deal_in`, `opponent_tsumo`, `horizontal`, `draw`）
- `round_delta[4]`, `final_rank`, `final_score`
- `future_draw[34]`
- `factual_deal_in_by_action[A][3]`
- `factual_deal_in_points_by_action[A][3]`
- `factual_deal_in_mask[A][3]`
- `counterfactual_deal_in_by_action[A][3]`
- `counterfactual_deal_in_points_by_action[A][3]`
- `counterfactual_deal_in_mask[A][3]`
- `label_masks`

実牌譜から直接観測できる候補別放銃教師は、実際に選ばれた打牌またはリーチ打牌の
rowだけである。未選択候補へ安全を意味する0を入れてはならず、maskをfalseにする。
Simulatorが生成した候補別反実仮想はcounterfactual列だけに保存し、factual列へ混ぜない。

## ModelOutput V1

完成版forwardは次を返す。

- `policy_logits[B,A]`
- `response_logits[B,3]`（pass, call, win）
- `call_type_logits[B,3]`（chi, pon, kan）
- `riichi_logits[B,A]`（discard候補条件付き）
- `rank_logits[B,4]`
- `final_score[B]`（`(score-25000)/10000`）
- `round_delta[B,4]`（点差/10000）
- `tenpai_logits[B,3]`
- `wait_logits[B,3,34]`
- `shanten_logits[B,3,8]`（-1..6）
- `future_draw_logits[B,34]`
- `deal_in_logits[B,A,3]`
- `deal_in_point_logits[B,A,3,8]`
- `round_result_logits[B,A,5]`

flat `policy_logits` はaction確率の最終互換面として残す。response/call/riichi Headは
希少な鳴きとリーチを独立に学習し、推論時のflat policy再較正と反実仮想探索に使う。

## Loss V1

`total = sum(lambda[name] * loss[name])` とし、各lossを個別に記録する。

- policy: 合法手限定label-smoothed CE
- response/call_type: masked CE
- riichi: discard候補条件付きmasked BCE
- rank/result/shanten: masked CE
- final_score: 正規化targetへのHuber
- round_delta: 正規化targetへのHuber
- tenpai/wait/future_draw/deal_in: masked BCE-with-logits
- deal_in_point: `deal_in=true` かつlabel validなcellだけのCE

すべてのlossは `sum(mask * element_loss) / max(sum(mask), 1)` とする。maskが0件なら
そのHeadはgradient 0の有限なlossを返す。欠損ラベルを負例へ変換してはならない。

## Checkpoint V1

必須keyは `model_state`, `optimizer_state`, `scheduler_state`, `epoch`, `global_step`,
`architecture_version`, `model_output_version`, `decision_schema_version`, `dataset_version`,
`feature_version`, `label_version`,
`git_commit`, `seed`, `model_config`, `training_config`, `dataset_fingerprint`,
`dataset_manifest_hashes`, `rng_state`, `selection` とする。

## 大規模学習開始ゲート

全Head shape、全Head backward、finite、missing mask、合法手mask、相手seat mapping、
秘密情報非依存、旧checkpoint warm-start、v3 DataLoader、100〜1000件overfit、
統一推論API、同一seed再現性がすべて合格するまで大規模学習を開始しない。
