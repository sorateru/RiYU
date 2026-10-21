# 時系列・他家推定モデル V2

`sequence-transformer-v2` は、現在局面の49×34平面に加えて直近96件の公開イベントを時系列で読む別系統のモデルです。相手の手牌や牌山は入力しません。牌譜変換時だけ完全情報を教師ラベルとして使い、推論時は公開情報から予測します。

大規模学習で固定するaction軸、相手3席軸、factual/counterfactual教師、全Head、masked Loss、checkpointの完成契約は [Decision V3 / Mahjong Model V1](decision_v3_model_contract.md) を正とします。

## 教師ラベル

- 実際の選択（打牌、リーチ、チー、ポン、カン、パス、和了）
- 最終順位・最終持ち点
- 局終了時の4人の実得失点
- 各判断時点の4人のシャンテン、他家の聴牌、待ち牌
- 判断後12自摸までに現れた牌種
- 実際の候補打牌に対する相手別放銃、放銃時失点bin、5種の局終了事象

`observation` と `public_events` は観測者に見える情報だけです。隠れ手牌、実牌山、未来自摸は目的変数にのみ入り、入力列へコピーしません。半荘単位でtrain/validation/final-holdoutを固定分割し、final-holdoutは学習・checkpoint選択・較正に使いません。

## 変換

既存の許諾確認済みオフラインDBからV2 shardを作ります。
年別`tar.xz`を内包するユーザー提供ZIPは`tools/import_tenhou_xml_archives.py`で直接v3 shardへ変換できる。変換は二重アーカイブをストリームし、展開後の生XMLをディスクに残さない。

```bash
PYTHONPATH=python .venv/bin/python tools/import_tenhou_sequence_db.py \
  data/high_rank/2020.db \
  --output-dir data/high_rank/2020-sequence-v2 \
  --max-decisions 200000 \
  --shard-decisions 25000 \
  --authorization-reference user-confirmed-permitted-2026-10-19
```

変換は各判断時点で4人分の厳密シャンテン・待ちを計算するため、V1より遅くなります。`conversion-report.json` に受理局、棄却局、例外の種類を随時保存します。出力先は空のディレクトリに限定し、入力SQLiteはread-onlyで開きます。

## 学習

```bash
PYTHONPATH=python .venv/bin/python tools/train_sequence_supervised.py \
  data/high_rank/2020-sequence-v2/part-*.parquet \
  --checkpoint artifacts/checkpoints/tenhou-sequence-v2-unvalidated.pt \
  --epochs 3 --batch-size 48 --device cpu
```

発熱を抑えたい場合はCPU、`--batch-size 16`、短いepochから始めます。`--warm-start` で同じV2 checkpointから継続できます。`--preset tiny` は配線確認専用で、実戦強度を目的にしません。

学習時はtrain splitだけから聴牌・待ち・放銃ラベルの陽性率を集計し、`negative / positive` の不均衡重みを自動設定します。重みは既定で最大30倍に制限され、validationとfinal-holdoutのラベルは推定に使いません。実際に使った値はcheckpointの `effective_positive_weights` と `.training.json` に保存します。固定値との比較には `--no-auto-positive-weights`、上限変更には `--maximum-positive-weight` を使います。

Policyのlabel smoothingは合法手の間だけに確率質量を配り、paddingされた候補には配りません。既定値は0.02で、`--policy-label-smoothing` から変更できます。

最終 `policy_logits` は単一の巨大分類器だけではなく、flat候補scoreへ
`pass/call/win`、`chi/pon/kan`、候補別リーチ確率の対数確率を合成したものです。
したがって学習・checkpoint評価・Web推論で同じ階層Policyを使用し、補助Headだけ学習して
実推論では無視する状態にはなりません。

`--opponent-temporal-latents`（既定で有効）は、Transformer後の公開イベントを相対席ごとにmaskし、学習可能な席queryでattention poolingします。得られた相手別時系列latentは聴牌・待ち・シャンテン・候補別放銃・放銃時打点Headで共有します。誰の公開行動もない席はゼロのイベント文脈と件数0になり、隠れ手牌や他家の非公開自摸牌は入力されません。旧checkpointの設定にはこのフラグがないため従来のglobal Headとして読み込み、warm-start時にだけ新しい相手別層を初期化できます。

`--warm-start` はモデルに加えて、同じHead構成ならOptimizer stateとepoch番号も復元します。Headを新たに追加した場合は共有層だけを移植し、Optimizerは新規作成します。各epochにはaccuracyだけでなく、聴牌precision/recall、待ち牌Top-1、相手別放銃Headのprecision/recall/Brierを記録します。

学習率は既定で50 epoch周期のCosine schedulerを使い、Optimizer・scheduler・乱数・
global stepを一緒に復元します。入力Parquetと同じdirectoryの `manifest.json` はSHA-256を
含めてcheckpointへ保存します。周期と最小学習率比は
`--scheduler-t-max-epochs`、`--minimum-learning-rate-ratio` で変更できます。

## 大規模学習前ゲート

100〜1000 decisionを意図的に記憶させ、全Headに有効教師が存在し、Policyが98%以上、
total lossが初期値の25%以下、各Head lossが改善することを自動確認します。

```bash
PYTHONPATH=python .venv/bin/python tools/check_sequence_v3_overfit.py \
  data/sequence-v3/part-*.parquet --decisions 256 --epochs 100
```

`--no-require-complete-label-coverage` は配線スモーク専用です。本番開始判定では使用しません。
`SequenceBeliefPredictor.evaluate_state(state)` は同じModelOutputから、候補Policy、厳密
シャンテン/受け入れ、相手別放銃率・条件付き失点、局結果分布、順位期待、全belief Headを
Web/Arena共通形式で返します。

出力checkpointは `validated=false` のため、そのまま解析サービスには接続できません。独立対局集合で方策一致率、聴牌Brier、待ち牌precision/recall、局収支誤差、Arena成績を評価して承認した後にだけ、次の形で探索の事前分布として使用します。

放銃率は方策確率と独立に較正します。係数はvalidationのみで学習し、相手別Brier・Log Loss・ECEを記録します。

```bash
PYTHONPATH=python .venv/bin/python tools/calibrate_sequence_risk.py \
  artifacts/checkpoints/sequence-unvalidated.pt data/sequence/part-*.parquet \
  --output artifacts/checkpoints/sequence-risk-calibrated.pt

# 通常はvalidation。final-holdoutを開くには2つの明示フラグが必要。
PYTHONPATH=python .venv/bin/python tools/evaluate_sequence_checkpoint.py \
  artifacts/checkpoints/sequence-risk-calibrated.pt data/sequence/part-*.parquet
```

```bash
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py \
  --port 8000 \
  --checkpoint artifacts/checkpoints/policy-validated.pt \
  --belief-checkpoint artifacts/checkpoints/sequence-v2-validated.pt
```

V2の聴牌率・待ち分布は隠れ手牌サンプラーを誘導し、予測局収支は相手想定打点を補正します。モデルを読めない、未承認、または探索が失敗した場合は従来の公開情報ヒューリスティックへ戻ります。

## スモーク結果

`artifacts/smoke/sequence-v2-data` は2半荘・28局・1,706判断の配線確認用データです。全28局が受理され、棄却0でした。`sequence-v2-tiny.pt` の1 epoch結果は検証方策一致率30.6%、他家聴牌二値精度75.9%です。データ量もモデル容量も小さく、強さの根拠にはなりません。

## 53k転移学習結果

許諾確認済みの128半荘からV2を53,459判断生成し、47,407判断をtrain、半荘単位で分離した6,052判断をvalidationに使用しました。局面Encoder・Residual backbone・動的Action・方策/価値Headは、既存V1の20万判断checkpointから移植しています。

- 方策候補 `tenhou-sequence-v2-transfer-53k-best-unvalidated.pt`: 方策一致率67.61%、聴牌Brier 0.1045、局収支MAE 2,844点。
- 条件付き聴牌候補 `tenhou-sequence-v2-transfer-53k-conditional-waits-unvalidated.pt`: 方策一致率66.75%、聴牌Brier 0.1006、聴牌recall 17.1%、局収支MAE 2,759点。
- 待ち牌Top-1命中率は約6.6%で、探索priorとして承認できる水準には未到達。
- 将来自摸Top-5 recallは約15.4%で、ランダム基準（5/34≒14.7%）に近い。

いずれも同じvalidationでモデル選択・測定しているため、`validated=false`のままです。独立holdoutと半荘Arenaを通すまで実運用へ昇格しません。
