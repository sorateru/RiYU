# 時系列・他家推定モデル V2

`sequence-transformer-v2` は、現在局面の49×34平面に加えて直近96件の公開イベントを時系列で読む別系統のモデルです。相手の手牌や牌山は入力しません。牌譜変換時だけ完全情報を教師ラベルとして使い、推論時は公開情報から予測します。

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
