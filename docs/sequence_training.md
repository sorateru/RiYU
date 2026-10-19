# 時系列・他家推定モデル V2

`sequence-transformer-v2` は、現在局面の49×34平面に加えて直近96件の公開イベントを時系列で読む別系統のモデルです。相手の手牌や牌山は入力しません。牌譜変換時だけ完全情報を教師ラベルとして使い、推論時は公開情報から予測します。

## 教師ラベル

- 実際の選択（打牌、リーチ、チー、ポン、カン、パス、和了）
- 最終順位・最終持ち点
- 局終了時の4人の実得失点
- 各判断時点の4人のシャンテン、他家の聴牌、待ち牌
- 判断後12自摸までに現れた牌種

`observation` と `public_events` は観測者に見える情報だけです。隠れ手牌、実牌山、未来自摸は目的変数にのみ入り、入力列へコピーしません。半荘単位でtrain/validationを分けるため、同じ半荘の別局が両側へ漏れることもありません。

## 変換

既存の許諾確認済みオフラインDBからV2 shardを作ります。

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

```bash
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py \
  --port 8000 \
  --checkpoint artifacts/checkpoints/policy-validated.pt \
  --belief-checkpoint artifacts/checkpoints/sequence-v2-validated.pt
```

V2の聴牌率・待ち分布は隠れ手牌サンプラーを誘導し、予測局収支は相手想定打点を補正します。モデルを読めない、未承認、または探索が失敗した場合は従来の公開情報ヒューリスティックへ戻ります。

## スモーク結果

`artifacts/smoke/sequence-v2-data` は2半荘・28局・1,706判断の配線確認用データです。全28局が受理され、棄却0でした。`sequence-v2-tiny.pt` の1 epoch結果は検証方策一致率30.6%、他家聴牌二値精度75.9%です。データ量もモデル容量も小さく、強さの根拠にはなりません。
