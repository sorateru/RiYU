# 人間牌譜によるカリキュラム学習

## 方針

人間牌譜は、自己対戦初期の無作為な癖を減らす教師あり事前学習に使い、そのcheckpointからPPO自己対戦へ移る。公開閲覧できる牌譜であることと、機械学習へ利用できることは同義ではない。

このリポジトリは牌譜サイトを巡回・一括ダウンロードしない。利用者自身が権利を持つ記録、機械学習利用の明示許可を得た記録、または明確なオープンライセンスのデータだけを、観測者別イベント形式へ変換して受け付ける。

## 天鳳

天鳳公式マニュアルの「牌譜を使用する場合の注意事項」では、一般の麻雀への応用を目的とする場合は `support@c-egg.com` への問い合わせが必要とされ、天鳳と競合する製品への開発・応用には使用できないと記載されている。

そのため天鳳由来データは、用途について得た明示的な許可を `authorization_reference` で識別できる場合だけ読み込む。公開牌譜URL、GitHub上の派生dataset、第三者が付けたコードライセンスだけでは、元牌譜の利用許可を確認したことにはしない。

## Provenance manifest

`mahjong-ai-human-dataset/v1` manifestには以下が必要になる。

- dataset IDと取得元
- `user_owned`、`explicit_permission`、`open_license` のいずれか
- 許可文書・同意記録・ライセンスを識別するreference
- `machine_learning_allowed=true`
- `contains_personal_identifiers=false`
- 各`decision/v1` ParquetのSHA-256

天鳳・雀魂を取得元に含む場合は `explicit_permission` を必須とする。これは権利を自動判定するものではなく、誤って由来不明データを学習へ混入させないための実装上の門である。

## 学習順序

```bash
python tools/build_dataset.py normalized-logs/*.json \
  --output data/human-decisions.parquet

python tools/create_human_manifest.py data/human-decisions.parquet \
  --output data/human-manifest.json \
  --dataset-id human-v1 \
  --source private-consented-games \
  --rights-basis explicit_permission \
  --authorization-reference permission-record-id \
  --confirm-ml-rights --confirm-anonymized

python tools/train_curriculum.py \
  --manifest data/human-manifest.json \
  --checkpoint artifacts/checkpoints/human-ppo-unvalidated.pt
```

`train_curriculum.py` は人間の行動を模倣する教師あり学習を行い、その重みだけをPPOへwarm startする。PPO optimizerは新規作成し、人間dataset fingerprintとwarm-start checkpoint hashをlineageとして最終checkpointへ残す。
