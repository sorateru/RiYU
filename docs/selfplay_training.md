# 自己対戦PPO学習

## 実装済みの閉ループ

`tools/train_selfplay_ppo.py` は一つの反復で次を実行する。

1. 現在の `MahjongNetwork` を4席共通のPolicyとして固定する。
2. 動的な合法手ごとのlogitをtemperature付き分布へ変換し、決定論的seedで行動をサンプリングする。
3. 観測状態、合法手、選択index、旧log probability、旧score valueを保存する。
4. 局終了時の点差を10,000点単位のMonte Carlo returnとして各意思決定へ割り当てる。
5. `return - old_value` を正規化したadvantageとし、clipped PPO、score-value MSE、entropy正則化で更新する。
6. モデル、optimizer、反復番号、設定、最終metricsをcheckpointへ保存する。

再開時はモデルだけでなくoptimizer stateと反復番号も復元する。収集時と更新時で同じtemperatureを使用し、PPOの旧・新確率比が同じ分布定義になるようにする。

## 実行

```bash
python tools/train_selfplay_ppo.py \
  --checkpoint artifacts/checkpoints/ppo-baseline-unvalidated.pt \
  --iterations 10 \
  --rounds-per-iteration 8 \
  --workers 4 \
  --update-epochs 4 \
  --batch-size 128
```

機能確認だけなら `--small-model --iterations 1 --rounds-per-iteration 1 --update-epochs 1` を使用できる。

複数workerでは反復開始時のモデルsnapshotを固定し、局ごとに独立したseedを割り当てる。通常はspawnプロセス並列を使い、OSの同期primitiveが利用できない制限環境ではthreadへフォールバックする。全局の収集が完了してから一度だけモデルを更新するため、収集中に重みが変わることはない。

## checkpointの扱い

PPOを完走したcheckpointは `trained=true` だが、既定では `validated=false`、`policy_calibrated=false` である。そのためReviewerやリアルタイムAPIは、Arena評価とcalibrationに合格するまで推奨モデルとして表示しない。

この段階の報酬は局単位の最終点差であり、疎である。実用強度には多数局、教師あり事前学習、固定相手Arena、平均順位の信頼区間、放銃率・和了率・calibrationの比較が必要になる。短いsmoke runのloss低下だけで強さを主張しない。
