# オフライン牌譜・データセット契約

## 共通イベントログ v1

`mahjong-ai-event-log/v1` は、同じ対局を4人それぞれの観測イベント列として保存する。

```json
{
  "format": "mahjong-ai-event-log/v1",
  "log_id": "stable-id",
  "rules": null,
  "observations": {"0": [], "1": [], "2": [], "3": []},
  "result": {"final_scores": [25000, 25000, 25000, 25000]},
  "provenance": {"source": "offline_export", "converter_version": "adapter-v1"}
}
```

各列はsequence 0から連続する。自分のDRAWは牌必須、他家DRAWは必ず `null` でなければならず、違反ログは読み込み時に拒否する。元形式、変換器版、ルール、状態schema版をParquetへ引き継ぐ。

`tools/build_dataset.py` は、各意思決定直前のobservation、動的legal_actions、expert_action、最終順位・点数を検証し、Zstandard圧縮Parquetへ出力する。専門家行動がRustの合法手集合にないログは黙って除外せず失敗させる。

## 自己対戦データセット v1

`tools/run_selfplay.py` は局ごとに独立したseedを割り当て、`mahjong-ai-selfplay/v1` ディレクトリを生成する。実行順序やworker数が変わっても、同じbase seedと設定から各局の軌跡を再現できる。

```text
replay-shards/
  manifest.json
  part-00000.parquet
  part-00001.parquet
```

各transitionは `observation`、`legal_actions`、`action_index`、`player`、`reward`、`done` に加え、`round_index`、`round_seed`、`decision_index`、`old_log_probability`、`old_value` を持つ。ランダム方策でも一様分布の旧log probabilityを記録し、モデル方策では収集時のPolicy/Valueを記録する。manifestは生成器版、base seed、局数、transition数、実行backend、shardごとの局範囲・行数・SHA-256を記録する。

通常環境ではspawn方式のプロセス並列を使用する。OSがプロセス同期primitiveを提供しない制限環境ではthreadへフォールバックし、その事実を `execution_backend` に記録する。生成完了時に全hash、Parquet行数、連続した局範囲、合計値を再検証する。既存datasetディレクトリへの上書きは拒否する。

## 雀魂を含む外部サービス

このリポジトリは通信傍受、非公開API、暗号・protobuf解析、ログイン自動化、対局中取得を実装しない。利用者が適法に保有する保存済みデータを、別途レビュー済みのアダプターで上記契約へ変換した後だけ受け付ける。サービス固有コード、画像、商標、牌画像は配布物へ同梱しない。
