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

## 出所付きReplay Bundle v1

`mahjong-ai-replay-bundle/v1` は、半荘を構成する複数のEventLogと、取得・変換・利用権の来歴を一つにまとめる。雀魂などサービス固有のオフラインexportを別アダプターでEventLogへ変換した後、RiYUへ渡す正式な境界である。

機械可読な定義は [`schemas/replay-bundle-v1.schema.json`](../schemas/replay-bundle-v1.schema.json) に置く。v1のトップレベル、source、converterに未知フィールドがある場合は入力ミスとして拒否する。

```json
{
  "format": "mahjong-ai-replay-bundle/v1",
  "source": {
    "platform": "mahjong_soul",
    "acquisition": "user_export",
    "authorization": "user_owned",
    "converter": {"name": "reviewed-local-adapter", "version": "1.0"},
    "share_url": "https://game.mahjongsoul.com/?paipu=..."
  },
  "logs": [
    {"format": "mahjong-ai-event-log/v1"}
  ]
}
```

対応値:

- `platform`: `mahjong_soul`, `tenhou`, `local`, `other`
- `acquisition`: `user_export`, `manual_conversion`, `authorized_adapter`, `generated`
- `authorization`: `user_owned`, `explicit_permission`, `open_license`, `generated`

外部サービスに `generated` authorizationは指定できない。`share_url` はHTTPSかつ認証情報を含まず、宣言したplatformの公式ドメインに一致するURLだけを許可する。解析レスポンスではURL本文を返さず `url-sha256:` 参照へ置換する。bundle全体に対してEventLogの連番・秘密牌漏洩検査を再実行する。

既存のEventLogからbundleを作る例:

```bash
PYTHONPATH=python .venv/bin/python tools/create_replay_bundle.py \
  converted-rounds/*.json \
  --output replay-bundle.json \
  --platform mahjong_soul \
  --acquisition user_export \
  --authorization user_owned \
  --converter-name reviewed-local-adapter \
  --converter-version 1.0 \
  --share-url 'https://game.mahjongsoul.com/?paipu=...'
```

出力先が既に存在する場合は上書きしない。Web APIは最大64局、合計256判断、16 MiBまで受け付ける。

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

`mahjong-soul-decoded-replay/v1` は、利用者が適法に保有しローカルでdecode済みの4人打ち牌譜を受け渡す中間形式である。`records`（または `actions`）には `new_round`、`draw`、`discard`、`call`、`kan`、`dora_reveal`、`win`、`draw_game` を記録でき、`.lq.RecordNewRound` など対応する `Record*` 名も受理する。牌は `1m`〜`9s`、`0m`/`5mr`（赤五）、`1z`〜`7z` で表す。

全視点の秘密情報を正しく分離するため、各局の開始レコードには4人全員の13枚配牌が必要である。和了・流局には4人分の `score_deltas`（または `delta_scores`）が必要である。欠落した配牌、鳴き元を一意に決められない副露、理由不明の途中流局、点数差分のない終局は推測せず拒否する。変換例:

```bash
PYTHONPATH=python .venv/bin/python tools/convert_mahjong_soul.py \
  decoded-replay.json \
  --output replay-bundle.json \
  --authorization user_owned \
  --share-url 'https://game.mahjongsoul.com/?paipu=...'
```

任意導入のChrome拡張は、NAGA向け支援拡張と同じく、牌譜閲覧画面でブラウザが既に受信した `fetchGameRecord` 応答だけを識別し、ローカルでprotobuf decodeする。雀魂サーバーへの追加リクエスト、ログイン自動化、対局中取得、入力操作は行わない。decode後はプレイヤー名・アカウント情報を送らず、局イベントと最終点だけをloopbackのRiYUへ送る。実装の出所とライセンスは `apps/riyu-extension/NOTICE` に記録する。

## 雀魂共有URLのローカル解決

`tools/register_mahjong_soul_replay.py` は、公式共有URLと検証済みdecode済みJSONを、ユーザー管理のローカルキャッシュで関連付ける。ファイル名は `paipu` IDを直接使わずSHA-256で生成する。Web/APIへ同じURLを入力すると、HTTPS・公式ドメイン・認証情報なし・単一 `paipu` IDを再検証してからキャッシュを解決し、通常のReplay Bundle検証へ渡す。

この仕組みは外部ネットワークへアクセスしない。将来公式APIまたは明示的に認可されたadapterが提供された場合も、URL入力後のEventLog変換・秘密牌検査・解析部分を変更せずresolverだけを交換できる。
