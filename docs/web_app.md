# RiYU WebとFastAPIの統合

`apps/web` は正規化済み牌譜を局・巡目単位で見るReact UIです。EventLog、Replay Bundle、ローカルdecode済み雀魂JSON、ローカル登録済みの雀魂共有URLと観測席0〜3を入力できます。非公開APIには接続しません。

## 一体起動

```bash
cd apps/web
npm install
npm run build
cd ../..

PYTHONPATH=python .venv/bin/python tools/serve_analysis.py \
  --port 8000 \
  --web-dist apps/web/dist
```

`http://127.0.0.1:8000/` がUIです。ビルド後のUIをFastAPIが配信するため、一体起動時にCORS設定は不要です。入力中のJSONはブラウザ内に下書き保存されますが、解析した牌譜をサーバーのディスクへ保存しません。

## 開発起動

ターミナル1:

```bash
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py --port 8000
```

ターミナル2:

```bash
cd apps/web
npm run dev
```

Viteは `/api` を `http://127.0.0.1:8000` へproxyします。直接別オリジンのAPIを使う場合だけ `VITE_API_BASE_URL` を設定します。

## 牌譜入力

ファイル選択とドラッグ＆ドロップは複数ファイルに対応します。テキスト欄では次の6形式を受け付けます。

- 1局のEventLogオブジェクト
- EventLogオブジェクトの配列
- `{ "logs": [...] }` 形式のbundle
- 出所情報を含む `mahjong-ai-replay-bundle/v1`
- 4人分の配牌を含む `mahjong-soul-decoded-replay/v1`
- ローカル登録済みの雀魂公式HTTPS共有URL

同時解析は最大64局、合計256打牌判断、16 MiBです。局一覧と巡目移動は解析画面から操作できます。Replay Bundleを使用すると、解析画面に雀魂などの取得元が表示されます。共有URL本文は解析結果へ含めず、短いSHA-256参照だけを返します。

ブラウザの永続ストレージへ牌譜本文は保存しません。共有URLは公式ドメイン、HTTPS、単一の `paipu` IDを検査し、`data/replay-url-cache` のローカル登録データだけに解決します。URL本文は保存せず、解析レスポンスでは短いSHA-256参照だけを返します。

```bash
PYTHONPATH=python .venv/bin/python tools/register_mahjong_soul_replay.py \
  'https://game.mahjongsoul.com/?paipu=...' decoded-replay.json
```

別のキャッシュ場所を使う場合は、登録CLIとサーバーの両方へ同じ `--cache-dir` / `--replay-cache-dir` を指定します。

NAGA型のワンクリック連携には `apps/riyu-extension` のChrome拡張を使用します。拡張は表示中の雀魂牌譜をdecodeし、`POST /api/extension-import` へ匿名化したイベントを送信します。APIは最大8件、5分間だけ解析結果をメモリ保持し、一時token付きのRiYU画面を開きます。ディスクへ牌譜を保存しません。導入手順は [browser_extension.md](browser_extension.md) を参照してください。

## API契約

```json
{
  "logs": [
    { "format": "mahjong-ai-event-log/v1" }
  ],
  "observer": 0
}
```

互換性のため単一局の `{"log": ...}` も受け付けます。バックエンドは4観測streamの連番と非公開ツモ牌漏洩を検査し、指定席の打牌判断だけを再生します。承認済みモデルがない場合は、シャンテンと受け入れによる厳密牌効率を表示し、放銃率と期待値は `N/A` にします。承認済みモデルを起動引数で渡した場合だけ、モデル推定値に切り替わります。

主要APIは `GET /api/health` と `POST /api/replay-analysis` です。後者は `log`、`logs`、`bundle`、`decoded_mahjong_soul`、`url` のいずれか一つだけを受け付けます。healthレスポンスの `replay_import` から対応形式と上限を確認できます。
