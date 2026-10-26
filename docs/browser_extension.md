# RiYU 雀魂牌譜インポート拡張

## 概要

`apps/riyu-extension` は、NAGA向け雀魂牌譜検討サポーターと同じ構成のChrome Manifest V3拡張です。ブラウザ版雀魂の牌譜閲覧画面で既に受信された `fetchGameRecord` 応答だけを保持し、ユーザーが「RiYUで解析」を押した時にローカルdecodeしてRiYUへ送ります。

- 雀魂サーバーへ追加リクエストを送らない
- プレイヤー名、account ID、装飾情報をRiYUへ送らない
- 送信先は `http://127.0.0.1` または `http://localhost` のみ
- 対局中の助言、画面操作、自動打牌を行わない
- APIは解析結果を最大8件、5分だけメモリ保持し、牌譜本文をディスク保存しない

牌譜取得・decodeの一部はApache-2.0の `MahjongSoul-review-supporter` を改変して利用しています。帰属とライセンスは [`apps/riyu-extension/NOTICE`](../apps/riyu-extension/NOTICE) にあります。

## ビルド

```bash
cd apps/riyu-extension
npm install
npm run build
```

生成先は `apps/riyu-extension/.output/chrome-mv3` です。

## Chromeへの読み込み

1. Chromeで `chrome://extensions` を開く
2. 「デベロッパー モード」を有効にする
3. 「パッケージ化されていない拡張機能を読み込む」を押す
4. `apps/riyu-extension/.output/chrome-mv3` を選択する
5. 拡張をツールバーへ固定する

拡張を読み込んだ後、既に開いていた雀魂タブは一度再読み込みしてください。

## 使用方法

RiYUを先に起動します。

```bash
cd apps/web
npm run build
cd ../..
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py \
  --port 8787 \
  --web-dist apps/web/dist
```

続いて:

1. Chrome版雀魂で解析したい牌譜を開く
2. 牌譜が表示された状態でRiYU拡張を開く
3. 観戦席を選ぶ
4. 「RiYUで解析」を押す
5. 新しいタブでRiYU解析画面が開く

取得できない場合は、牌譜画面を開いたままページを再読み込みし、牌譜をもう一度開いてください。デスクトップアプリ版・モバイルアプリ版にはChrome拡張を読み込めません。

現在は牌譜本体がWebSocket応答内に含まれる通常形式を対象にしています。雀魂側が巨大牌譜を `data_url` だけで返した場合は追加取得を行わず、未対応エラーを表示します。
