# ローカルAI対局

`/local-play` は、自作Rust Simulator上で人間1人とAI 3人が東南戦を行うローカル対局機能です。`/analyze` と `/ws/lab` の解析機能とは独立しています。

## 境界

- 入力はローカルSimulatorのボタン操作だけです。
- 外部サービスの画面取得、通信、アカウント、自動打牌に接続しません。
- 対局レスポンスは人間席の観測状態と合法手だけを含み、他家の手牌と壁順は含みません。
- 対局中はAIの推奨手、候補確率、局面解析を表示しません。
- 対局セッションはプロセス内メモリだけに保存され、サーバー終了時に破棄されます。

## 起動

```bash
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py --port 8000 \
  --play-checkpoint artifacts/checkpoints/tenhou-phoenix-2020-response-balanced-ev.pt

cd apps/reviewer
npm run dev
```

ブラウザーまたはTauri画面の「AI対局」から開始席と任意のseedを選びます。「局面解析」へ切り替えても対局セッションは保持されます。

## API

- `POST /local-play/sessions`: 東南戦を開始する。
- `GET /local-play/sessions/{session_id}`: 現在の人間席観測を取得する。
- `POST /local-play/sessions/{session_id}/actions`: 合法手を1つ適用する。
- `DELETE /local-play/sessions/{session_id}`: 対局セッションを破棄する。

AIの手番は、各合法手の一手後に対する最終持ち点予測、厳密計算のシャンテン数・受け入れ枚数、鳳凰卓教師Policyの事前補正を合成します。さらに公開情報だけから相手の手牌と残り山を12通り生成し、上位4候補を3巡先までロールアウトして、和了・放銃・他家和了の推定収支を加えます。

探索器へ相手の実手牌や牌山順は渡しません。河、副露、リーチ、巡目、見えている牌から確率的に生成した候補世界だけを使用します。表示される探索期待点と確率は未較正の推定値であり、厳密な点数期待値ではありません。

現在の天鳳変換器にはカン教師がないため、カンは合法手として解析しますが、対局AIでは保守的な罰則を与えています。
