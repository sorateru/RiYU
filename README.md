# mahjong-ai "RiYU"


四人リーチ麻雀の厳密ルール、オフライン牌譜、深層学習、自己対戦、保存画像認識を一つの観測状態契約で扱う研究基盤です。Phase 1〜9の実行可能な基盤を実装していますが、強い学習済みモデルはまだ同梱しておらず、NAGA級性能を主張しません。

対象は保存牌譜、保存スクリーンショット、自作Simulator、明示的に許諾された入力元での友人間研究です。雀魂連携は牌譜閲覧画面だけで動く任意導入のChrome拡張に加え、利用者自身の友人戦画面を読み取るローカルVision Adapterを用意しています。通信傍受、非公開APIへの追加リクエスト、ログイン自動化、入力操作・自動打牌は実装しません。

## 主な実装

- Rust: `GameState`、イベント再生、動的合法手、シャンテン・受け入れ、役・符・翻・点数、フリテン、搶槓、流局、局進行
- Rust Simulator: 136牌生成、4人別観測投影、ツモ・打牌・鳴き・和了・流局
- Data: 観測者別イベントログ、厳密再生検証、教師データParquet
- AI: 49×34 Encoder、動的Action、Residual Policy/Value、公開捨て牌順Transformer、他家聴牌・待ち・局収支Head
- Training: Dataset/DataLoader、学習・validation、checkpoint、MLflow、Self Play、Replay Buffer、PPO部品
- Oracle/Opponent: 完全情報Teacherの型分離、蒸留、テンパイ・待ち・打点Head、Brier/Log Loss
- Perception: 保存PNG/許諾済みmacOS画面向けOpenCVテンプレート分類、scene guard、window capture、レイアウト/profile検証、確信度・不明値
- API/UI: FastAPI + WebSocket + React/Tauri。オフライン局面または許諾済みの正規化ストリームを解析し、厳密値・モデル推定・N/Aを区別

詳細は [architecture](docs/architecture.md)、[roadmap](docs/roadmap.md)、[rule matrix](docs/rules_phase2.md)、[feature encoder](docs/feature_encoder.md)、[data format](docs/data_format.md)、[safety/distribution](docs/safety_distribution.md) を参照してください。

## セットアップ

必要環境はコア用Rust 1.85.1、Python 3.11以上、Node.jsです。Tauriシェルだけは `apps/reviewer/src-tauri/rust-toolchain.toml` でRust 1.90.0を固定します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all,dev,reference-tests]'
maturin develop

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
python -m pytest -q
python tools/crosscheck_shanten.py
```

新RiYU Web UI（複数局の牌譜解析）:

```bash
cd apps/web
npm install
npm run build
cd ../..
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py \
  --port 8000 --web-dist apps/web/dist
```

`http://127.0.0.1:8000/` を開き、許諾済み入力から生成した `mahjong-ai-event-log/v1` JSONを読み込む。複数局は複数ファイル、JSON配列、または `{"logs": [...]}` でまとめて解析できる。雀魂はユーザー保有の `mahjong-soul-decoded-replay/v1` を直接読み込むか、取得元・変換器・利用権を記録する `mahjong-ai-replay-bundle/v1` に変換して読み込める。UIとAPIは同一のloopbackサーバーから配信される。開発中は別ターミナルで `cd apps/web && npm run dev` を実行すると、`/api` が `127.0.0.1:8000` へproxyされる。

雀魂共有URLをそのまま貼り付ける場合は、利用者が保有するdecode済みJSONを一度ローカル登録する。以後は同じ `paipu` IDの公式共有URLだけで解析できる。URL本文やpaipu IDはキャッシュのファイル名・解析レスポンスへ保存しない。

```bash
PYTHONPATH=python .venv/bin/python tools/register_mahjong_soul_replay.py \
  'https://game.mahjongsoul.com/?paipu=...' decoded-replay.json
```

NAGAと同じ操作感にする場合はChrome拡張をビルドして読み込む。ブラウザ版雀魂で牌譜を開き、拡張の「RiYUで解析」を押すと、表示中の牌譜だけがloopbackのRiYUへ送られ解析画面が開く。詳細は [browser extension](docs/browser_extension.md) を参照。

メインWeb UIの「リアルタイム」は、許諾済みの雀魂画面を読み取るVision Adapterだけを自動検出する。
牌譜解析と同じ `apps/web` 内で切り替えられ、通信傍受や非公開API入力は表示対象にしない。
旧ローカルAI対局と単一GameState解析のReact/Tauri UIは `apps/reviewer` に残している。

画面キャプチャ、ローカルAPI、メインWeb UIをまとめて起動する場合:

```bash
PYTHONPATH=python .venv/bin/python tools/run_majsoul_webapp.py \
  local-data/majsoul-vision.json
```

ローカル解析サービス（HTTP `/analyze` とWebSocket `/ws/lab`）:

```bash
python tools/serve_analysis.py --port 8787
```

WebSocketは正規化済みの観測 `GameState`、または雀魂adapter境界の公開観測イベントを受け付けます。外部サービスの通信傍受や非公開API処理は含みません。雀魂用には許可記録付きsource profile、共通`GameEvent`変換、継続State同期、意思決定時だけの推論、read-only viewer、保存イベントリプレイを用意しています。[realtime protocol](docs/realtime_protocol.md) と [雀魂event adapter](docs/majsoul_event_adapter.md) を参照してください。

人間1人対AI 3人のローカル東南戦は、対戦用checkpointを指定して起動します。局面解析とは別の `/local-play` APIで、対局中に推奨手は表示しません。

```bash
python tools/serve_analysis.py --port 8000 \
  --play-checkpoint artifacts/checkpoints/tenhou-phoenix-2020-200k.pt
cd apps/reviewer
pnpm dev
```

AIの強さを比較するには、座席を入れ替えるArenaを実行し、Reviewerの「AI対戦結果」で出力JSONを開きます。以下は模倣学習方策と牌効率ベースラインの比較です。1シードにつき4半荘を実行します。

```bash
PYTHONPATH=python .venv/bin/python tools/evaluate_arena.py \
  artifacts/checkpoints/tenhou-phoenix-2020-200k.pt \
  artifacts/checkpoints/tenhou-phoenix-2020-200k.pt \
  --policy-a imitation --policy-b tile_efficiency \
  --seed-blocks 16 --workers 2 --output artifacts/reports/arena-vs-efficiency.json
```

結果は指定した相手と方策に対する相対評価です。ローカルAI対局に近い方策を評価する場合は、A側を `hybrid_search` に変更できます。

教師データと学習:

```bash
python tools/build_dataset.py logs/*.json --output decisions.parquet
python tools/train_supervised.py decisions.parquet --checkpoint baseline.pt --epochs 3
```

時系列V2（4視点同期、カン教師、局収支、他家手牌・将来自摸を目的変数だけに使用）の変換と学習は [sequence training](docs/sequence_training.md) を参照してください。

生成されたcheckpointは既定で未承認です。独立validation、calibration、Arena評価、モデルカードが完了するまでUIは推奨として表示しません。

モデル自己対戦とPPO更新:

```bash
python tools/train_selfplay_ppo.py \
  --checkpoint artifacts/checkpoints/ppo-baseline-unvalidated.pt \
  --iterations 10 --rounds-per-iteration 8 --update-epochs 4

# 中断したcheckpointから継続
python tools/train_selfplay_ppo.py \
  --checkpoint artifacts/checkpoints/ppo-baseline-unvalidated.pt \
  --resume --iterations 10 --rounds-per-iteration 8
```

各反復で現在のPolicyから合法手をサンプリングし、旧Policy確率・旧Value・局終了時点差を使ってclipped PPO、Value loss、Entropy正則化を更新します。optimizerと反復番号もcheckpointに保存するため再開できます。詳細は [self-play training](docs/selfplay_training.md) を参照してください。

許諾済み人間牌譜からのカリキュラム学習:

```bash
python tools/create_human_manifest.py data/human-decisions.parquet \
  --output data/human-manifest.json \
  --dataset-id consented-human-v1 \
  --source private-consented-games \
  --rights-basis explicit_permission \
  --authorization-reference permission-record-id \
  --confirm-ml-rights --confirm-anonymized

python tools/train_curriculum.py \
  --manifest data/human-manifest.json \
  --checkpoint artifacts/checkpoints/human-ppo-unvalidated.pt \
  --supervised-epochs 5 --ppo-iterations 10 \
  --rounds-per-iteration 32 --ppo-workers 4
```

manifestはML利用権、匿名化、ファイルSHA-256を必須にします。天鳳・雀魂由来データは `explicit_permission` 以外では読み込みを拒否します。詳細は [human data](docs/human_data.md) を参照してください。

自己対戦と保存画像:

```bash
python tools/run_selfplay.py \
  --rounds 1000 --workers 0 --shard-rounds 64 \
  --output replay-shards
python tools/analyze_image.py saved.png \
  --layout docs/perception_layout.example.json \
  --templates /path/to/licensed-tile-templates
```

`--workers 0` は利用可能CPU数を自動選択します。出力にはZstandard圧縮Parquet shard、seed・局範囲・SHA-256を記録した `manifest.json` が含まれ、生成後に整合性を自動検証します。

画像テンプレートやサービス固有素材は権利上の混同を避けるため同梱していません。

macOS上の許諾済み雀魂画面を読み取り専用で追跡するAdapterも追加しています。
連続frame確認とRust `GameState`検証を通った差分だけを既存のrealtime viewerへ送ります。
セットアップ、対応event、現時点の制限は
[雀魂画面認識Adapter](docs/majsoul_vision_adapter.md)を参照してください。

## リポジトリ構成

```text
engine/core/                  Rustルールと四人Simulator
bindings/python/              PyO3境界
python/mahjong_ai/ai/         Encoder、Policy/Value、Opponent
python/mahjong_ai/data/       共通牌譜、Parquet
python/mahjong_ai/training/   教師あり、自己対戦、Oracle
python/mahjong_ai/perception/ 保存画像認識
python/mahjong_ai/apps/       Reviewer / streaming API
apps/web/                     牌譜解析・画面キャプチャ型リアルタイム統合UI
apps/reviewer/                旧React/TauriローカルUI
evaluation/                   評価指標（Python package内）
tools/                        CLIと参照テスト
docs/                         契約・設計・安全境界
```

## 配布

プロジェクト本体のライセンスと著作権者表記は所有者の決定待ちです。この状態では第三者配布を開始しないでください。外部サービス由来の牌譜・画像・商標・ロゴは同梱せず、利用条件を個別に確認してください。依存ライセンスは [docs/dependencies.md](docs/dependencies.md) に記録しています。
