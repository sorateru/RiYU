# mahjong-ai

四人リーチ麻雀AIを継続的に研究・評価するための基盤です。最終目標は高精度な牌譜解析ですが、現在は **Phase 1（局面・行動・イベントの基礎）** です。未学習モデルの出力を実用上の推奨として扱いません。

このプロジェクトは自作シミュレーター、オフライン牌譜、友人同士の研究を対象にします。オンライン対局サービスの通信解析、不正なリアルタイム補助、自動操作は対象外です。

## 現在実装済み

- 34種の牌と赤牌の型・検証
- 四人麻雀用の独立した `RuleConfig`
- AIが観測できる情報だけを持つ `GameState`
- 学習・シミュレーター専用の `CompleteGameState`（通常状態とは別型）
- `Action`（打牌、リーチ、チー、ポン、カン、ロン、ツモ、パス）
- イベント列による局面更新と再構築
- JSONシリアライズ、スキーマバージョン、イベント連番
- 不正イベントの拒否、失敗時に状態を変更しない原子的更新
- Rust APIを呼ぶ最小のPyO3バインディング

シャンテン、受け入れ、和了・役・点数、完全な合法手生成はPhase 2です。現在のイベント検証は状態遷移と公開情報の整合性を担い、麻雀ルール全体の合法性を保証するものではありません。

## 必要環境

- Rust 1.85.1（`rust-toolchain.toml` で固定）
- Python 3.11以上
- maturin 1.9系

### Rust

Rust未導入の場合は [rustup](https://rustup.rs/) の公式手順で導入します。その後、リポジトリ直下で実行します。

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

### Pythonバインディング

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.9,<2" "pytest>=8,<10"
maturin develop
python -m pytest -q
```

使用例:

```python
from mahjong_ai import replay_events_json

state_json = replay_events_json(observer=0, events_json="[]")
```

空イベント列では未開始状態を返します。局面を開始するには、連番0の `round_start` イベントから入力してください。

## リポジトリ構成

```text
engine/core/       Rust製の正本ドメインモデルと状態遷移
bindings/python/   PyO3バインディング
python/mahjong_ai/ Pythonパッケージ
docs/              アーキテクチャ、状態契約、ロードマップ
```

今後追加する `ai/`、`training/`、`perception/`、`ui/` などは、責務を持つ実装が始まるフェーズで作成します。空ディレクトリや動かないダミーは置きません。

詳しい設計は [docs/architecture.md](docs/architecture.md)、JSON契約は [docs/game_state.md](docs/game_state.md)、開発順序は [docs/roadmap.md](docs/roadmap.md) を参照してください。

## ライセンス

プロジェクト本体のライセンスは未決定です。公開・配布前に決定してください。依存ライブラリの確認結果は [docs/dependencies.md](docs/dependencies.md) に記録しています。
