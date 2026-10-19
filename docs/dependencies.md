# 依存ライブラリ方針

確認日: 2026-10-18

RustコアはAI/UI依存を持たず、Python・画像・UI依存はextrasとアプリ側へ分離する。以下の版は2026-10-18にローカルmetadataとlockfileで確認した実動作環境であり、`pyproject.toml` は互換範囲を指定する。

| 依存 | 用途 | 採用範囲 | ライセンス/保守判断 |
|---|---|---|---|
| serde / serde_json 1.x | 型付きJSON | Rustコア | Rustで広く利用されるシリアライズ基盤。Serde公式文書を正本とする。MIT OR Apache-2.0 |
| thiserror 2.x | 型付きエラー | Rustコア | deriveによるエラー実装だけに限定。MIT OR Apache-2.0 |
| PyO3 0.29.3 | Rust/Python FFI | bindingsのみ | Python 3.13を含むサポート対象Pythonに対応。ABI3をPython 3.11基準で使用。MIT OR Apache-2.0 |
| maturin 1.9以上2未満 | Python wheel/dev build | ビルド時のみ | PyO3公式推奨のビルド手段。Python 3.8+、macOS/Linux/Windowsをサポート。MIT OR Apache-2.0 |
| MahjongRepository/mahjong 1.4.0 | シャンテン差分テスト | 任意の開発検証のみ | MIT。実行時依存にはせず、固定seedランダム牌姿の期待値生成だけに使用 |
| NumPy 2.5.4 | 特徴量・画像配列 | Python共通 | BSD-3-Clauseほか（wheel metadataの複合式を確認） |
| PyTorch 2.14.1 | Policy/Value/Opponent/Oracle | `learning` extra | Apache-2.0ほか。CPU/MPS/CUDAは実行環境ごとに固定する |
| PyArrow 23.0.1 | Parquet | `learning` extra | Apache-2.0 |
| MLflow 3.17.1 | 実験管理 | `learning` extra | Apache-2.0 |
| OpenCV headless 4.14.0.94 | 保存画像認識 | `vision` extra | Apache-2.0 |
| FastAPI 0.144.0 / Uvicorn 0.54.0 | ローカルReviewer API | `service` extra | MIT / BSD-3-Clause |
| React 19.3.1 / Vite 7.3.7 | Reviewer UI | Node app | MIT / MIT |
| Tauri 2.12.x | Desktop shell | Node/Rust app | MIT OR Apache-2.0 |

Tauri 2.12.xのMSRVはコアの1.85.1より新しいため、デスクトップシェルだけRust 1.90.0を使用する。コアWorkspaceの固定版は変更しない。

参照:

- [Serde documentation](https://serde.rs/)
- [PyO3 user guide](https://pyo3.rs/)
- [PyO3 supported Python versions](https://pyo3.rs/main/building-and-distribution/multiple-python-versions.html)
- [maturin user guide](https://www.maturin.rs/)
- [maturin platform support](https://www.maturin.rs/platform_support)
- [MahjongRepository/mahjong](https://github.com/MahjongRepository/mahjong)

バージョン更新はCIで `cargo test`、`clippy`、Pythonスモークテストを通したうえで行う。プロジェクト本体のライセンスは所有者が決めるまで未定とし、依存のライセンスと混同しない。
