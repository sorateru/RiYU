# 依存ライブラリ方針

確認日: 2026-10-18

Phase 1は依存面積を小さく保ち、Rustコアでは `serde`、`serde_json`、`thiserror`、Python境界では `PyO3` とビルド用 `maturin` だけを使用する。PyTorch、NumPy、OpenCV、MLflowは実際に利用するフェーズまで導入しない。

| 依存 | 用途 | 採用範囲 | ライセンス/保守判断 |
|---|---|---|---|
| serde / serde_json 1.x | 型付きJSON | Rustコア | Rustで広く利用されるシリアライズ基盤。Serde公式文書を正本とする。MIT OR Apache-2.0 |
| thiserror 2.x | 型付きエラー | Rustコア | deriveによるエラー実装だけに限定。MIT OR Apache-2.0 |
| PyO3 0.29.3 | Rust/Python FFI | bindingsのみ | Python 3.13を含むサポート対象Pythonに対応。ABI3をPython 3.11基準で使用。MIT OR Apache-2.0 |
| maturin 1.9以上2未満 | Python wheel/dev build | ビルド時のみ | PyO3公式推奨のビルド手段。Python 3.8+、macOS/Linux/Windowsをサポート。MIT OR Apache-2.0 |

参照:

- [Serde documentation](https://serde.rs/)
- [PyO3 user guide](https://pyo3.rs/)
- [PyO3 supported Python versions](https://pyo3.rs/main/building-and-distribution/multiple-python-versions.html)
- [maturin user guide](https://www.maturin.rs/)
- [maturin platform support](https://www.maturin.rs/platform_support)

バージョン更新はCIで `cargo test`、`clippy`、Pythonスモークテストを通したうえで行う。プロジェクト本体のライセンスは所有者が決めるまで未定とし、依存のライセンスと混同しない。

