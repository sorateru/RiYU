# 雀魂画面認識Adapter

## 現在の実装範囲

画面認識Adapterは、macOSの指定矩形を読み取り専用で取得し、設定済みslotから牌を分類する。
同じ結果を既定3 frame連続で確認し、現在の`GameState`に対して合法な差分だけを
`majsoul-normalized-event/v1`へ変換する。マウス・キーボード入力、ブラウザ注入、通信傍受、
非公開API利用は行わない。

```text
許諾済みの対局画面
    │ macOS Screen Recording / 保存PNG
    ▼
layout crop → template分類 → 連続frame安定化
    ▼
局面差分検出 → 正規化event → Rust GameState検証
    ▼
既存realtime解析 → read-only viewer
```

認識できる差分は、自家のツモ・打牌、各家の河への打牌追加、ドラ表示追加である。
layoutに`meld_<player>_<group>`領域を追加すればチー・ポン・槓も検出する。
他家の非公開ツモ牌は保存せず、合法な手番進行から`tile=null`のツモだけを補う。

誤った状態で解析を継続しないことを優先している。未知牌、閾値未満、複数河の同時変化、
slot飛び、非単調なドラ・河、現在のphaseと矛盾する差分ではイベントを送らず停止する。

## 1. 画面領域とlayoutの調整

初回だけ画像認識とWebSocket serviceの依存関係を導入する。

```bash
.venv/bin/pip install -e '.[vision,service]'
```

雀魂を固定解像度・固定倍率で表示し、対象画面について必要な許諾を確認する。
macOSの「プライバシーとセキュリティ > 画面収録」で実行terminal/Codexに許可した後、
1 frameを保存する。

```bash
PYTHONPATH=python .venv/bin/python tools/capture_vision_frame.py \
  local-data/calibration.png \
  --left 0 --top 0 --width 1280 --height 720
```

[perception_layout.example.json](perception_layout.example.json)は座標形式の例であり、
雀魂のwindow、拡大率、themeに合わせて全slotを実画像に重ねて調整する必要がある。
`observer=0`では`river_0`が自家、1が下家、2が対面、3が上家である。

## 2. 自分の画面からtemplateを作る

権利上、雀魂の牌画像はrepositoryへ同梱しない。手元の許諾済み画像から、34種、赤5、空slotを
切り出す。engine内部表記は`m1..m9`、`p1..p9`、`s1..s9`、`z1..z7`、赤5は
`m5_red`等である。

```bash
PYTHONPATH=python .venv/bin/python tools/extract_vision_template.py \
  local-data/calibration.png \
  --layout docs/perception_layout.example.json \
  --region hand --slot 0 --label m1 \
  --variant bottom --templates local-data/vision-templates
```

同じ牌の上下左右・theme違いは`--variant top`等で複数登録できる。空slotも
`--label empty --variant river-right`のように各向きで登録する。全slotを分類できなければ
Adapterは安全のため待機する。

## 3. 保存frameで先に検証する

設定の`capture`を次のように変え、時系列PNGを名前順に配置する。

```json
{
  "capture": {
    "mode": "directory_replay",
    "directory": "../local-data/vision-replay"
  }
}
```

serverとAdapterを別terminalで起動する。

```bash
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py --port 8787
PYTHONPATH=python .venv/bin/python tools/run_vision_adapter.py \
  local-data/vision-replay.json --max-frames 300 --log-level DEBUG
```

Reviewer UIは`majsoul-vision-local` sessionを検出してviewerへ接続する。

## 4. live画面で起動する

[majsoul_vision.example.json](majsoul_vision.example.json)をコピーし、capture座標、layout、
template path、開始局の`bootstrap`を実際の友人戦に合わせる。`authorization_reference`は
必要な許諾を確認した記録の識別子であり、単なる文字列入力が許諾そのものを代替するものではない。

```bash
PYTHONPATH=python .venv/bin/python tools/run_vision_adapter.py \
  local-data/majsoul-vision.json --log-level INFO
```

## 制限

現在は1局開始時の13枚・ドラ1枚を基準にするphase-1実装である。次局への自動遷移、局情報・点数の
OCR、立直牌の横向き検出、和了・流局結果、他家ツモ切り判定は未実装である。したがって、現時点で
友人戦半荘全体を完全自動追跡できるとは扱わない。まず保存frameでslotとtemplate精度を測定し、
1局の通常ツモ・打牌追跡を通すための基盤である。
