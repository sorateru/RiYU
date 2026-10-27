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
layoutに`meld_<player>_<group>`領域を追加すればチー・ポン・槓も検出する。さらに
OCRと視覚markerからリーチ、ロン・ツモ・流局、点数移動、新局を確定し、半荘内で次局へ
自動遷移する。
他家の非公開ツモ牌は保存せず、合法な手番進行から`tile=null`のツモだけを補う。

誤った状態で解析を継続しないことを優先している。未知牌、閾値未満、複数河の同時変化、
slot飛び、非単調なドラ・河、現在のphaseと矛盾する差分ではイベントを送らず、その局を
desync状態にする。次の13枚・ドラ1枚・空の河を認識すると、OCR済み局情報から検証済みsnapshotを
生成して自動復旧する。起動時が局途中でも、13枚の手牌とツモ牌、ドラ1枚、全河、局情報、残り枚数が
揃い、副露がない自分の打牌待ちなら公開情報snapshotを直接検証して復元する。それ以外はイベント順を
推測せず、自分の安全に復元できる手番または次局を待つ。

## 1. 画面領域とlayoutの調整

初回だけ画像認識とWebSocket serviceの依存関係を導入する。

```bash
.venv/bin/pip install -e '.[vision,service]'
brew install tesseract
```

雀魂を固定解像度・固定倍率で表示し、対象画面について必要な許諾を確認する。
macOSの「プライバシーとセキュリティ > 画面収録」で実行terminal/Codexに許可した後、
1 frameを保存する。
Retina画面で取得PNGが2倍pixelになる場合は、Adapterが指定した論理サイズへ自動正規化する。

```bash
PYTHONPATH=python .venv/bin/python tools/capture_vision_frame.py \
  local-data/calibration.png \
  --left 0 --top 0 --width 1280 --height 720
```

[perception_layout.example.json](perception_layout.example.json)は座標形式の例であり、
雀魂のwindow、拡大率、themeに合わせて全slotを実画像に重ねて調整する必要がある。
`observer=0`では`river_0`が自家、1が下家、2が対面、3が上家である。
`rotations`には分類前に牌cropを時計回りに回す角度を指定できる。標準配置では
`river_1=90`、`river_2=180`、`river_3=270`とし、上下左右で同じ牌templateを共有する。

座標を確定する前にoverlayを出力し、手牌、ツモ牌、ドラ、河、中央表示の矩形を目視確認する。

```bash
PYTHONPATH=python .venv/bin/python tools/render_vision_layout.py \
  local-data/calibration.png \
  --layout local-data/perception_layout.json \
  --output local-data/calibration-overlay.png
```

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

## 3. OCRと視覚markerを設定する

layoutの`text_fields`は、全項目を1回のTesseract processで読む。次局自動遷移には
`hand_number`、`honba`、`riichi_sticks`、`score_0..score_3`が必要で、場風は
`wind_east..wind_north`、親は`dealer_0..dealer_3` markerから得る。終局には
`delta_0..delta_3`を使い、差分表示を読めないthemeでは結果後の`score_0..score_3`との差から
自動計算する。サンプル座標は形式例なので実画面に合わせて必ず修正する。
装飾枠に接する連結成分はOCR前に除去し、縦向きscoreの低confidence結果はそのfieldだけを
single-line modeで再認識する。

```json
{
  "text_fields": {
    "score_0": {
      "rect": [600, 595, 100, 28],
      "type": "integer",
      "min": -99900,
      "max": 999900
    }
  },
  "markers": {
    "riichi_0": {"rect": [560, 570, 160, 20]},
    "result_ron": {"rect": [480, 210, 320, 70]},
    "winner_0": {"rect": [520, 530, 240, 70]}
  }
}
```

markerには`on` cropが必要で、`off` cropも用意すると誤検出をより強く拒否できる。

```bash
PYTHONPATH=python .venv/bin/python tools/extract_vision_marker.py \
  local-data/riichi-on.png \
  --layout docs/perception_layout.example.json \
  --marker riichi_0 --state on \
  --templates local-data/vision-templates

PYTHONPATH=python .venv/bin/python tools/extract_vision_marker.py \
  local-data/riichi-off.png \
  --layout docs/perception_layout.example.json \
  --marker riichi_0 --state off \
  --templates local-data/vision-templates
```

結果markerは`result_ron`、`result_tsumo`、`result_draw`、`result_abortive_draw`、
席markerは`winner_0..3`と`loser_0..3`を使う。途中流局では
`draw_reason_nine_terminals`、`four_winds`、`four_riichi`、`four_kan`、`triple_ron`の
いずれかを必要とする。

保存画像1枚の認識値とconfidenceは次で確認できる。

```bash
PYTHONPATH=python .venv/bin/python tools/inspect_vision_annotations.py \
  local-data/result-screen.png \
  --layout docs/perception_layout.example.json \
  --templates local-data/vision-templates
```

## 4. 保存frameで先に検証する

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

## 5. live画面で起動する

[majsoul_vision.example.json](majsoul_vision.example.json)をコピーし、capture座標、layout、
template path、開始局の`bootstrap`を実際の友人戦に合わせる。`authorization_reference`は
必要な許諾を確認した記録の識別子であり、単なる文字列入力が許諾そのものを代替するものではない。
校正画像、設定、templateは`local-data/`に保存し、Gitには含めない。

製品利用時はCodexから起動せず、画面収録権限を付与したTerminalから実行する。
初回またはフロント変更後に、牌譜解析とリアルタイム解析を統合したメインWebアプリをbuildする。

```bash
npm run build --prefix apps/web
```

通常は次の1コマンドで、ローカルAPI、Webアプリ、Vision Adapterをまとめて起動する。

```bash
PYTHONPATH=python .venv/bin/python tools/run_majsoul_webapp.py \
  local-data/majsoul-vision.json
```

WebアプリはVision設定の`websocket_url`と同じポートで配信され、既定で
`http://127.0.0.1:8787/#realtime`からリアルタイム画面へ移動できる。トップ画面の
「リアルタイム」からも同じ画面を開ける。UIは`source_acquisition=screen_capture`、
`source_kind=authorized_external_adapter`、許諾記録ありの雀魂Vision Adapterだけを表示する。
バックエンドとAdapterは`Ctrl+C`でまとめて終了できる。
設定した画面座標は、Safariの実際の表示位置と一致している必要がある。
Safariのwindow名ではなく固定された画面矩形を取得するため、雀魂タブがその矩形内で前面に見えて
いる必要がある。全画面表示は必須ではない。Web UIには未知slot、最低confidence、連続frame数、
認識済みの手牌・ツモ牌・ドラ枚数が表示され、別タブや座標ずれを接続不良と区別できる。

個別起動して問題を切り分ける場合は、serverとAdapterを別Terminalで実行する。

```bash
PYTHONPATH=python .venv/bin/python tools/serve_analysis.py --port 8787
PYTHONPATH=python .venv/bin/python tools/run_vision_adapter.py \
  local-data/majsoul-vision.json --log-level INFO
```

## ツモ切りの既知性と残る実機作業

`tsumogiri_<player>_<river_slot>` marker（または全捨て牌共通の表示がある場合は
`tsumogiri_<player>`）が確定した場合は、その値を使う。雀魂themeが静止画上に判定可能な
表示を残さない場合、他家ツモ切りは原理的に静止画から確定できない。この場合は
`tsumogiri_known=false`として履歴へ保存し、Sequence入力では0/1のどちらにも寄せず0.5にする。
誤って手出し扱いすることはない。

コード上の次局遷移、OCR、リーチ、終局、既知性、次局復旧は実装済みである。残るのは利用する
雀魂window・倍率・themeごとの座標、牌template、marker templateの作成と、実際の友人戦録画を
使ったconfidence調整である。UI変更やサービス固有画像をrepositoryへ同梱しても、この実機校正を
代替することはできない。
