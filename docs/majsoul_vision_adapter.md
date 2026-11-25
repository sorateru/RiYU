# 雀魂画面認識Adapter

## 現在の実装範囲

画面認識Adapterは、macOSの指定矩形または明示的に選んだSafari windowを読み取り専用で取得し、
scene anchorで雀魂画面であることを確認してから設定済みslotの牌を分類する。
同じ結果を既定3 frame連続で確認し、現在の`GameState`に対して合法な差分だけを
`majsoul-normalized-event/v1`へ変換する。マウス・キーボード入力、ブラウザ注入、通信傍受、
非公開API利用は行わない。

```text
許諾済みの対局画面
    │ macOS Screen Recording / 保存PNG
    ▼
scene guard / 任意の位置補正 → layout crop → template分類 → 連続frame安定化
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
生成して自動復旧する。起動時や一時的な画面消失後が局途中でも、自分の打牌待ちで、手牌・ツモ牌、
全河、全副露、ドラ、局情報、残り枚数、点数、親、リーチ種別が揃えば公開情報snapshotを直接検証して
復元する。副露は牌、横向き位置、裏向き牌、加槓の重なりから種類・鳴き元・鳴いた牌が一意に
決まる場合だけ採用する。旧profileの個別副露markerも読み取れるが、新profileでは要求しない。情報不足は
`PARTIAL`、差分矛盾は`DESYNCHRONIZED`として保持し、牌状態まで捨てたり値を推測したりしない。

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

横向きのリーチ宣言牌は`slot_variants`で、同じ論理河slotへ縦・横の候補cropを登録する。
各候補は独立に分類され、1位と2位の差が閾値未満なら向きを確定しない。河は固定格子なので、
副露用の可変幅走査ではなく論理slotごとの候補比較を継続する。

```json
{
  "slot_variants": {
    "river_0": [
      [
        {"rect": [505, 465, 30, 42], "rotation": 0, "orientation": "vertical"},
        {"rect": [500, 470, 42, 30], "rotation": 90, "orientation": "horizontal"}
      ]
    ]
  }
}
```

実際の`slot_variants`は、その河の全論理slot分を記述する。横向き候補が安定して選ばれた新規河牌は
リーチ宣言の視覚証拠になる。通常の`riichi_<player>` markerも併用できる。

副露は[Qiita「雀魂の画像認識 Vol.4」](https://qiita.com/xenepic/items/786f110eb5be6ccf246d)
と同じ方針を主経路にする。`meld_scans`で副露領域を射影補正し、右端から縦牌cropと横牌cropを
同時分類して高confidence側だけを採用する。横牌を選んだ場合は上段cropも確認し、同牌なら加槓と
判定する。選択した向きの幅だけcut-lineを左へ進める。RiYUでは記事方式に安全条件を追加し、
縦横差が`tile_minimum_margin`未満、途中牌がunknown、上段牌が曖昧な場合はcut-lineを進めず
副露全体をunknownにする。最初の誤りを後続牌へ連鎖させないためである。

```json
{
  "meld_scans": {
    "meld_0_0": {
      "rect": [550, 845, 1274, 138],
      "perspective_destination": [[-22, -30], [0, 137], [1273, 137], [1319, -30]],
      "horizontal_rect": [-87, 76, 87, 58],
      "vertical_rect": [-58, 51, 58, 83],
      "upper_horizontal_rect": [-87, 7, 87, 58],
      "horizontal_step": 102,
      "vertical_step": 71,
      "max_tiles": 4,
      "from_offsets": [3, 1, 2, 3]
    }
  }
}
```

上の数値は記事の1920×1080系スクリーンショットを説明するための値で、RiYUの1470×762 profileへ
そのままコピーしない。各席・各副露位置について保存実画面へoverlayし、ROI、射影後の牌位置、
縦横strideを調整する。`from_offsets`は右から数えた横牌位置と鳴き元の相対席の対応であり、
実画面で一意に検証できない位置は`null`にする。その場合は牌種を読めても副露eventは確定しない。

旧profileとの互換用に`meld_layouts`も残す。`meld_scans`が同じ領域にあれば走査方式を優先し、
なければ空・チー・ポン・大明槓・暗槓・加槓の候補配置を比較する。`role`は`tile`、`empty`、
`back`で、取得牌のslotだけ`called=true`にする。`from_offset`は副露者から見た鳴き元の相対席である。

```json
{
  "meld_layouts": {
    "meld_0_0": [
      {
        "name": "pon-right",
        "kind": "pon",
        "from_offset": 1,
        "slots": [
          {"rect": [900, 620, 38, 56], "role": "tile"},
          {"rect": [940, 628, 56, 38], "role": "tile", "rotation": 90, "called": true},
          {"rect": [998, 620, 38, 56], "role": "tile"}
        ]
      },
      {
        "name": "ankan",
        "kind": "ankan",
        "slots": [
          {"rect": [900, 620, 38, 56], "role": "back"},
          {"rect": [940, 620, 38, 56], "role": "tile"},
          {"rect": [980, 620, 38, 56], "role": "tile"},
          {"rect": [1020, 620, 38, 56], "role": "back"}
        ]
      }
    ]
  }
}
```

上記座標は形式例であり、そのまま本番に使用しない。4人×4副露について実画面に重ねて調整する。
暗槓の裏向き牌は全席共通の`templates/visuals/back__<variant>.png`を使用するため、席・副露番号別の
専用markerは不要である。

別タブ、壁紙、RiYU自身を牌として誤認しないため、変化しにくい歯車、卓中央の固定枠等を
`scene_anchors`へ2個以上登録する。`search_rect`内でanchorを探索し、見つからないframeは
`game_screen_not_found`として全slot解析を停止する。`align=true`はwindow位置や倍率の変動を
実録画で検証済みの場合だけ有効にする。複数anchorから平行移動・倍率・軽微な回転を推定し、
anchor間の幾何が一致しないframeは補正せず拒否する。`false`ではscene確認だけを行い、牌cropを
再サンプリングしない。

```json
{
  "scene_anchors": {
    "settings_button": {
      "rect": [1100, 10, 64, 64],
      "search_rect": [950, 0, 300, 180],
      "minimum_confidence": 0.82,
      "minimum_scale": 0.9,
      "maximum_scale": 1.1,
      "align": true,
      "required": true
    }
  }
}
```

anchor cropは牌と同様にユーザー自身の画面から作る。

```bash
PYTHONPATH=python .venv/bin/python tools/extract_vision_anchor.py \
  local-data/calibration.png \
  --layout local-data/perception_layout.json \
  --anchor settings_button --variant default \
  --templates local-data/vision-templates
```

座標を確定する前にoverlayを出力し、手牌、ツモ牌、ドラ、河、中央表示の矩形を目視確認する。

```bash
PYTHONPATH=python .venv/bin/python tools/render_vision_layout.py \
  local-data/calibration.png \
  --layout local-data/perception_layout.json \
  --output local-data/calibration-overlay.png
```

## 2. 自分の画面からtemplateを作る

権利上、雀魂の牌画像はrepositoryへ同梱しない。画像検索で見つけた第三者の雀魂スクリーンショットも、
明示的な利用許諾がなければtemplateとして取得・再配布しない。また、CC0の一般的な麻雀牌素材は
UI表示には使えても雀魂固有の描画とは異なり、認識templateの代用にはならない。
手元の許諾済み画像から、34種、赤5、空slotを切り出す。engine内部表記は
`m1..m9`、`p1..p9`、`s1..s9`、`z1..z7`、赤5は
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
抽出toolは`template_manifest.json`へラベル、向き、対象領域、元画像、crop、画像size、登録日時を
記録する。許諾済みのネット画像を事前にローカル保存して使う場合は、`--source-reference`へ元URLを
指定する。現在のUI、牌design、画面方向、倍率に実画像で一致しない素材は登録しない。

記事の牌一覧や赤ドラ表のようなtemplate sheetは、layoutを経由せず直接cropできる。ネット画像は
本番中に取得せず、事前に保存・目視照合して追加variantにする。既存templateは置き換えない。

```bash
PYTHONPATH=python .venv/bin/python tools/extract_vision_template.py \
  local-data/article-reference/vol4/red-dora.png \
  --crop 127 205 126 205 --label p5_red --variant qiita-vol4-red-dora \
  --target-region article-red-dora \
  --source-reference 'https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/381769/a582dd20-28e6-d739-ca9e-101a35eda223.png' \
  --templates local-data/vision-templates
```

profile全体の不足は機械判定できる。34種・赤5三種、全河、4人×4副露、場風・親・リーチ・
結果marker、scene anchorが一覧化される。副露復元は160個の種類・鳴き元markerではなく、
16領域の`meld_scans`または`meld_layouts`と共有裏向き牌templateの有無を検査する。

副露牌や裏向き牌も既存の抽出toolで登録できる。

```bash
PYTHONPATH=python .venv/bin/python tools/extract_vision_template.py \
  local-data/calibration.png \
  --layout local-data/perception_layout.json \
  --region meld_0_0 --meld-pattern ankan --meld-slot 0 \
  --label back --variant bottom \
  --templates local-data/vision-templates
```

```bash
PYTHONPATH=python .venv/bin/python tools/manage_vision_profile.py check \
  local-data/majsoul-vision.json
```

## 3. OCRと視覚markerを設定する

layoutの`text_fields`は、全項目を1回のTesseract processで読む。次局自動遷移には
`hand_number`、`honba`、`riichi_sticks`、`score_0..score_3`が必要で、場風と親はそれぞれ
OCRの`wind`・`dealer`、または`wind_east..wind_north`・`dealer_0..dealer_3` markerで得る。
終局は`delta_0..delta_3`を3 frame確認し、現在のGameState phaseと組み合わせて
ロン・ツモ・通常流局、和了者、放銃者を導出する。差分表示を読めないthemeでは、
結果後の`score_0..score_3`と直前点数の差も使える。サンプル座標は形式例なので実画面に合わせて必ず修正する。
装飾枠に接する連結成分はOCR前に除去し、縦向きscoreの低confidence結果はそのfieldだけを
single-line modeで再認識する。
小さい局番号など一括OCRで安定しないfieldは`ocr_psm`に`7`（1行）または`10`（1文字）を
指定すると、そのfieldだけを元の色のcropから再認識する。`ocr_minimum_confidence`はfield固有の
下限であり、許容値が狭く実画面で誤読試験を済ませたfieldにだけ設定する。

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
    "draw_reason_four_winds": {"rect": [480, 210, 320, 70]}
  }
}
```

UI markerは導出できない意味だけに限定する。リーチは横向き河牌、ロン・ツモ・
和了者・放銃者は点数差分とGameState phaseから導出するため、席別templateは必須ではない。
途中流局の理由は公開情報から常に一意とは限らないため、ここだけ意味別templateを使う。
markerには`on` cropが必要で、`off` cropも用意すると誤検出をより強く拒否できる。

```bash
PYTHONPATH=python .venv/bin/python tools/extract_vision_marker.py \
  local-data/four-winds-on.png \
  --layout docs/perception_layout.example.json \
  --marker draw_reason_four_winds --state on \
  --templates local-data/vision-templates

PYTHONPATH=python .venv/bin/python tools/extract_vision_marker.py \
  local-data/four-winds-off.png \
  --layout docs/perception_layout.example.json \
  --marker draw_reason_four_winds --state off \
  --templates local-data/vision-templates
```

途中流局では
`draw_reason_nine_terminals`、`four_winds`、`four_riichi`、`four_kan`、`triple_ron`の
いずれかを必要とする。従来の`result_*`、`winner_*`、`loser_*`、`riichi_*`もfallbackとして
使用できるが、新規profileの完了条件には含めない。

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

認識速度、scene miss、未知slot数は保存frameで繰り返し測定できる。正解付き評価では
[perception_benchmark.example.json](perception_benchmark.example.json)と同じ形式でframeごとの牌、OCR、
marker、`melds`の種類・牌・取得位置・鳴き元、期待event、局ID、半荘IDを記録する。局・半荘の完全同期率、誤event、取りこぼし、desync、
再同期もJSONへ保存する。`--debug-dir`は失敗frame、ROI、上位2候補、confidence、template差分、
副露配置候補、推定event、GameState拒否理由を保存するため、通常運用では指定しない。

```bash
PYTHONPATH=python .venv/bin/python tools/benchmark_vision_profile.py \
  local-data/majsoul-vision.json local-data/vision-replay \
  --annotations local-data/vision-replay/annotations.json \
  --output local-data/vision-benchmark-result.json \
  --debug-dir local-data/vision-benchmark-failures
```

profileは画像を公開せず、利用者自身の端末間でchecksum付きbundleとして移動できる。

```bash
PYTHONPATH=python .venv/bin/python tools/manage_vision_profile.py export \
  local-data/majsoul-vision.json local-data/my-profile.riyu-vision
PYTHONPATH=python .venv/bin/python tools/manage_vision_profile.py import \
  local-data/my-profile.riyu-vision local-data/imported-profile
```

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
  local-data/majsoul-vision.json \
  --save-latest-frame local-data/latest-capture.png
```

launcherは起動前にポート二重使用、画面収録権限、scene、profile不足を確認する。同じ8787番で
既存RiYUが動いている場合は、新Adapterを誤接続させず終了する。`--save-latest-frame`は診断中だけ
指定し、対局画面を含むPNGをローカルで上書き保存する。共有やGit追加は行わない。

評価済みAIを使う場合は`--checkpoint`を明示する。指定がなければシャンテン・受け入れ・合法手等の
厳密計算のみで、Web UIにも「AI未読込」と表示される。学習済み・評価済みでないcheckpointは
server起動時に拒否される。

```bash
PYTHONPATH=python .venv/bin/python tools/run_majsoul_webapp.py \
  local-data/majsoul-vision.json \
  --checkpoint artifacts/checkpoints/validated.pt
```

WebアプリはVision設定の`websocket_url`と同じポートで配信され、既定で
`http://127.0.0.1:8787/#realtime`からリアルタイム画面へ移動できる。トップ画面の
「リアルタイム」からも同じ画面を開ける。UIは`source_acquisition=screen_capture`、
`source_kind=authorized_external_adapter`、許諾記録ありの雀魂Vision Adapterだけを表示する。
バックエンドとAdapterは`Ctrl+C`でまとめて終了できる。
設定した画面座標は、Safariの実際の表示位置と一致している必要がある。
`capture.mode=macos_region`では雀魂タブが固定矩形内で前面に見えている必要があり、全画面表示は
必須ではない。RiYUを別windowで前面表示しながら取得する場合は`capture.mode=macos_window`と
Safari windowを指定する。window IDはSafari再起動で変わるため、`window_owner: "Safari"`なら
起動時に前面Safari windowのIDを読み取り専用AppleScriptで解決する。`content_region`はwindow
capture PNG内のゲーム領域で、layoutの出力サイズへ正規化される。

```json
{
  "capture": {
    "mode": "macos_window",
    "window_owner": "Safari",
    "content_region": {"left": 0, "top": 140, "width": 1470, "height": 762},
    "width": 1470,
    "height": 762
  }
}
```

Safari内で雀魂とは別のタブへ切り替えた場合は、window captureでもscene guardが停止する。
Web UIには未知slot、最低confidence、連続frame数、profile不足、認識済みの手牌・ツモ牌・
ドラ枚数が表示され、別タブや座標ずれを接続不良と区別できる。

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
