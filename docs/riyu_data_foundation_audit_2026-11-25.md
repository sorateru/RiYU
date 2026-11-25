# RiYU 牌譜データ基盤 調査・再設計報告

調査日: 2026-11-25  
対象: `/Users/ciel/Documents/ChatGPT/麻雀`  
制約: 読み取り専用調査。元データ、加工済みデータ、checkpointは変更・削除・再生成していない。

## 0. 結論

1. **主な元牌譜はJSONではない。** `data/high_rank/2020.db` のSQLite `logs.log_content`に、1対局ごとの天鳳XMLがbzip2圧縮BLOBとして保存されている。
2. DBには328,424対局、3,196,694局、圧縮ログ本体2,046,905,381 bytes、DBファイル2,211,529,728 bytesがある。期間は`log_id`上で2020-01-01〜2020-12-31。SQLite整合性は`ok`、XML全件の解凍・parseも成功した。
3. 内訳は四麻半荘218,858、四麻東風35,246、三麻半荘74,143、三麻東風177。現行変換器・Rust engineは四麻前提なので、三麻は別dataset/profileに隔離すべきである。
4. 元XMLは全員の配牌・全ツモ・全打牌を物理牌IDで持つ。実際の進行上の全手牌は復元できる。さらに全対局に`SHUFFLE seed`があり、天鳳の公開山生成方式を正しく実装・検証すれば未使用山・王牌まで一意復元できる可能性が高い。ただし現行コードにはその実装がなく、イベント列だけから未使用山・裏ドラ未公開部分を推測してはならない。
5. 現行変換は、ルール`GO`、shuffle seed、日時、段位/R、プレイヤー識別子、切断、物理牌コピーIDなどを落とす。特に`rules=None`のままRust既定ルールで再生するため、元の東風/半荘・三麻・速卓等の区別が学習状態へ正しく継承されない。
6. 既存200k decision datasetはPASS 0件、ロン0件、ツモ0件、カン0件である。別のresponse補助datasetにPASS 39,758件、sequence-v2にPASS 8,715件があるが、未応答をすべてPASSにしたものではなく「次が通常ツモなら全員が見送った」と確定できる窓だけを補っている。
7. 既存Parquetには重複合法手が残る（200k: 50 entry、response: 247、sequence-v2: 57）。現行Rustはsort+dedup済みだが、既存artifactは再生成しない限り直らない。
8. 新基盤は、`raw immutable -> typed canonical events -> replay/checkpoint -> decision/opportunity -> legal action -> oracle/exact/factual/simulated labels`の層に分ける。観測入力と非公開教師、実牌譜の事実と反実仮想simulationを物理的にもschema上も分離する。
9. 正本はtyped Parquet（ZSTD）、監査・少量交換はJSONL.zst、元BLOBは原形式のままimmutable保存を推奨する。1判断ごとに巨大なJSON stateを複製する現方式は長期基盤にしない。

本文のラベル:

- **事実**: 実ファイル、全件走査、ソースコード、一次資料で確認。
- **推定**: 事実からの外挿。仮定を明記。
- **未確定**: 現在の証拠では断定しない。
- **提案**: 新設計。

---

## 1. 元データ

### 1.1 保存場所と実体

| 項目 | 実測 |
|---|---:|
| 元DB | `data/high_rank/2020.db` |
| SQLite table | `logs`, `last_downloads` |
| DBファイル容量 | 2,211,529,728 bytes (約2.06 GiB) |
| 対局row | 328,424 |
| `log_id` distinct | 328,424 |
| `log_hash` distinct | 328,424 |
| 圧縮BLOB総量 | 2,046,905,381 bytes |
| 解凍XML総量 | 4,703,512,870 bytes |
| 局数 (`INIT`) | 3,196,694 |
| XML直列要素 | 304,559,154 |
| SQLite integrity | `ok` |
| bzip2/XML parse error | 0 / 328,424 |
| 期間 (`log_id`) | 2020-01-01 00時〜2020-12-31 23時 |

SQLite schema:

```sql
CREATE TABLE logs(
  log_id text primary key,
  is_tonpusen int,
  is_hirosima int,
  is_processed int,
  was_error int,
  log_content text,
  log_hash text
);
CREATE TABLE last_downloads(name text, date int);
```

`log_content`の宣言型は`text`だが、実値はbzip2 BLOBである。全rowは`is_processed=1, was_error=0`。

### 1.2 ルール種別

|  | 四麻 | 三麻 (`is_hirosima=1`) | 合計 |
|---|---:|---:|---:|
| 半荘 (`is_tonpusen=0`) | 218,858 | 74,143 | 293,001 |
| 東風 (`is_tonpusen=1`) | 35,246 | 177 | 35,423 |
| 合計 | 254,104 | 74,320 | 328,424 |

`GO.type`は11種類あり、支配的な値は169: 218,816件、185: 74,136件、225: 35,242件である。元bitfieldをそのまま保持し、解釈済みrule profileと併存させるべきである。

### 1.3 取得元と利用条件

**事実:** 同梱された第三者READMEは、鳳凰卓牌譜を`phoenix-logs`で取得したDBが作者から提供され、Google Driveへ置かれた経路を記す。[third-party README](../third_party/riichi-mahjong-ml-dataset/README.md)

**事実:** `phoenix-logs`のコードはMITだが、これは牌譜内容の利用許諾ではない。天鳳公式は、牌譜の著作権は天鳳にあるとし、一般麻雀への応用は問い合わせが必要、天鳳を必要としないサービスや競合製品への応用を不可としている。[天鳳公式マニュアル](https://tenhou.net/man/index.html)

**事実:** 現行manifestは`rights_basis=explicit_permission`、`authorization_reference=user-confirmed-permitted-2026-10-19`、`machine_learning_allowed=true`、`redistribution_allowed=false`と宣言している。

**判定:** リポジトリ内には許諾文書本文はなく参照文字列だけである。実運用・再配布の法的可否を本調査だけで確定しない。少なくとも以下を満たすまで、raw DBと派生データを外部公開しない。

- 許諾文書をアクセス制御されたrights registryで参照可能にする。
- 許諾対象（研究/商用、モデル学習、派生物、保存期間、再配布）を明記する。
- コードのMITと牌譜の権利を分離する。

### 1.4 段位・Rating

`UN.dan`の実プレイヤー席1,239,376件は次の通り。コード対応は天鳳mjlog仕様の公開解析資料と整合する。

| dan code | 段位 | 席数 | 比率 |
|---:|---|---:|---:|
| 16 | 七段 | 725,026 | 58.50% |
| 17 | 八段 | 403,523 | 32.56% |
| 18 | 九段 | 96,489 | 7.79% |
| 19 | 十段 | 11,024 | 0.89% |
| 20 | 天鳳位 | 3,314 | 0.27% |

三麻の空席74,320件は`dan=0, rate=1500`であり、実プレイヤーとして数えてはいけない。実プレイヤーのR分布:

| R帯 | 席数 |
|---|---:|
| 2000–2099 | 243,347 |
| 2100–2199 | 693,915 |
| 2200–2299 | 274,475 |
| 2300–2399 | 26,844 |
| 2400–2499 | 795 |

最大観測値は2462.54。元XMLには名前もあるため、raw DBは個人識別子を含むものとして扱う。Parquet manifestの`contains_personal_identifiers=false`は匿名化後artifactにだけ適用できる。

各対局の最初の`UN`だけを採用し、空名と再接続時の重複を除いた不可逆hash集計では、非空プレイヤー識別子3,712、対局席exposure 1,239,376。最多1名は3,886席（0.314%）、上位10名は31,575席（2.548%）、上位100名は209,562席（16.909%）。極端な単独player支配ではないが、上位100名の影響は無視できず、player-aware評価が必要である。名前変更・共有IDはrawだけでは判定できない。

### 1.5 元XMLの完全な構造

全件走査で確認したタグと属性の和集合:

| tag | 属性 | 意味 |
|---|---|---|
| `mjloggm` | `ver` | 1対局を包むroot、牌譜version |
| `SHUFFLE` | `seed`, `ref` | 山生成seedと参照情報 |
| `GO` | `type`, `lobby` | ルール/卓/速度等bitfield、ロビー |
| `UN` | `n0..n3`, `dan`, `rate`, `sx` | 名前、段位、R、属性。再接続で複数回現れ得る |
| `TAIKYOKU` | `oya` | 対局開始時の親 |
| `INIT` | `seed`, `ten`, `oya`, `hai0..hai3` | 局開始。局/本場/供託/骰子/表ドラ、点数、親、全員配牌 |
| `T/U/V/W{0..135}` | なし | player 0/1/2/3のツモ。suffixは物理牌ID |
| `D/E/F/G{0..135}` | なし | player 0/1/2/3の打牌。suffixは物理牌ID |
| `N` | `who`, `m` | 鳴き/槓。`m`は副露bitfield |
| `REACH` | `who`, `step`, step 2で`ten` | リーチ宣言/受理、受理後点数 |
| `DORA` | `hai` | 追加ドラ表示牌 |
| `AGARI` | `ba`,`hai`,`m`,`machi`,`ten`,`yaku`,`yakuman`,`doraHai`,`doraHaiUra`,`who`,`fromWho`,`sc`,`owari`,`paoWho` | 和了。winner/from、和了手、待ち、符点/点数区分、役、ドラ、点差、最終点/精算、責任払い |
| `RYUUKYOKU` | `type`,`ba`,`hai0..hai3`,`sc`,`owari` | 流局理由、聴牌者の開示手、点差、最終点/精算 |
| `BYE` | `who` | 切断 |

主要配列:

- 物理牌IDは`0..135`、`id // 4`が34牌種、同一牌種内のcopyを保持する。赤5は通常16/52/88で識別される。
- `INIT.seed = [局index, 本場, 供託, 骰子1, 骰子2, 初期ドラ表示牌ID]`。
- `INIT.ten`と`REACH.ten`は100点単位。
- `AGARI.sc`/`RYUUKYOKU.sc`は4人分の`[局前点, delta]`対（100点単位）。
- `owari`は最終点と精算値の対。現行converterは最終点だけを残す。
- `yaku`は`[役ID, 翻]`対、`yakuman`は役満ID列。
- `RYUUKYOKU.type`実測: 通常463,039、`yao9` 44,907、`nm` 1,660、`kaze4` 1,534、`reach4` 290、`kan4` 69、`ron3` 73。

### 1.6 代表的な実例

元データはJSONではない。個人名を除いた構造例:

```xml
<mjloggm ver="2.3">
  <SHUFFLE seed="..." ref="..."/>
  <GO type="169" lobby="0"/>
  <UN n0="..." n1="..." n2="..." n3="..."
      dan="16,17,16,18" rate="..." sx="..."/>
  <TAIKYOKU oya="0"/>
  <INIT seed="0,0,0,2,2,93" ten="250,250,250,250" oya="0"
        hai0="...13 ids..." hai1="..." hai2="..." hai3="..."/>
  <T95/><D35/><U.../>
  <N who="0" m="5195"/>
  <REACH who="3" step="1"/><G21/><REACH who="3" ten="..." step="2"/>
  <AGARI who="1" fromWho="2" machi="..." ten="30,7700,0"
         yaku="..." doraHai="..." doraHaiUra="..." sc="..."/>
</mjloggm>
```

現行converterが同じ実牌譜から作るobserver 0の冒頭（実測例）:

```json
{
  "format": "mahjong-ai-event-log/v1",
  "log_id": "7bae33...0acb-r00",
  "rules": null,
  "observations": {
    "0": [
      {
        "sequence": 0,
        "type": "round_start",
        "round": {"wind":"east","hand_number":1,"honba":0,"riichi_sticks":0,"dealer":0},
        "scores": [25000,25000,25000,25000],
        "initial_hand": ["13 tile objects"],
        "dora_indicator": {"kind":"s6","red":false},
        "remaining_tiles": 70
      },
      {"sequence":1,"type":"draw","player":0,"tile":{"kind":"s7","red":false},"rinshan":false},
      {"sequence":2,"type":"discard","player":0,"tile":{"kind":"s9","red":false},"tsumogiri":false,"riichi_declaration":false},
      {"sequence":3,"type":"draw","player":1,"tile":null,"rinshan":false}
    ]
  },
  "result": {
    "initial_scores": [25000,25000,25000,25000],
    "round_score_deltas": [0,8700,-7700,0],
    "final_scores": [8900,42800,26600,21700]
  }
}
```

`observations[1]`では同じsequence 3のtileが見え、他家ツモは`null`になる。4 streamを一組で保持すれば実進行上の全ツモを回収できるが、1 stream単独は意図的に部分観測である。

### 1.7 欠損、重複、破損

**確認済み:** 

- SQLite内部整合性、bzip2解凍、XML parseは全件成功。
- `log_id`、`log_hash`は全件一意。byte-identicalなBLOB重複は検出されない。
- semantic duplicate（別IDだが同じ対局）は全件同値比較していないため未確定。
- 1対局あたり局数は1〜26。途中終了・東風・連荘を含むので固定8局とみなせない。
- DBには三麻74,320対局が混在する。四麻converterに渡すと`INIT observer hand must contain 13 tiles`になる。

既存200k変換report（先頭付近560対局）:

- games 560、rounds 4,044、accepted 3,981、rejected 52、decisions 200,000。
- error 202件: 三麻由来の13枚配牌欠落132、旧版の`south` parse bug 18、expert action not legal 52。
- 現行`_kind_index`は`south`を数牌と誤認しない修正済みだが、既存artifactは旧変換結果。

### 1.8 復元可能性

| 情報 | 判定 | 根拠/境界 |
|---|---|---|
| 全員の初期手牌 | 直接 | `INIT.hai0..3` |
| 全員の実ツモ | 直接 | `T/U/V/W` |
| 全員の各時点実手牌 | 計算可能 | 配牌+ツモ-打牌-副露を物理IDでreplay |
| 全プレイヤー視点 | 計算可能 | 完全情報から各observerへprojection |
| 実際に引かれた山順 | 直接 | draw tag順。嶺上/通常の区分は副露進行から計算 |
| 未使用の山・王牌 | 条件付き計算可能 | `SHUFFLE seed`+公開された天鳳山生成アルゴリズムの実装・fixture検証が必要 |
| seedを捨てたEventLogから未使用山 | 不可能 | 観測されない物理牌の順序は失われる |
| 公開ドラ | 直接 | `INIT.seed[5]`, `DORA` |
| 裏ドラ | 和了時のみ直接 | `AGARI.doraHaiUra`; 非和了局はseed復元が必要 |
| 全員の実PASS | 不可能 | 元XMLにPASS tagがない |
| 次が通常ツモとなった応答窓のPASS | 確定可能 | 誰も鳴き/ロンせず窓が閉じた |
| 鳴き/ロンで打ち切られた他者応答 | 一部censored | 優先順位から一部否定はできるが、全員が実際に何を選んだかは記録されない |

---

## 2. 現在の加工処理

### 2.1 プログラムと段階

```text
SQLite logs.log_content (bzip2 XML)
  -> decode_archive_xml / ElementTree
  -> convert_document (1対局 -> 複数 EventLog round)
  -> 4 observer streams
  -> ObservationReplayer (Rust GameState)
  -> Rust legal_actions
  -> DecisionRecord / SequenceDecisionRecord
  -> ZSTD Parquet
  -> FeatureEncoder / SequenceFeatureEncoder
  -> PyTorch tensor / loss
```

主要コード:

- [Tenhou converter](../python/mahjong_ai/data/tenhou.py)
- [EventLog contract](../python/mahjong_ai/data/log.py)
- [decision/v1 builder](../python/mahjong_ai/data/dataset.py)
- [sequence/v4 builder](../python/mahjong_ai/data/sequence_dataset.py)
- [SQLite importer](../tools/import_tenhou_db.py)
- [sequence importer](../tools/import_tenhou_sequence_db.py)
- [Rust legal actions](../engine/core/src/legal.rs)
- [observable encoder](../python/mahjong_ai/ai/encoder.py)

### 2.2 既存artifact

| dataset | rows | round IDs | bytes | 主なaction |
|---|---:|---:|---:|---|
| `2020-decisions-200k` decision/v1 | 200,000 | 3,981 | 46,008,196 | discard 189,341; riichi 2,895; chi 3,334; pon 4,430 |
| response supplemental v2 | 43,109 | 3,924 | 15,523,986 | pass 39,758; ron 1,980; tsumo 1,371 |
| sequence-v2 | 53,459 | 878 | 20,683,291 | discard 41,646; pass 8,715; chi 702; pon 957; kan 66; riichi 624; ron 428; tsumo 321 |

全28 Parquetの実容量は128,223,669 bytes。ただし`2020-decisions-200k`の8 shardは`decisions-a/b`の8 shardとSHA-256が同一で、ディスク上に同じ200k rowsが二重保存されている。学習globで両方を渡すと教師が二重計上される。

### 2.3 判断抽出

- 自家`discard`を打牌教師にする。
- `REACH step=1`直前の状態で、直後の打牌を含む`riichi` actionを教師にする。
- 自家の`chi`, `pon`, `kan`を教師にする。
- winnerの`ron`, `tsumo`を教師にする。
- 次イベントが通常`draw`で、直前状態に非PASS合法応答があれば`pass`教師を追加する。
- PASS時にロンが合法だった場合、synthetic passをreplayerへ適用し一時/リーチ振聴を更新する。
- higher-priority call等でcensorされた応答はPASSにしない。
- terminal eventはpolicy教師にはするが、その後のstate replayは行わない。
- DB読込順は`ORDER BY log_id`、上限判断数で打ち切るため、既存200kは全2020年の無作為sampleではなく年初側の少数対局に強く偏る。

### 2.4 合法手とAction identity

Rustが観測状態から`discard, riichi, chi, pon, kan, ron, tsumo, abortive_draw, pass`を生成する。現行コードはsort後に`dedup()`する。Python `action_key`はconsume/kan tilesを牌種+赤でsortするが、固定のversioned `ActionId`型ではなくcanonical JSON文字列である。

既存artifactの完全同一legal action重複:

- decision200k: 50
- response supplemental: 247
- sequence-v2: 57

物理的に同じ意味の非赤コピーを別候補として扱わないActionId正規化が必要である。

### 2.5 現在の特徴量

`observable-planes-v1`: `float32[49,34]`。

- 自手牌枚数4 plane、赤、自摸牌/赤
- 4人の河、副露、リーチ、一発
- ドラ表示、既知枚数、見かけ残存枚数
- 親、自席、場風、4人点数、巡目、残り牌、本場、供託
- action phase、自家リーチ

Action: `float32[N,83]`。

- 9 action type、対象牌34、from 4、consume 34、赤、ツモ切り

Sequenceは直近公開event、rules/memory、相手別danger featureを追加する。観測encoderは`private_hands`や`wall`を拒否する。

### 2.6 現在の教師信号

**保存済みdecision/v1:** chosen action、最終順位、最終点。

**保存済みsequence-v2:** 上記に加え、公開event、4人局収支、4人シャンテン、他家聴牌、他家待ち、将来12ツモの牌種、終局種別。

**コード上のsequence/v4（保存済み大規模artifactなし）:** chosen action index、相手別hidden tile counts、次打牌/response/call、chosen actionのfactual放銃・局結果・局収支・最終順位/点、counterfactual用の別列とmask。

重要: 未選択actionのcounterfactual列は現行牌譜変換ではmask=false/0 placeholderであり、結果が観測されたことを意味しない。

実在するsequence-v2 Parquet先頭rowの要約:

```json
{
  "log_id": "7bae33...0acb-r00",
  "observer": 0,
  "event_sequence": 2,
  "expert_action": {"type":"discard","tile":{"kind":"s9","red":false},"tsumogiri":false},
  "legal_actions": ["13 candidate action objects"],
  "final_rank": 4,
  "final_score": 8900,
  "round_score_delta": 0,
  "all_round_score_deltas": [0,8700,-7700,0],
  "all_player_shanten": [3,3,2,3],
  "opponent_tenpai": [false,false,false,false],
  "opponent_waits": [[],[],[],[]],
  "future_draws": [13,31,7,30,22,24,7,1,10,14,23,27],
  "result_type": "agari",
  "dataset_version": "mahjong-ai-sequence-decision/v2"
}
```

`observation`には同時点のobserver可視state、`public_events`には公開履歴がJSON文字列として入る。`future_draws`、他家shanten/waits、最終結果は教師であり入力へ入れてはいけない。

### 2.7 分割

- 現行3-way splitは`raw hanchan hash = log_id.rsplit('-r',1)[0]`で80/10/10相当に分け、半荘を跨がせない。
- 旧`split_by_log`はround suffix付き`log_id`をhashするため、同一半荘の別局がtrain/validationへ分かれ得る。`train_curriculum.py`はこの旧関数を使う。
- 保存済み53k sequence学習は47,407 train / 6,052 validationで、独立holdoutは未使用。同じvalidationで選択・測定されている。

### 2.8 現在失われる情報

| 元XMLの情報 | 現行EventLog/decisionでの状態 |
|---|---|
| `SHUFFLE seed/ref` | 完全に欠落 |
| `GO type/lobby` | 欠落。`rules=None` |
| 日時/生`log_id` | SHA-256化後round IDのみ。日時列なし |
| `UN` 名前/段位/R/sx | 全て欠落 |
| `BYE` | 欠落 |
| 物理牌copy ID | 34種+赤へ縮約 |
| `TAIKYOKU` | 欠落 |
| `owari`の精算値 | 最終点以外は欠落 |
| `AGARI.hai/m/ba/paoWho` | 一部または全部をdecisionで未使用 |
| 役/yakuman/裏ドラ | converterの`round_result`には一部残るが、既存decision/v1とsequence-v2学習列には載らない |
| 流局理由/開示聴牌手 | converterに残るが既存学習ではほぼ未使用 |
| 全山/王牌 | seedを捨てるため復元不能 |
| 応答機会のcensor理由 | 明示列なし |

---

## 3. 生成可能な教師データ

凡例: `D` 元牌譜から直接、`E` rule/replayで厳密計算、`S` 追加simulation、`X` 生成不可。コスト/容量は相対。

### 3.1 方策

| 教師 | 取得 | 有用性 | コスト/容量 | 漏洩/注意 |
|---|---|---|---|---|
| 全合法手 | E | 必須 | 中/中 | observer stateだけで生成。rule version必須 |
| 実選択action | D | 必須 | 低/低 | 打牌/鳴き/リーチ/和了は明確 |
| PASS | D+E | 高 | 中/低 | 次が通常ツモなら確定。call/winで閉じた他者窓はcensored |
| 観測できなかった応答 | X | 負例にしないことが重要 | — | `mask=false`, `opportunity_status=censored` |
| 未選択actionの即時合法性 | E | 高 | 中/中 | 将来結果ではない |
| 未選択actionの将来結果 | S | 高だがbiasあり | 非常に高/高 | simulator/opponent policy/seed/uncertaintyを記録 |

### 3.2 攻撃

| 教師 | 取得 | 有用性 | 計算/容量 | 漏洩 | 境界 |
|---|---|---|---|---|---|
| シャンテン | E | 高 | 低/低 | 自家0、他家高 | 全員の実手牌から厳密。engine差分検証が必要 |
| 受け入れ牌種/理論枚数 | E | 高 | 中/中 | 自家0、実残数は高 | 各候補打牌後に厳密 |
| 実残り山中の受け入れ枚数 | E（seed復元後） | 分析用 | 中/低 | 非常に高 | oracle専用。入力へ混ぜない |
| 聴牌までの巡目 | D+E | 中〜高 | 中/低 | 高 | 実軌跡のみfactual。未選択はS |
| 和了までの巡目 | D | 中 | 低/低 | 高 | 実際のwinnerのみ。未和了はcensored/time-to-event扱い |
| 成立した役 | D | 高 | 低/低 | 終局未来 | `AGARI.yaku/yakuman` |
| 現時点で成立可能な役 | E | 中 | 中〜高/中 | 自家0、他家高 | 「可能」の時間範囲を定義。無制限将来可能性は不適切 |
| 和了打点 | D/E | 高 | 中/低 | 他家/未来は高 | 実結果はD、現局面のron/tsumoはE |
| 未和了結果 | D | 高 | 低/低 | 終局未来 | 放銃/他家ツモ/横移動/流局と局収支を保存 |

### 3.3 守備

| 教師 | 取得 | 有用性 | 計算/容量 | 漏洩 | 境界 |
|---|---|---|---|---|---|
| 他家聴牌/待ち | E | 非常に高 | 中/中 | 非常に高 | full truth手牌から厳密。oracle label |
| 他家フリテン | E（一部条件付き） | 高 | 中/低 | 高 | 捨て牌フリテンは厳密。temporary/riichi furitenはPASS censorを正しく扱う必要 |
| 各合法打牌への即時ロン合法性 | E | 非常に高 | 高/中〜高 | 非常に高 | 同じhidden stateで候補打牌だけ差し替えるone-step exact counterfactual |
| 即時ロン点数 | E | 非常に高 | 高/中 | 非常に高 | score engineのrule coverage範囲。複数ロン/責任払い等を検証 |
| 実際の放銃 | D | 高 | 低/低 | 終局未来 | terminal event+delta |
| 将来の放銃 | D/S | 高 | 低〜非常に高/中 | 高 | factual chosenとsimulated unchosenを別列にする |
| 他家ツモ/横移動 | D | 高 | 低/低 | 終局未来 | winner/fromから分類 |

### 3.4 他家推定

| 教師 | 取得 | 有用性 | 計算/容量 | 漏洩 | 注意 |
|---|---|---|---|---|---|
| 手牌構成/変化 | D+E | 高 | 中/高 | 非常に高 | 物理ID replay。oracle専用 |
| シャンテン | E | 高 | 低/低 | 非常に高 | oracle専用 |
| 次打牌 | D | 高 | 低/低 | 未来 | 実軌跡の次行動だけ |
| 鳴き/リーチ選択 | D | 高 | 低/低 | 未来 | 実選択だけ。無選択窓はcensor判定 |
| 公開行動履歴 | D | 必須 | 低/中 | なし | inputとして使用可能 |
| 未選択action後の他家行動 | S | 探索用 | 非常に高/高 | simulation由来 | 実牌譜には存在しない |

### 3.5 価値

| 教師 | 取得 | 有用性 | 計算/容量 | 漏洩 | 注意 |
|---|---|---|---|---|---|
| 4人局収支 | D | 非常に高 | 低/低 | 終局未来 | `sc` delta。供託で単純sum=0でない場合を許容 |
| 局終了状態 | D+E | 高 | 中/中 | 終局未来 | terminal + replay state |
| 半荘最終点/順位 | D | 非常に高 | 低/低 | 半荘未来 | `owari`; 同点規則をrule profile化 |
| 段位戦reward | E+外部spec | 目的関数 | 低/低 | 半荘未来 | 段位/R/卓/長さ/順位はある。時点別配点表をversion管理 |
| 正確なR変動 | Xの場合あり | 補助 | — | — | 対局数補正等に必要な履歴がraw logにない |
| 未選択actionの半荘結果 | S | offline Q用 | 非常に高/中〜高 | simulation由来 | 人間の実結果ではない |

### 3.6 生成できない教師

- playerが内心検討した未選択行動、クリックしたがpriorityで無効になった応答。
- 未選択行動を選んだ世界で人間全員が実際に取ったであろう将来行動。
- raw logだけからの正確なネットワーク遅延、思考時間、UI表示タイミング。
- seed algorithmを実装・検証しない状態での未使用山/王牌順。
- 対局前の段位pt残高、完全なRating更新履歴。

---

## 4. 推奨データ基盤

### 4.1 レイヤ

```text
L0 Raw immutable
  SQLite/BLOB original + SHA-256 + rights/acquisition manifest
L1 Canonical match/round/event truth
  typed physical tiles, raw payload, interpreted rules, players (restricted)
L2 Replay checkpoints
  deterministic engine state checkpoints + event offsets
L3 Decision/opportunity
  observer, observable prefix, response-window status, chosen action
L4 LegalAction
  unique versioned ActionId, action axis, exact immediate labels
L5 Teacher families
  factual / oracle-hidden / exact-counterfactual / simulated-counterfactual
L6 Feature views
  model-specific encoders generated lazily or cached by version
```

L0/L1 truthをモデル都合で変更しない。特徴量planeはL6派生物で、正本にしない。

### 4.2 推奨schema（要点）

`matches`:

```text
match_id, source_object_sha256, source_log_id_hash, started_at,
platform, raw_format, raw_schema_version,
rule_raw_type, rule_profile_id, player_count, match_length,
rights_record_id, converter_version, quality_flags,
player_pseudonyms[4], dan_codes[4], ratings[4], final_scores[4], final_ranks[4]
```

`rounds`:

```text
match_id, round_id, round_index, wind, hand_number, honba, kyotaku,
dealer, start_scores[4], end_scores[4], deltas[4], end_reason,
shuffle_seed_ref, raw_event_start/end, validation_status
```

`events`（1 row/event、typed union payload）:

```text
match_id, round_id, seq, event_type, actor, target,
tile_physical_id, tile_type_id, red,
meld_id, called_from, consume_physical_ids,
riichi_step, score_deltas, result_payload,
visibility_mask, raw_tag, raw_payload
```

`decisions`:

```text
decision_id, match_id, round_id, seq_before, observer,
decision_kind, phase, opportunity_status,
chosen_action_id?, chosen_observed, terminal_censor,
observation_checkpoint_id, public_prefix_end
```

`legal_actions`:

```text
decision_id, action_index, action_id, action_schema_version,
action_type, primary_tile_type/red, from_relative,
consume_multiset, chosen,
immediate_ron_legal[3], immediate_ron_points[3], exact_mask[3]
```

teacher tables:

- `teacher_factual`: 実軌跡の結果。chosen actionだけ。
- `teacher_oracle_state`: hidden hands/shanten/waits/furiten/live wall。モデル入力権限から分離。
- `teacher_exact_cf`: one-step legality/score/ukeire等、数学的に厳密な未選択候補情報。
- `teacher_simulated_cf`: simulator version、rollout policy、seed、sample count、mean/distribution/SEを必須化。

提案する加工後1 decisionの論理例（実値と未観測を混同しない）:

```json
{
  "decision_id": "d:v1:...",
  "match_id": "m:sha256:...",
  "round_id": 0,
  "seq_before": 2,
  "observer": 0,
  "decision_kind": "turn",
  "opportunity_status": "selected",
  "observation_ref": {"checkpoint_id":"cp:...","public_prefix_end":1},
  "legal_actions": [
    {
      "action_index": 0,
      "action_id": "a:v1:discard:s9:normal:hand",
      "chosen": true,
      "exact": {"shanten_after":3,"ukeire_types":["..."]},
      "factual": {"round_delta":0,"round_result":"horizontal","mask":true},
      "simulated": null
    },
    {
      "action_index": 1,
      "action_id": "a:v1:discard:m1:normal:hand",
      "chosen": false,
      "exact": {"shanten_after":4,"immediate_ron_legal":[false,false,false]},
      "factual": {"mask":false},
      "simulated": {"mask":false}
    }
  ],
  "oracle_ref": "restricted/oracle/part-...parquet#row=..."
}
```

ここで未選択`m1`の実局収支を0と書かない。即時ロン合法性やシャンテンはexact、将来結果はsimulationを実行した場合だけ別provenanceで埋める。

### 4.3 ActionId

JSON文字列をIDとせず、`ActionSchema/v1`のcanonical tupleをhashする。

- discard: `(type, tile_type, red, tsumogiri)`
- riichi: `(type, discard_type, red, tsumogiri)`
- chi/pon: `(type, called_type/red, from_relative, sorted consume(type/red multiset))`
- kan: kind別に同様。shouminkanは元ponのsemantic meld IDを使う。
- ron: `(type, target_relative)`
- tsumo/pass/abortive: type+reason

同種非赤の物理copy差はactionとしてcollapseし、赤/非赤の選択で公開結果が変わる場合は区別する。`action_id`一意性を全decisionでassertする。

### 4.4 観測とoracleの分離

- observable table/fileには、観測者の手牌、自摸、公開情報、ruleだけ。
- hidden/oracle fileは別path、別Arrow schema、別loader権限。
- feature cache keyは`observation prefix hash + encoder version`。
- hidden truthを変更してもobservable serialization/featuresが不変、というmetamorphic testを必須にする。

### 4.5 JSON / JSONL / Parquet

| 形式 | 容量 | 変換 | 学習read | schema変更 | 推奨用途 |
|---|---|---|---|---|---|
| JSON（1対局/1大object） | 大 | 簡単 | 遅い、random access弱い | 柔軟だが型弱い | 小規模debug/交換 |
| JSONL.zst | 中 | streaming容易 | 中、列選択不可 | 柔軟 | canonical event監査export |
| Parquet+ZSTD | 小〜中 | Arrow schema設計が必要 | 高速、列pruning/filter | versioned schema必須 | 正式event/decision/teacher store |

rawは原BLOBを不変保存、canonical以降はParquetを正本、JSONL.zstを人間可読exportにする。

---

## 5. 品質管理

### 5.1 raw/canonical

- SQLite/file hash、BLOB hash、bzip2、XML parse、未知tag/attributeを全件検査。
- `log_id`/content/shuffle seed/event canonical hashの複数段duplicate検査。
- 物理牌ID0..135の一意性、配牌+山=136、同一物理牌の二重出現を検査。
- 各局で手牌枚数、draw/discard交互、鳴き消費、槓/嶺上/ドラ枚数を検査。
- `sc` delta、REACH供託、本場、終局点、次局開始点、`owari`を照合。
- source `GO.type`とDB flags、engine RuleConfigを照合。unknown bitは捨てない。

### 5.2 decisions/PASS/censor

- 各decisionでActionId unique、chosenがlegalに一意一致。
- 応答窓ごとに`offered_definite / passed_definite / selected / censored / not_legal`を持つ。
- 次が通常drawの窓では、非選択legal response全員に確定PASSを要求。
- call/winで閉じた窓はpriority proofがない限りPASSを禁止。
- ron見送り後のtemporary/riichi furiten遷移を別test。
- PASS率、call opportunity率、選択率をaction type/巡目/段位/rule別に監視。

### 5.3 未来情報

- observation tableに`future`, `hidden`, `final`, `wall`, `opponent_hand`列をschemaレベルで禁止。
- decision時点以降のeventを切ってもobservable encodingが同一。
- hidden hands/wallを合法な別worldへ置換してもobservable encodingが同一。
- teacher joinは`decision_id`だけ、training loaderは明示allowlist列だけ読む。
- factual/counterfactual maskの重複をreject。

### 5.4 split

- split keyは必ず`match_id`。局/decision単位hashは禁止。
- raw/canonical/event/teacherの全派生物が同じsplit assignmentを継承。
- SHA/signature duplicate clusterは同じsplitへまとめる。
- 最終評価は時期holdout（例: 2020 Q4）も併設し、distribution shiftを測る。
- player-generalization評価はpseudonym clusterを跨がせないsplitを別に作る。ただし4人対局はgraph partitionとなるため、除外率を報告する。
- train/validation/final-testでmatch/player/date/duplicate intersection=0を自動assert。

### 5.5 偏り

- 七/八/九/十段/天鳳位、R帯、四麻東風/半荘、速卓、月、player exposureで分布を出す。
- player上位1/10/100のdecision shareを監視し、必要ならmatch weightをcap。
- action typeだけでなく「合法機会数」を分母にしてPASS/鳴き/リーチ/和了率を報告。
- 三麻を四麻に混ぜず、別schema/rule engineが完成するまでquarantine。

---

## 6. 容量・時間見積もり

### 6.1 実測

| 処理 | 実測 |
|---|---:|
| raw DB | 2.21 GB (decimal) |
| 解凍XML | 4.70 GB |
| decision/v1 200k | 46.0 MB = 約230 B/decision |
| sequence-v2 53,459 | 20.7 MB = 約387 B/decision |
| decision 100k変換 1 worker | 6,291〜6,367秒 |
| sequence-v2 4 worker計53,459 | wall最大2,424秒 |

### 6.2 全量外挿（推定）

既存sequence sampleは約60.9 decisions/round。全3,196,694局へ単純外挿すると約1.95億decision。ただし三麻除外、局面抽出定義、PASS窓、エラーで変わる。

| 成果物 | planning range | 根拠 |
|---|---:|---|
| L0 raw immutable | 2.3〜5 GB | DB原本+manifest/hash。解凍copyを常設しない |
| typed canonical events Parquet | 5〜15 GB | 物理ID、raw payload、metadataを含む。未benchmark |
| observable decision/action tables | 30〜80 GB | 1.5〜2.0億 rows、state JSON重複を避ける |
| oracle/exact teacher | 30〜120 GB | label familyを分離し必要列だけ生成 |
| rollout counterfactual | 生成条件次第でTB級 | 全action×複数seedを常設しない。集約分布のみ推奨 |

現行Pythonの実測速度をそのまま全量へ外挿すると、decision/v1で約2か月、sequence系で約3か月のwall timeになり得る。これは**現行実装の外挿**であり、Rust一括replay/Arrow batch化後の実測ではない。最初に1,000/10,000対局benchmarkを行い、全量ETAを更新する。

---

## 7. 移行方法

既存DB/Parquetは変更しない。

1. DB snapshotのSHA-256、SQLite integrity、row/hash/期間をmanifest化。
2. rights registryを作り、raw accessと派生物利用条件を固定。
3. 四麻だけの小さなgolden cohort（通常和了、複数ロン、全流局種、全槓、赤絡み、副露priority、切断）を選ぶ。
4. `GO/UN/SHUFFLE/BYE`を落とさないcanonical decoderを新規versionで作る。
5. Rust replayへphysical tile/full-truth inputを追加し、4 observer projectionと一致検証。
6. ActionSchema/v1とopportunity/censor schemaを凍結。
7. factual/oracle/exact/simulated teacherを別tableとして実装。
8. golden cohortで旧converter/新converter/元XMLをevent-by-event比較。
9. 1,000対局benchmark、容量/時間/エラー率を測定。
10. 期間・playerを考慮したmatch-level split manifestを先に固定。
11. shard単位で段階変換。各shardにsource range、row count、hash、reject理由を記録。
12. 旧artifactは`legacy`として読み取り専用保持し、新datasetとの混載をloaderで拒否。

## 8. 実装・検証順序

1. **権利/原本/ルールinventory gate**: source、許諾、三麻、GO bit、PIIを確定。
2. **canonical event gate**: raw XML情報を無損失で保持しround-trip監査。
3. **physical tile/replay gate**: 136枚保存、全員手牌、点数、終局一致。
4. **ActionId gate**: 全decision unique、chosen exactly-one、赤/非赤fixture。
5. **opportunity gate**: PASS確定とcensorをpriority fixtureで検証。
6. **leakage gate**: observable/oracle物理分離、metamorphic test。
7. **teacher gate**: factual/exact/simulated provenanceとmask。
8. **split gate**: match/date/player/duplicate leakage=0。
9. **1k/10k benchmark**: 容量、throughput、reject taxonomy。
10. **全量変換**: shard再開可能、immutable、hash検証。
11. **学習はその後**: dataset cardとsealed final testを固定してから開始。

---

## 9. 根拠となる重要コード上の注意

- [tenhou.py](../python/mahjong_ai/data/tenhou.py): `SHUFFLE/GO/UN/BYE`をEventLogへ渡さず、raw IDをSHA-256化し、物理IDをkind/redへ縮約する。
- [dataset.py](../python/mahjong_ai/data/dataset.py): 次の通常drawから安全なPASSを補い、expert actionがRust合法手にない場合はrejectする。
- [legal.rs](../engine/core/src/legal.rs): 現行はlegal actionsをsort+dedupする。既存Parquet生成時の重複は残る。
- [supervised.py](../python/mahjong_ai/training/supervised.py): 新3-way splitは半荘groupだが、旧`split_by_log`はround ID単位。
- [sequence_dataset.py](../python/mahjong_ai/data/sequence_dataset.py): hidden labelsとfuture drawは教師だけに使用し、未選択counterfactual maskをfalseにする。
- [encoder.py](../python/mahjong_ai/ai/encoder.py): opponent hand/wall/private fieldを通常入力として拒否する。

この報告は元データにない将来結果を補完していない。山完全復元、semantic duplicate全件、全対局のRust semantic replay、正確な権利範囲は、それぞれ明記した追加検証が完了するまで未確定とする。
