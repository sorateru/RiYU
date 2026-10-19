# Feature Encoder v1

`observable-planes-v1` は観測可能 `GameState` だけを `float32[49, 34]` へ変換する。34列は `m1..m9, p1..p9, s1..s9, east, south, west, north, white, green, red` の固定順である。

主なplaneは、自家手牌4枚one-hot、赤牌、ツモ牌、4人の河・副露・リーチ・一発、ドラ表示牌、既知枚数と推定残存枚数、親・自家・場風、4人の点数、巡目、残り牌、本場、供託、行動phaseである。赤い公開牌は別planeに保持する。

合法手は固定IDへ潰さず、各局面の `Vec<Action>` を `float32[N, 83]` にする。行動種別、対象牌、元プレイヤー、使用牌、赤、ツモ切りを符号化し、ネットワークはこのN候補だけを採点する。

完全情報を示すフラグ、`private_hands`、`wall` が入力にあればEncoderは例外にする。チェックポイントは `encoder_version` を必須メタデータとして保存し、異なる版の推論を拒否する。
