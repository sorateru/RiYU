import type { DecodedRecord } from "./record-decode"

export function buildDecodedReplay(record: DecodedRecord): Record<string, unknown> {
  const finalScores = [0, 0, 0, 0]
  for (const player of record.head?.result?.players ?? []) {
    if (Number.isInteger(player.seat) && player.seat >= 0 && player.seat < 4) {
      finalScores[player.seat] = Number(player.part_point_1)
    }
  }
  return {
    format: "mahjong-soul-decoded-replay/v1",
    records: record.actions.map((action) => ({
      name: action.name,
      data: normalizeActionData(action.name, action.data),
    })),
    final_scores: finalScores,
  }
}

export function normalizeActionData(name: string, source: any): Record<string, unknown> {
  const data = JSON.parse(JSON.stringify(source)) as Record<string, any>
  const copyNumber = (key: string) => {
    if (Number.isInteger(source?.[key])) data[key] = Number(source[key])
  }

  // protobuf.js stores scalar zero defaults on prototypes. JSON.stringify
  // omits them, so copy fields whose zero value carries game meaning.
  if (name === "RecordNewRound") {
    for (const key of ["chang", "ju", "ben", "liqibang"]) copyNumber(key)
  }
  if (
    [
      "RecordDealTile",
      "RecordDiscardTile",
      "RecordChiPengGang",
      "RecordAnGangAddGang",
    ].includes(name)
  ) {
    copyNumber("seat")
  }
  if (["RecordChiPengGang", "RecordAnGangAddGang", "RecordLiuJu"].includes(name)) {
    copyNumber("type")
  }
  if (name === "RecordHule" && Array.isArray(source?.hules)) {
    data.hules = source.hules.map((hule: any, index: number) => {
      const plain = data.hules?.[index] ?? {}
      if (Number.isInteger(hule?.seat)) plain.seat = Number(hule.seat)
      plain.zimo = Boolean(hule?.zimo)
      return plain
    })
  }
  return data
}
