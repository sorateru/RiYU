import type {
  ReplayAnalysis,
  TurnAnalysis,
} from "../types/analysis"

const east1Turns: TurnAnalysis[] = [
  {
    turn: 1,

    handTiles: [
      { tile: "1m", recommendation: 2 },
      { tile: "2m", recommendation: 3 },
      { tile: "3m", recommendation: 4 },
      { tile: "5m", recommendation: 48 },
      { tile: "6m", recommendation: 8 },
      { tile: "7m", recommendation: 5 },
      { tile: "3p", recommendation: 4 },
      { tile: "4p", recommendation: 5 },
      { tile: "5p", recommendation: 6 },
      { tile: "6s", recommendation: 3 },
      { tile: "7s", recommendation: 4 },
      { tile: "8s", recommendation: 4 },
      { tile: "東", recommendation: 5 },
      {
        tile: "9p",
        recommendation: 3,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 48,
        shanten: 2,
        ukeire: 31,
        dealInRisk: 0.8,
        expectedValue: 410,
      },
      {
        tile: "東",
        score: 27,
        shanten: 2,
        ukeire: 29,
        dealInRisk: 0.4,
        expectedValue: 360,
      },
      {
        tile: "9p",
        score: 14,
        shanten: 2,
        ukeire: 26,
        dealInRisk: 0.6,
        expectedValue: 290,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: [],
      right: [],
      opposite: [],
      left: [],
    },

    remainingTiles: 69,

    dora: "4p",
  },

  {
    turn: 2,

    handTiles: [
      { tile: "1m", recommendation: 2 },
      { tile: "2m", recommendation: 2 },
      { tile: "3m", recommendation: 3 },
      { tile: "5m", recommendation: 54 },
      { tile: "6m", recommendation: 7 },
      { tile: "7m", recommendation: 5 },
      { tile: "3p", recommendation: 3 },
      { tile: "4p", recommendation: 4 },
      { tile: "5p", recommendation: 5 },
      { tile: "6s", recommendation: 3 },
      { tile: "7s", recommendation: 4 },
      { tile: "8s", recommendation: 3 },
      { tile: "東", recommendation: 2 },
      {
        tile: "8p",
        recommendation: 3,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 54,
        shanten: 2,
        ukeire: 32,
        dealInRisk: 0.9,
        expectedValue: 450,
      },
      {
        tile: "8p",
        score: 21,
        shanten: 2,
        ukeire: 28,
        dealInRisk: 0.7,
        expectedValue: 370,
      },
      {
        tile: "東",
        score: 13,
        shanten: 2,
        ukeire: 27,
        dealInRisk: 0.4,
        expectedValue: 320,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: ["東"],
      right: ["9s"],
      opposite: ["1m"],
      left: ["2p"],
    },

    remainingTiles: 65,

    dora: "4p",
  },

  {
    turn: 3,

    handTiles: [
      { tile: "1m", recommendation: 2 },
      { tile: "2m", recommendation: 2 },
      { tile: "3m", recommendation: 2 },
      { tile: "5m", recommendation: 59 },
      { tile: "6m", recommendation: 6 },
      { tile: "7m", recommendation: 4 },
      { tile: "3p", recommendation: 4 },
      { tile: "4p", recommendation: 4 },
      { tile: "5p", recommendation: 5 },
      { tile: "6s", recommendation: 3 },
      { tile: "7s", recommendation: 3 },
      { tile: "8s", recommendation: 2 },
      { tile: "8p", recommendation: 2 },
      {
        tile: "9m",
        recommendation: 2,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 59,
        shanten: 1,
        ukeire: 30,
        dealInRisk: 1.2,
        expectedValue: 560,
      },
      {
        tile: "9m",
        score: 18,
        shanten: 1,
        ukeire: 27,
        dealInRisk: 0.8,
        expectedValue: 420,
      },
      {
        tile: "8p",
        score: 12,
        shanten: 1,
        ukeire: 25,
        dealInRisk: 0.9,
        expectedValue: 380,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: ["東", "9p"],
      right: ["9s", "中"],
      opposite: ["1m", "9m"],
      left: ["2p", "白"],
    },

    remainingTiles: 61,

    dora: "4p",
  },

  {
    turn: 4,

    handTiles: [
      { tile: "1m", recommendation: 2 },
      { tile: "2m", recommendation: 2 },
      { tile: "3m", recommendation: 2 },
      { tile: "5m", recommendation: 63 },
      { tile: "6m", recommendation: 5 },
      { tile: "7m", recommendation: 4 },
      { tile: "3p", recommendation: 3 },
      { tile: "4p", recommendation: 4 },
      { tile: "5p", recommendation: 4 },
      { tile: "6s", recommendation: 3 },
      { tile: "7s", recommendation: 2 },
      { tile: "8s", recommendation: 2 },
      { tile: "8p", recommendation: 2 },
      {
        tile: "1s",
        recommendation: 2,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 63,
        shanten: 1,
        ukeire: 29,
        dealInRisk: 1.4,
        expectedValue: 630,
      },
      {
        tile: "1s",
        score: 17,
        shanten: 1,
        ukeire: 27,
        dealInRisk: 0.7,
        expectedValue: 450,
      },
      {
        tile: "8p",
        score: 10,
        shanten: 1,
        ukeire: 24,
        dealInRisk: 1.1,
        expectedValue: 390,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: ["東", "9p", "9m"],
      right: ["9s", "中", "3m"],
      opposite: ["1m", "9m", "3p"],
      left: ["2p", "白", "6m"],
    },

    remainingTiles: 57,

    dora: "4p",
  },

  {
    turn: 5,

    handTiles: [
      { tile: "1m", recommendation: 2 },
      { tile: "2m", recommendation: 2 },
      { tile: "3m", recommendation: 2 },
      { tile: "5m", recommendation: 66 },
      { tile: "6m", recommendation: 4 },
      { tile: "7m", recommendation: 3 },
      { tile: "3p", recommendation: 3 },
      { tile: "4p", recommendation: 3 },
      { tile: "5p", recommendation: 4 },
      { tile: "6s", recommendation: 2 },
      { tile: "7s", recommendation: 2 },
      { tile: "8s", recommendation: 2 },
      { tile: "8p", recommendation: 2 },
      {
        tile: "7p",
        recommendation: 3,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 66,
        shanten: 1,
        ukeire: 29,
        dealInRisk: 1.8,
        expectedValue: 680,
      },
      {
        tile: "7p",
        score: 15,
        shanten: 1,
        ukeire: 27,
        dealInRisk: 1.1,
        expectedValue: 490,
      },
      {
        tile: "8p",
        score: 9,
        shanten: 1,
        ukeire: 25,
        dealInRisk: 1.6,
        expectedValue: 430,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: ["東", "9p", "9m", "1s"],
      right: ["9s", "中", "3m", "2s"],
      opposite: ["1m", "9m", "3p", "東"],
      left: ["2p", "白", "6m", "8s"],
    },

    remainingTiles: 53,

    dora: "4p",
  },

  {
    turn: 6,

    handTiles: [
      { tile: "1m", recommendation: 2 },
      { tile: "2m", recommendation: 2 },
      { tile: "3m", recommendation: 2 },
      { tile: "5m", recommendation: 69 },
      { tile: "6m", recommendation: 4 },
      { tile: "7m", recommendation: 3 },
      { tile: "3p", recommendation: 2 },
      { tile: "4p", recommendation: 3 },
      { tile: "5p", recommendation: 3 },
      { tile: "6s", recommendation: 2 },
      { tile: "7s", recommendation: 2 },
      { tile: "8s", recommendation: 2 },
      { tile: "7p", recommendation: 2 },
      {
        tile: "6p",
        recommendation: 2,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 69,
        shanten: 1,
        ukeire: 28,
        dealInRisk: 2.1,
        expectedValue: 730,
      },
      {
        tile: "6p",
        score: 14,
        shanten: 1,
        ukeire: 25,
        dealInRisk: 1.5,
        expectedValue: 480,
      },
      {
        tile: "7p",
        score: 8,
        shanten: 1,
        ukeire: 24,
        dealInRisk: 1.7,
        expectedValue: 430,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: ["東", "9p", "9m", "1s", "8p"],
      right: ["9s", "中", "3m", "2s", "4m"],
      opposite: ["1m", "9m", "3p", "東", "7s"],
      left: ["2p", "白", "6m", "8s", "西"],
    },

    remainingTiles: 49,

    dora: "4p",
  },

  {
    turn: 7,

    handTiles: [
      { tile: "1m", recommendation: 2 },
      { tile: "2m", recommendation: 2 },
      { tile: "3m", recommendation: 2 },
      { tile: "5m", recommendation: 70 },
      { tile: "6m", recommendation: 3 },
      { tile: "7m", recommendation: 3 },
      { tile: "3p", recommendation: 2 },
      { tile: "4p", recommendation: 3 },
      { tile: "5p", recommendation: 3 },
      { tile: "6p", recommendation: 2 },
      { tile: "6s", recommendation: 2 },
      { tile: "7s", recommendation: 2 },
      { tile: "8s", recommendation: 2 },
      {
        tile: "5s",
        recommendation: 2,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 70,
        shanten: 1,
        ukeire: 28,
        dealInRisk: 2.6,
        expectedValue: 770,
      },
      {
        tile: "5s",
        score: 13,
        shanten: 1,
        ukeire: 25,
        dealInRisk: 1.8,
        expectedValue: 500,
      },
      {
        tile: "6p",
        score: 8,
        shanten: 1,
        ukeire: 23,
        dealInRisk: 2.2,
        expectedValue: 450,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: ["東", "9p", "9m", "1s", "8p", "7p"],
      right: ["9s", "中", "3m", "2s", "4m", "1p"],
      opposite: ["1m", "9m", "3p", "東", "7s", "2m"],
      left: ["2p", "白", "6m", "8s", "西", "9p"],
    },

    remainingTiles: 45,

    dora: "4p",
  },

  {
    turn: 8,

    handTiles: [
      { tile: "1m", recommendation: 4 },
      { tile: "2m", recommendation: 3 },
      { tile: "3m", recommendation: 2 },
      { tile: "5m", recommendation: 72 },
      { tile: "5m", recommendation: 4 },
      { tile: "6m", recommendation: 3 },
      { tile: "7m", recommendation: 2 },
      { tile: "3p", recommendation: 1 },
      { tile: "4p", recommendation: 1 },
      { tile: "5p", recommendation: 2 },
      { tile: "6s", recommendation: 1 },
      { tile: "7s", recommendation: 1 },
      { tile: "8s", recommendation: 1 },
      {
        tile: "9p",
        recommendation: 3,
        drawn: true,
      },
    ],

    candidates: [
      {
        tile: "5m",
        score: 72,
        shanten: 1,
        ukeire: 28,
        dealInRisk: 3.2,
        expectedValue: 820,
      },
      {
        tile: "8p",
        score: 18,
        shanten: 1,
        ukeire: 24,
        dealInRisk: 7.8,
        expectedValue: 510,
      },
      {
        tile: "1s",
        score: 7,
        shanten: 2,
        ukeire: 41,
        dealInRisk: 1.1,
        expectedValue: 120,
      },
    ],

    scores: {
      self: 25000,
      right: 25000,
      opposite: 25000,
      left: 25000,
    },

    discards: {
      self: ["東", "9p", "9m", "1s", "8p", "7p", "5s"],
      right: ["9s", "中", "3m", "2s", "4m", "1p", "南"],
      opposite: ["1m", "9m", "3p", "東", "7s", "2m", "白"],
      left: ["2p", "白", "6m", "8s", "西", "9p", "3s"],
    },

    remainingTiles: 41,

    dora: "4p",
  },
]

export const mockAnalysis: ReplayAnalysis = {
  rounds: [
    {
      id: "east-1",
      label: "東1局",
      turns: east1Turns,
    },

    {
      id: "east-2",
      label: "東2局",

      turns: [
        {
          ...east1Turns[0],

          turn: 1,

          scores: {
            self: 26800,
            right: 23200,
            opposite: 25000,
            left: 25000,
          },

          dora: "7s",

          candidates: [
            {
              tile: "東",
              score: 61,
              shanten: 2,
              ukeire: 30,
              dealInRisk: 0.4,
              expectedValue: 380,
            },
            {
              tile: "9p",
              score: 23,
              shanten: 2,
              ukeire: 28,
              dealInRisk: 0.5,
              expectedValue: 330,
            },
            {
              tile: "1m",
              score: 10,
              shanten: 2,
              ukeire: 25,
              dealInRisk: 0.6,
              expectedValue: 280,
            },
          ],

          handTiles: [
            { tile: "1m", recommendation: 10 },
            { tile: "2m", recommendation: 2 },
            { tile: "3m", recommendation: 2 },
            { tile: "4m", recommendation: 2 },
            { tile: "5m", recommendation: 2 },
            { tile: "6m", recommendation: 2 },
            { tile: "2p", recommendation: 2 },
            { tile: "3p", recommendation: 2 },
            { tile: "4p", recommendation: 2 },
            { tile: "5s", recommendation: 2 },
            { tile: "6s", recommendation: 2 },
            { tile: "7s", recommendation: 2 },
            { tile: "東", recommendation: 61 },
            {
              tile: "9p",
              recommendation: 23,
              drawn: true,
            },
          ],
        },

        {
          ...east1Turns[1],

          turn: 2,

          scores: {
            self: 26800,
            right: 23200,
            opposite: 25000,
            left: 25000,
          },

          dora: "7s",
        },

        {
          ...east1Turns[2],

          turn: 3,

          scores: {
            self: 26800,
            right: 23200,
            opposite: 25000,
            left: 25000,
          },

          dora: "7s",
        },

        {
          ...east1Turns[3],

          turn: 4,

          scores: {
            self: 26800,
            right: 23200,
            opposite: 25000,
            left: 25000,
          },

          dora: "7s",
        },

        {
          ...east1Turns[4],

          turn: 5,

          scores: {
            self: 26800,
            right: 23200,
            opposite: 25000,
            left: 25000,
          },

          dora: "7s",
        },
      ],
    },

    {
      id: "east-3",
      label: "東3局",

      turns: [
        {
          ...east1Turns[0],

          turn: 1,

          scores: {
            self: 25100,
            right: 22900,
            opposite: 27500,
            left: 24500,
          },

          dora: "3m",
        },

        {
          ...east1Turns[1],

          turn: 2,

          scores: {
            self: 25100,
            right: 22900,
            opposite: 27500,
            left: 24500,
          },

          dora: "3m",
        },

        {
          ...east1Turns[2],

          turn: 3,

          scores: {
            self: 25100,
            right: 22900,
            opposite: 27500,
            left: 24500,
          },

          dora: "3m",
        },
      ],
    },
  ],
}