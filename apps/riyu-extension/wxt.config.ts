import { defineConfig } from "wxt"

export default defineConfig({
  srcDir: "src",
  manifest: {
    name: "RiYU 雀魂牌譜インポート",
    description: "開いている雀魂牌譜をローカルRiYUへ送信します。",
    version: "0.1.0",
    permissions: ["storage", "activeTab"],
    host_permissions: [
      "https://game.mahjongsoul.com/*",
      "https://mahjongsoul.game.yo-star.com/*",
      "https://game.maj-soul.net/*",
      "https://game.maj-soul.com/*",
      "http://127.0.0.1/*",
      "http://localhost/*"
    ]
  }
})
