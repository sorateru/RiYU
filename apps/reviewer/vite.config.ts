import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      "/health": "http://127.0.0.1:8787",
      "/analyze": "http://127.0.0.1:8787",
      "/realtime": "http://127.0.0.1:8787",
      "/local-play": "http://127.0.0.1:8787",
      "/ws": { target: "ws://127.0.0.1:8787", ws: true },
    },
  },
});
