import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The standalone build is served by the FastAPI process on a single origin, so
// the client always uses origin-relative API paths. These dev proxies make the
// same relative paths work while Vite serves the UI on its own port.
const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/artifacts": { target: BACKEND, changeOrigin: true },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/artifacts": { target: BACKEND, changeOrigin: true },
    },
  },
});
