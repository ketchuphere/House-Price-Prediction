import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    hmr: { overlay: false },
    proxy: {
      // During development, proxy API calls to the FastAPI backend at :8000.
      // The frontend calls relative /predict — Vite forwards it transparently.
      "/predict": {
        target: "http://localhost:8000",
        changeOrigin: true,
        secure: false,
      },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
      "/api":    { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
    dedupe: [
      "react", "react-dom",
      "react/jsx-runtime", "react/jsx-dev-runtime",
      "@tanstack/react-query", "@tanstack/query-core",
    ],
  },
}));
