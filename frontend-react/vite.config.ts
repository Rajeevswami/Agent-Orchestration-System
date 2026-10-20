import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app is served by FastAPI under /react (see backend/app/api/app.py).
export default defineConfig({
  plugins: [react()],
  base: "/react/",
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
