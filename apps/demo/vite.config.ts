import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig, loadEnv } from "vite"

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "")
  const proxyTarget = env.VITE_COORDINATOR_PROXY_TARGET?.trim()
  const proxyApiKey =
    env.COORDINATOR_API_KEY?.trim() || env.VITE_COORDINATOR_API_KEY?.trim()

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: proxyTarget
      ? {
        proxy: {
          "/api": {
            target: proxyTarget,
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api/, ""),
            headers: proxyApiKey
              ? {
                  Authorization: `Bearer ${proxyApiKey}`,
                }
              : undefined,
          },
        },
      }
      : undefined,
  }
})
