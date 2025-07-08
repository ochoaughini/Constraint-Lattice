import { defineConfig, type PluginOption } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import tsconfigPaths from 'vite-tsconfig-paths'

// https://vite.dev/config/
export default defineConfig({
  root: __dirname,
  base: '/',
  plugins: [react(), tsconfigPaths()] as PluginOption[],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 5174,
    strictPort: true,
    open: true
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  }
})
