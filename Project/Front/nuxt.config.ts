// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },

  modules: [
    '@nuxtjs/tailwindcss',
    '@pinia/nuxt',
    '@vueuse/nuxt',
  ],

  css: ['~/assets/css/main.css'],

  app: {
    head: {
      title: '译手 HandTalk AI - 实时手语翻译',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: '译手 (HandTalk AI) - 中国手语实时翻译应用，让沟通无障碍' },
        { name: 'theme-color', content: '#1A1A1A' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap' },
        { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css' },
      ],
    },
  },

  runtimeConfig: {
    public: {
      // 默认指向手语识别 FastAPI 服务端口（与 README 中 uvicorn --port 8000 一致）
      apiBase: process.env.API_BASE_URL || 'http://localhost:8000/api/v1',
      // 上传图片 / 视频识别接口单独配置基础地址，避免重复拼接 /api/v1 前缀导致 404
      uploadBase: process.env.UPLOAD_BASE_URL || 'http://localhost:8000',
      wsUrl: process.env.WS_URL || 'ws://localhost:8000',
    },
  },

  typescript: {
    strict: true,
    // typeCheck 关闭：当前 vue-tsc 3.x 与 Vue 3.5 + TS 5.9 DOM 类型定义存在深层不兼容
    typeCheck: false,
  },

  compatibilityDate: '2025-01-15',
})