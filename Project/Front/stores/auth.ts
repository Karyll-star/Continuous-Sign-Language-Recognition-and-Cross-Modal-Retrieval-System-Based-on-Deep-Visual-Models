// 已废弃 - 应用仅保留实时手语翻译页面
import { defineStore } from 'pinia'

export const useAuthStore = defineStore('auth', () => {
  const isAuthenticated = ref(false)
  return { isAuthenticated }
})
