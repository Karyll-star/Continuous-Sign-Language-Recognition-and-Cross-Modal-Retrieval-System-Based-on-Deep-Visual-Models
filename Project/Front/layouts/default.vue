<template>
  <div class="min-h-screen flex flex-col">
    <!-- 主内容区（全屏，无头部/底部导航） -->
    <main class="flex-1">
      <slot />
    </main>

    <!-- Toast 提示 -->
    <Toast />
  </div>
</template>

<script setup lang="ts">
// 默认布局 - 仅保留 Toast，无 Header / Footer
</script>
