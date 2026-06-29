<template>
  <div class="min-h-screen bg-gray-50 flex">
    <!-- 侧边栏（组件待实现，当前 admin 页面使用 layout: false 内联了侧边栏） -->
    <!-- <AdminSidebar /> -->

    <!-- 主内容区 -->
    <div class="flex-1 ml-64">
      <!-- 顶部栏（组件待实现） -->
      <!-- <AdminTopbar /> -->

      <!-- 页面内容 -->
      <div class="p-6">
        <slot />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// 管理后台布局
</script>

