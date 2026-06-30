<template>
  <div class="relative min-h-screen overflow-hidden bg-stone-50">

    <!-- 顶部导航栏 -->
    <div class="sticky top-0 z-30 border-b border-white/60 bg-white/60 backdrop-blur-xl">
      <div class="max-w-6xl mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center gap-3">
            <NuxtLink
              to="/"
              class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-700 hover:bg-emerald-500/20 transition-colors text-sm font-medium"
            >
              <i class="bi bi-camera-video text-base"></i>
              <span class="hidden sm:inline">实时翻译</span>
            </NuxtLink>
            <div class="h-6 w-px bg-slate-200/60"></div>
            <span class="text-sm font-medium tracking-wide text-slate-500">
              视频手语翻译
            </span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-[11px] text-slate-400 hidden sm:inline">上传 MP4 / MOV · 3-15s</span>
          </div>
        </div>
      </div>
    </div>

    <div class="relative max-w-6xl mx-auto px-4 py-10">
      <!-- 顶部标题 -->
      <div class="mb-12 animate-fade-up md:flex md:items-end md:justify-between">
        <div class="max-w-xl">
          <p class="mb-3 text-[11px] font-semibold tracking-[0.22em] text-emerald-600 uppercase">
            VIDEO · SIGN LANGUAGE · CTC
          </p>
          <h1
            class="font-semibold tracking-tight text-slate-900 text-3xl sm:text-4xl md:text-5xl leading-tight"
          >
            上传手语视频，一次性完成整段翻译
          </h1>
          <p class="mt-4 text-base md:text-lg text-slate-600">
            上传一段手语视频，系统将利用 CTC 模型对整段视频进行识别，并生成流畅自然的中文翻译。
          </p>
        </div>
        <div class="mt-6 md:mt-0 md:ml-10">
          <div
            class="inline-flex items-center gap-3 rounded-full border border-stone-200 bg-white/80 px-5 py-2 text-[11px] font-medium text-slate-500 shadow-sm rotate-[-2deg]"
          >
            <span
              class="inline-flex h-7 w-7 items-center justify-center rounded-full bg-emerald-500 text-[10px] font-semibold text-white"
            >
              CTC
            </span>
            <span class="tracking-[0.16em] uppercase">Batch video translate</span>
          </div>
        </div>
      </div>
      <div class="grid gap-8 md:grid-cols-12 md:items-start">
        <!-- 左：上传与预览 -->
        <div
          class="relative rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)] md:col-span-7 animate-fade-up animation-delay-100"
        >
          <div
            class="relative border border-dashed rounded-2xl p-8 text-center transition-all duration-300 bg-stone-50"
            :class="[
              isDragging
                ? 'border-emerald-500 bg-emerald-50'
                : 'border-stone-300 hover:bg-white hover:border-emerald-500'
            ]"
            @dragenter.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @dragover.prevent
            @drop.prevent="handleDrop"
          >
            <div v-if="!videoPreviewUrl" class="space-y-4">
              <div
                class="w-20 h-20 mx-auto mb-3 rounded-2xl bg-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-300/40"
              >
                <i class="bi bi-camera-video text-3xl text-white"></i>
              </div>
              <h3 class="text-xl font-semibold text-slate-900">
                拖拽或点击上传手语视频
              </h3>
              <p class="text-slate-500 text-sm">
                支持 MP4、MOV 等常见视频格式，建议时长 3-15 秒
              </p>
              <label
                class="group relative inline-flex items-center justify-center gap-2 rounded-xl border border-dashed border-stone-300 bg-white px-5 py-3 text-sm font-medium text-slate-800 shadow-sm cursor-pointer transition-colors hover:bg-stone-50"
              >
                <i class="bi bi-cloud-upload text-emerald-500 group-hover:text-emerald-600"></i>
                选择视频文件
                <input
                  type="file"
                  class="d-none"
                  accept="video/*"
                  @change="handleFileSelect"
                />
              </label>
              <p class="text-slate-400 text-xs">
                视频将在本地与服务器安全处理，仅用于本次翻译任务。
              </p>
            </div>

            <div v-else class="space-y-4">
              <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                  <div class="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center">
                    <i class="bi bi-film text-emerald-600"></i>
                  </div>
                  <div class="text-left">
                    <p class="font-medium text-slate-800 truncate max-w-[180px]">
                      {{ videoFile ? videoFile.name : '视频预览' }}
                    </p>
                    <p class="text-xs text-slate-500">
                      {{ videoFile ? formatSize(videoFile.size) : '来自历史记录' }}
                    </p>
                  </div>
                </div>
                <button
                  class="inline-flex items-center rounded-full px-3 py-1 text-xs text-slate-500 hover:bg-slate-100"
                  @click="clearVideo"
                >
                  <i class="bi bi-x mr-1"></i>移除
                </button>
              </div>

              <div class="rounded-xl overflow-hidden bg-slate-900/5 min-h-[200px] flex items-center justify-center">
                <video
                  v-if="videoPreviewUrl"
                  :src="videoPreviewUrl"
                  controls
                  playsinline
                  class="w-full rounded-xl bg-black"
                ></video>
                <div
                  v-else-if="result"
                  class="flex flex-col items-center justify-center gap-2 text-xs text-slate-400 px-6 py-10 text-center"
                >
                  <i class="bi bi-film text-2xl text-slate-300"></i>
                  <span>当前历史记录未保存原始视频文件，仅能查看文字结果。</span>
                </div>
              </div>
            </div>
          </div>

          <div v-if="uploading" class="mt-6 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-600">{{ uploadStatus }}</span>
              <span class="text-xs text-slate-500">{{ uploadProgress }}%</span>
            </div>
            <div class="h-2 bg-slate-100/80 rounded-full overflow-hidden">
              <div
                class="h-full bg-emerald-500 transition-all duration-300"
                :style="{ width: `${uploadProgress}%` }"
              ></div>
            </div>
          </div>

          <button
            class="mt-6 inline-flex w-full items-center justify-center rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-white shadow-[0_16px_32px_rgba(16,185,129,0.5)] transition-transform transition-shadow duration-200 hover:shadow-[0_20px_40px_rgba(16,185,129,0.6)] hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
            :disabled="!videoFile || uploading"
            @click="startVideoTranslation"
          >
            <i class="bi bi-lightning-charge mr-2"></i>
            开始视频翻译
          </button>
        </div>

        <!-- 右：结果展示 -->
        <div class="space-y-6 md:col-span-5">
          <div
            class="min-h-[220px] rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)] animate-fade-up animation-delay-200"
          >
            <div class="mb-4 flex items-center justify-between gap-2">
              <h3 class="flex items-center gap-2 font-semibold text-slate-900">
                <i class="bi bi-translate text-emerald-500"></i>
                翻译结果
              </h3>
              <button
                v-if="result"
                class="inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium hover:bg-emerald-100 transition-colors"
                :class="isCurrentFavorite ? 'text-pink-500 border-pink-200 bg-pink-50' : 'text-emerald-700'"
                @click.stop="toggleCurrentFavorite"
              >
                <span>{{ isCurrentFavorite ? '已收藏' : '收藏' }}</span>
              </button>
            </div>

            <div v-if="result" class="space-y-4">
              <div
                class="result-fade-in rounded-2xl border border-emerald-100 bg-emerald-50/70 p-4"
              >
                <p class="mb-1 text-xs text-slate-500">识别文本</p>
                <p class="break-words text-2xl font-semibold leading-snug text-slate-900">
                  {{ result.text }}
                </p>
                <p class="mt-2 text-xs text-slate-500">
                  置信度：<span class="font-semibold text-emerald-600">{{ Math.round(result.confidence) }}%</span>
                </p>
              </div>

              <div class="flex flex-wrap gap-2">
                <button
                  class="inline-flex items-center gap-1.5 rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm hover:bg-stone-50 transition-colors"
                  @click="playVoice"
                >
                  <i class="bi bi-volume-up text-emerald-500"></i> 播放语音
                </button>
                <button
                  class="inline-flex items-center gap-1.5 rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm hover:bg-stone-50 transition-colors"
                  @click="copyResult"
                >
                  <i class="bi bi-clipboard text-emerald-500"></i> 复制
                </button>
              </div>
            </div>

            <div v-else class="flex flex-col items-center justify-center py-10 text-slate-400">
              <i class="bi bi-file-earmark-play text-3xl mb-3 text-slate-300"></i>
              <p class="text-sm">上传并翻译视频后，结果将显示在这里</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useSpeech } from '~/composables/useSpeech'

useSeoMeta({
  title: '视频翻译 - 译手 HandTalk AI',
  description: '上传手语视频进行批量翻译',
})

const config = useRuntimeConfig()
const recognitionStore = useRecognitionStore()
const toast = useToast()
const speech = useSpeech()
const route = useRoute()

interface HistoryRecord {
  id: string
  type: string
  result: string
  confidence: number
  duration?: number
  thumbnail?: string
  videoUrl?: string
  createdAt: string
  favorite?: boolean
}

const videoFile = ref<File | null>(null)
const videoPreviewUrl = ref<string | null>(null)
const isDragging = ref(false)
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadStatus = ref('')
const result = ref<{ text: string; confidence: number; videoDuration?: number } | null>(null)
const currentHistoryId = ref<string | null>(null)

const isCurrentFavorite = computed(() => {
  if (!currentHistoryId.value) return false
  return (recognitionStore.favoriteHistory as HistoryRecord[]).some(
    (h) => h.id === currentHistoryId.value,
  )
})

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function toggleCurrentFavorite() {
  if (!currentHistoryId.value || !result.value) return
  const favList = recognitionStore.favoriteHistory as HistoryRecord[]
  const idx = favList.findIndex((h) => h.id === currentHistoryId.value)
  if (idx !== -1) {
    favList.splice(idx, 1)
    toast.success('已取消收藏')
  } else {
    favList.push({
      id: currentHistoryId.value,
      type: 'upload_video',
      result: result.value.text,
      confidence: result.value.confidence,
      createdAt: new Date().toISOString(),
    })
    toast.success('已收藏翻译结果')
  }
}

function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) setVideoFile(file)
}

function handleDrop(e: DragEvent) {
  isDragging.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file && file.type.startsWith('video/')) {
    setVideoFile(file)
  }
}

function setVideoFile(file: File) {
  videoFile.value = file
  if (videoPreviewUrl.value) URL.revokeObjectURL(videoPreviewUrl.value)
  videoPreviewUrl.value = URL.createObjectURL(file)
}

function clearVideo() {
  videoFile.value = null
  if (videoPreviewUrl.value) URL.revokeObjectURL(videoPreviewUrl.value)
  videoPreviewUrl.value = null
  result.value = null
  currentHistoryId.value = null
}

async function startVideoTranslation() {
  if (!videoFile.value) return
  uploading.value = true
  uploadProgress.value = 0
  uploadStatus.value = '上传中...'

  try {
    const formData = new FormData()
    formData.append('file', videoFile.value)

    const base = config.public.uploadBase || 'http://localhost:8000'
    const url = `${base}/api/v1/recognize/upload`

    const xhr = new XMLHttpRequest()
    await new Promise<void>((resolve, reject) => {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          uploadProgress.value = Math.round((e.loaded / e.total) * 30)
        }
      })
      xhr.addEventListener('load', () => resolve())
      xhr.addEventListener('error', () => reject(new Error('上传失败')))
      xhr.open('POST', url)
      xhr.send(formData)
    })

    if (xhr.status !== 200) {
      throw new Error(`服务器错误: ${xhr.status}`)
    }

    uploadProgress.value = 30
    uploadStatus.value = '识别中...'

    const data = JSON.parse(xhr.responseText)
    if (data.code !== 200 || !data.data) {
      throw new Error(data.message || '识别失败')
    }

    uploadProgress.value = 80
    uploadStatus.value = '处理结果...'

    const top = data.data.results?.[0] || data.data
    const confidence = top.confidence || data.data.confidence || 85
    result.value = {
      text: top.text || data.data.text || '',
      confidence,
      videoDuration: data.data.videoDuration || 0,
    }

    const historyId = data.id || Date.now().toString()

    let thumbnail: string | undefined
    try {
      thumbnail = await generateVideoThumbnail(videoFile.value!)
    } catch (e) {
      console.warn('生成视频缩略图失败:', e)
    }

    recognitionStore.addToHistory({
      id: historyId,
      type: 'upload_video',
      result: top.text,
      confidence,
      duration: data.videoDuration || 0,
      thumbnail,
      videoUrl: data.videoUrl ? `${config.public.uploadBase}${data.videoUrl}` : undefined,
      createdAt: data.createdAt || new Date().toISOString(),
      favorite: false,
    })

    currentHistoryId.value = historyId

    uploadStatus.value = '翻译完成'
    uploadProgress.value = 100
    toast.success('视频翻译完成')
  } catch (error: any) {
    console.error('视频翻译失败:', error)
    uploadStatus.value = '翻译失败'
    toast.error(error?.message || '视频翻译失败，请稍后重试')
  } finally {
    await new Promise(resolve => setTimeout(resolve, 500))
    uploading.value = false
  }
}

async function generateVideoThumbnail(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    const canvas = document.createElement('canvas')
    const url = URL.createObjectURL(file)
    let handled = false
    const cleanup = () => {
      if (handled) return
      handled = true
      URL.revokeObjectURL(url)
    }
    video.preload = 'metadata'
    video.src = url
    video.muted = true
    video.playsInline = true
    video.onloadeddata = () => {
      const targetTime = isFinite(video.duration) && video.duration > 0 ? Math.min(video.duration / 2, 1) : 0.1
      video.currentTime = targetTime
    }
    video.onseeked = () => {
      try {
        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 360
        const ctx = canvas.getContext('2d')
        if (!ctx) throw new Error('无法获取 Canvas 上下文')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
        cleanup()
        resolve(dataUrl)
      } catch (error) {
        cleanup()
        reject(error)
      }
    }
    video.onerror = () => { cleanup(); reject(new Error('视频加载失败')) }
  })
}

function playVoice() {
  if (result.value) speech.speak(result.value.text)
}

function copyResult() {
  if (result.value) {
    navigator.clipboard.writeText(result.value.text)
    toast.success('已复制到剪贴板')
  }
}

onUnmounted(() => {
  if (videoPreviewUrl.value) URL.revokeObjectURL(videoPreviewUrl.value)
})

onMounted(() => {
  const historyId = route.query.historyId as string | undefined
  if (!historyId) return
  if (process.client) recognitionStore.loadHistory()
  const record = (recognitionStore.history as HistoryRecord[]).find(
    (h) => h.id === historyId && h.type === 'upload_video',
  )
  if (!record) return
  result.value = {
    text: record.result,
    confidence: record.confidence,
    videoDuration: record.duration || 0,
  }
  currentHistoryId.value = record.id
  if (record.videoUrl) videoPreviewUrl.value = record.videoUrl
})
</script>

<style scoped>
@keyframes fade-up-soft {
  0% { opacity: 0; transform: translateY(16px) scale(0.98); }
  100% { opacity: 1; transform: translateY(0) scale(1); }
}
.animate-fade-up { animation: fade-up-soft 0.7s cubic-bezier(0.16, 1, 0.3, 1) both; }
.animation-delay-100 { animation-delay: 0.1s; }
.animation-delay-200 { animation-delay: 0.2s; }
.animation-delay-300 { animation-delay: 0.3s; }
.result-fade-in { animation: fade-up-soft 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }
</style>
