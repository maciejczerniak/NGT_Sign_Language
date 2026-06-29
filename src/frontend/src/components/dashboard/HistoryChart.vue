<template>
  <div class="chart-card">
    <div class="chart-card__header">
      <div>
        <h3 class="chart-title">{{ title }}</h3>
        <p class="chart-subtitle">{{ subtitle }}</p>
      </div>
      <div v-if="info" class="chart-info" @mouseenter="showInfo = true" @mouseleave="showInfo = false">
        <i class="pi pi-info-circle" />
        <div v-if="showInfo" class="chart-tooltip">{{ info }}</div>
      </div>
    </div>
    <div v-if="isEmpty" class="chart-empty">
      <i class="pi pi-chart-line" />
      No data available for the selected time range
    </div>

    <div v-else class="chart-body">
      <canvas ref="canvasRef" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
  type ChartDataset,
} from 'chart.js'

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip)

interface HistoryPoint {
  timestamp: string
  request_count: number
  error_count: number
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
}

const props = defineProps<{
  title: string
  subtitle: string
  info?: string
  data: HistoryPoint[]
  mode: 'latency' | 'requests'
  activeRange: string
}>()

const showInfo = ref(false)

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const isEmpty = computed(() => props.data.length === 0)

function formatLabel(ts: string): string {
  const d = new Date(ts)
  if (props.activeRange === '1h' || props.activeRange === '6h') {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  if (props.activeRange === '1d') {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

function buildChart() {
  if (!canvasRef.value || isEmpty.value) return

  const labels = props.data.map((d) => formatLabel(d.timestamp))

  let dataset1: number[]
  let dataset2: number[]

  if (props.mode === 'latency') {
    dataset1 = props.data.map((d) => d.p50_latency_ms)
    dataset2 = props.data.map((d) => d.p95_latency_ms)
  } else {
    dataset1 = props.data.map((d) => d.request_count)
    dataset2 = props.data.map((d) => d.error_count)
  }

  // In-place update if chart already exists
  if (chart) {
    chart.data.labels = labels
    chart.data.datasets[0]?.data.splice(0, chart.data.datasets[0].data.length, ...dataset1)
    chart.data.datasets[1]?.data.splice(0, chart.data.datasets[1].data.length, ...dataset2)
    chart.update('none')
    return
  }

  const datasets: ChartDataset<'line'>[] = props.mode === 'latency'
    ? [
        {
          label: 'p50 latency',
          data: dataset1,
          borderColor: 'rgba(56, 189, 248, 0.9)',
          backgroundColor: 'rgba(56, 189, 248, 0.07)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'p95 latency',
          data: dataset2,
          borderColor: 'rgba(168, 85, 247, 0.8)',
          backgroundColor: 'rgba(168, 85, 247, 0.05)',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: false,
          tension: 0.3,
        },
      ]
    : [
        {
          label: 'requests',
          data: dataset1,
          borderColor: 'rgba(74, 222, 128, 0.8)',
          backgroundColor: 'rgba(74, 222, 128, 0.07)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'errors',
          data: dataset2,
          borderColor: 'rgba(248, 113, 113, 0.8)',
          backgroundColor: 'rgba(248, 113, 113, 0.07)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.3,
        },
      ]

  chart = new Chart(canvasRef.value, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(20, 0, 40, 0.97)',
          borderColor: 'rgba(168, 85, 247, 0.35)',
          borderWidth: 1,
          titleColor: 'rgba(216, 180, 254, 0.9)',
          bodyColor: 'rgba(216, 180, 254, 0.7)',
          padding: 12,
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed.y ?? 0
              const suffix = props.mode === 'latency' ? ' ms' : ' req'
              return ` ${ctx.dataset.label}: ${v.toFixed(props.mode === 'latency' ? 2 : 0)}${suffix}`
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(168, 85, 247, 0.06)' },
          ticks: {
            color: 'rgba(216, 180, 254, 0.35)',
            font: { size: 11 },
            maxTicksLimit: 8,
            maxRotation: 0,
          },
          border: { color: 'rgba(168, 85, 247, 0.12)' },
        },
        y: {
          grid: { color: 'rgba(168, 85, 247, 0.06)' },
          ticks: {
            color: 'rgba(216, 180, 254, 0.35)',
            font: { size: 11 },
            callback: (v) => props.mode === 'latency' ? `${v}ms` : `${v}`,
          },
          border: { color: 'rgba(168, 85, 247, 0.12)' },
          beginAtZero: true,
        },
      },
    },
  })
}

onMounted(() => buildChart())
watch(() => [props.data, props.activeRange], () => {
  if (chart && props.data.length > 0) {
    const currentLabels = chart.data.labels as string[]
    const newLabels = props.data.map((d) => formatLabel(d.timestamp))
    if (currentLabels[0] !== newLabels[0]) {
      chart.destroy()
      chart = null
    }
  }
  buildChart()
}, { deep: true })

onUnmounted(() => {
  if (chart) {
    chart.destroy()
    chart = null
  }
})
</script>

<style scoped>
.chart-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(168, 85, 247, 0.15);
  border-radius: 14px;
  padding: 1.5rem;
}

.chart-card__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 1.25rem;
  gap: 1rem;
}

.chart-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: white;
  margin: 0 0 0.2rem;
  letter-spacing: 0.02em;
}

.chart-subtitle {
  font-size: 0.68rem;
  color: rgba(216, 180, 254, 0.4);
  margin: 0;
  letter-spacing: 0.04em;
}

.chart-empty {
  height: 340px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  color: rgba(216, 180, 254, 0.25);
  font-size: 0.75rem;
  letter-spacing: 0.04em;
}

.chart-empty .pi {
  font-size: 2rem;
  opacity: 0.3;
}

.chart-body {
  height: 340px;
  position: relative;
}

.chart-info {
  position: relative;
  cursor: help;
  color: rgba(216, 180, 254, 0.3);
  font-size: 0.75rem;
  flex-shrink: 0;
  transition: color 0.2s;
}

.chart-info:hover { color: rgba(216, 180, 254, 0.75); }

.chart-tooltip {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 240px;
  background: rgba(18, 0, 36, 0.98);
  border: 1px solid rgba(168, 85, 247, 0.35);
  border-radius: 8px;
  padding: 0.65rem 0.75rem;
  font-size: 0.68rem;
  color: rgba(216, 180, 254, 0.75);
  line-height: 1.6;
  z-index: 999;
  pointer-events: none;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  white-space: normal;
}
</style>
