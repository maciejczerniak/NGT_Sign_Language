<template>
  <div class="chart-card">
    <div class="chart-card__header">
      <div>
        <h3 class="chart-title">Prediction Distribution</h3>
        <p class="chart-subtitle">predicted letter counts (all time)</p>
      </div>
      <span class="total-badge">{{ totalPredictions.toLocaleString() }} total</span>
    </div>
    <div class="chart-body">
      <canvas ref="canvasRef" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import {
  Chart,
  BarController,
  BarElement,
  LinearScale,
  CategoryScale,
  Tooltip,
} from 'chart.js'

Chart.register(BarController, BarElement, LinearScale, CategoryScale, Tooltip)

const props = defineProps<{
  letterCounts: Record<string, number>
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const totalPredictions = computed(() =>
  Object.values(props.letterCounts).reduce((a, b) => a + b, 0)
)

function buildChart() {
  if (!canvasRef.value) return

  const entries = Object.entries(props.letterCounts).sort((a, b) => b[1] - a[1])
  if (entries.length === 0) return

  const labels = entries.map(([k]) => k)
  const data = entries.map(([, v]) => v)
  const colors = labels.map((_, i) => `hsla(${270 + i * 6}, 70%, ${60 - i * 0.5}%, 0.7)`)

  // In-place update if chart already exists
  if (chart) {
    chart.data.labels = labels
    chart.data.datasets[0]?.data.splice(0, chart.data.datasets[0].data.length, ...data)
    if (Array.isArray(chart.data.datasets[0]?.backgroundColor)) {
      (chart.data.datasets[0].backgroundColor as string[]).splice(0, colors.length, ...colors)
    }
    chart.update('none')
    return
  }

  chart = new Chart(canvasRef.value, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Predictions',
        data,
        backgroundColor: colors,
        borderColor: 'rgba(168, 85, 247, 0.2)',
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(20, 0, 40, 0.97)',
          borderColor: 'rgba(168, 85, 247, 0.3)',
          borderWidth: 1,
          titleColor: 'rgba(216, 180, 254, 0.9)',
          bodyColor: 'rgba(216, 180, 254, 0.7)',
          padding: 10,
          callbacks: {
            label: (ctx) => ` ${(ctx.parsed.y ?? 0).toLocaleString()} predictions`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: 'rgba(216, 180, 254, 0.5)', font: { size: 11, weight: 'bold' } },
          border: { color: 'rgba(168, 85, 247, 0.12)' },
        },
        y: {
          grid: { color: 'rgba(168, 85, 247, 0.06)' },
          ticks: { color: 'rgba(216, 180, 254, 0.35)', font: { size: 11 } },
          border: { color: 'rgba(168, 85, 247, 0.12)' },
          beginAtZero: true,
        },
      },
    },
  })
}

onMounted(() => buildChart())
watch(() => props.letterCounts, () => buildChart(), { deep: true })
</script>

<style scoped>
.chart-card {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(168, 85, 247, 0.18);
  border-radius: 12px;
  padding: 1.5rem;
}

.chart-card__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 1.25rem;
}

.chart-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: white;
  margin: 0 0 0.2rem;
  letter-spacing: 0.02em;
}

.chart-subtitle {
  font-size: 0.7rem;
  color: rgba(216, 180, 254, 0.45);
  margin: 0;
  letter-spacing: 0.04em;
}

.total-badge {
  font-size: 0.7rem;
  background: rgba(168, 85, 247, 0.15);
  border: 1px solid rgba(168, 85, 247, 0.3);
  border-radius: 20px;
  padding: 0.2rem 0.65rem;
  color: var(--color-brand-accent);
  white-space: nowrap;
  align-self: flex-start;
}

.chart-body {
  height: 340px;
  position: relative;
}
</style>
