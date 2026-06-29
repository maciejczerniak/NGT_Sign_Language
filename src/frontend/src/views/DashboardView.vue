<template>
  <div class="dashboard-root">
    <div class="bg-grid" aria-hidden="true" />

    <header class="top-nav">
      <div class="nav-left">
        <div class="nav-logo"><i class="pi pi-chart-line" /></div>
        <div>
          <span class="nav-title">Operations Console</span>
          <span class="nav-breadcrumb">Sign Language API</span>
        </div>
      </div>

      <div class="nav-center">
        <span class="range-label">TIME RANGE</span>
        <div class="range-selector">
          <button
            v-for="r in ranges"
            :key="r.value"
            class="range-btn"
            :class="{ 'range-btn--active': activeRange === r.value && !customMode }"
            @click="onRangeChange(r.value)"
          >
            {{ r.label }}
          </button>
          <button
            class="range-btn range-btn--custom"
            :class="{ 'range-btn--active': customMode }"
            @click="toggleCustomMode"
          >
            <i class="pi pi-calendar" />
            CUSTOM
          </button>
        </div>

        <Transition name="fade">
          <div v-if="customMode" class="date-picker-row">
            <input v-model="customFrom" type="date" class="date-input" :max="customTo || undefined" />
            <span class="date-sep">→</span>
            <input v-model="customTo" type="date" class="date-input" :min="customFrom || undefined" />
            <button class="date-apply-btn" :disabled="!customFrom || !customTo" @click="applyCustomRange">Apply</button>
          </div>
        </Transition>

        <div class="live-badge" :class="{ 'live-badge--paused': !autoRefresh }">
          <span class="live-dot" />
          {{ autoRefresh ? 'LIVE' : 'PAUSED' }}
        </div>
        <Button icon="pi pi-refresh" text rounded :loading="fetching" class="refresh-btn" title="Refresh now" @click="handleManualRefresh" />
      </div>

      <div class="nav-right">
        <Button
          icon="pi pi-arrow-left"
          label="Home Page"
          text
          class="back-to-app-btn"
          title="Return to the app"
          @click="goToApp"
        />
        <div class="user-chip">
          <i class="pi pi-user" />
          <span>{{ auth.user?.email ?? 'admin' }}</span>
        </div>
        <Button icon="pi pi-sign-out" text rounded class="logout-btn" title="Sign out" @click="handleLogout" />
      </div>
    </header>

    <main class="dashboard-body">

      <Transition name="slide-down">
        <div v-if="fetchError" class="fetch-error">
          <i class="pi pi-exclamation-triangle" />
          <span>Could not reach the API: <em>{{ fetchError }}</em></span>
          <button class="error-dismiss" @click="fetchError = ''"><i class="pi pi-times" /></button>
        </div>
      </Transition>

      <div class="page-header">
        <div>
          <h1 class="page-title">Operational Monitoring</h1>
          <p class="page-subtitle">
            Showing data for <span class="range-highlight">{{ activePeriodLabel }}</span>
            — last updated <span class="last-updated">{{ lastUpdatedLabel }}</span>
          </p>
        </div>
        <div class="header-right">
          <div class="health-pill" :class="`health-pill--${healthStatus}`">
            <span class="health-dot" />
            {{ healthStatus.toUpperCase() }}
          </div>
          <button class="edit-btn" :class="{ 'edit-btn--active': editMode }" @click="editMode = !editMode">
            <i :class="editMode ? 'pi pi-check' : 'pi pi-sliders-h'" />
            {{ editMode ? 'Done' : 'Edit Thresholds' }}
          </button>
        </div>
      </div>

      <!-- ── KPI cards ──────────────────────────────────────── -->
      <section class="kpi-grid">
        <MetricCard
          label="Total Requests" :value="metrics.total_requests" icon="pi-server" variant="default"
          :subtext="activePeriodLabel" :edit-mode="editMode"
          description="Total HTTP requests received in the selected time period."
        />
        <MetricCard
          label="Error Rate" :value="errorRatePct" unit="%" icon="pi-exclamation-circle"
          :variant="errorRatePct > thresholds.errorRate ? 'danger' : errorRatePct > 1 ? 'warning' : 'success'"
          :alert-threshold="thresholds.errorRate" :subtext="`4xx + 5xx — threshold: ${thresholds.errorRate}%`"
          :edit-mode="editMode" description="Percentage of requests that returned an error response."
          @threshold-change="(v) => saveThreshold('errorRate', v)"
        />
        <MetricCard
          label="p50 Latency" :value="metrics.p50_latency_ms" unit="ms" icon="pi-clock"
          :variant="metrics.p50_latency_ms > thresholds.p50 ? 'warning' : 'success'"
          :alert-threshold="thresholds.p50" :subtext="`median — threshold: ${thresholds.p50}ms`"
          :edit-mode="editMode" description="Median latency — half of requests are faster than this."
          @threshold-change="(v) => saveThreshold('p50', v)"
        />
        <MetricCard
          label="p95 Latency" :value="metrics.p95_latency_ms" unit="ms" icon="pi-bolt"
          :variant="metrics.p95_latency_ms > thresholds.p95 ? 'danger' : metrics.p95_latency_ms > thresholds.p50 ? 'warning' : 'success'"
          :alert-threshold="thresholds.p95" :subtext="`95th pct — threshold: ${thresholds.p95}ms`"
          :edit-mode="editMode" description="95% of requests complete faster than this value."
          @threshold-change="(v) => saveThreshold('p95', v)"
        />
        <MetricCard
          label="Total Predictions" :value="metrics.total_predictions" icon="pi-eye" variant="default"
          :subtext="activePeriodLabel" :edit-mode="editMode"
          description="Sign language inference calls in the selected time period."
        />
        <MetricCard
          label="Avg Confidence" :value="hasNoPredictions ? -1 : avgConfidencePct" unit="%" icon="pi-verified"
          :variant="hasNoPredictions ? 'default' : avgConfidencePct < thresholds.confidence ? 'warning' : 'success'"
          :alert-threshold="thresholds.confidence" :higher-is-better="true"
          :subtext="hasNoPredictions ? 'no predictions yet' : `drift proxy — threshold: ${thresholds.confidence}%`"
          :edit-mode="editMode" :no-data="hasNoPredictions"
          description="Average model confidence. Drop below threshold may indicate distribution shift."
          @threshold-change="(v) => saveThreshold('confidence', v)"
        />
      </section>

      <!-- ── Stock-style charts ─────────────────────────────── -->
      <section class="charts-grid">
        <HistoryChart
          title="Request & Error Volume"
          subtitle="requests & errors over time"
          info="Total HTTP requests (green) and error responses — 4xx and 5xx (red) — over time. A rising error line relative to requests means your error rate is climbing. Spikes in requests without matching error spikes are healthy traffic."
          mode="requests"
          :data="historyData"
          :active-range="activeRange"
        />
        <HistoryChart
          title="Latency Trend"
          subtitle="p50 & p95 response time (ms)"
          info="p50 (cyan) is the median response time — half of requests are faster. p95 (purple dashed) is the 95th percentile — 95% of requests complete within this time. A growing gap between p50 and p95 means occasional slow requests are getting worse."
          mode="latency"
          :data="historyData"
          :active-range="activeRange"
        />
      </section>

      <!-- ── Entropy trend full width ───────────────────────── -->
      <section class="chart-card">
        <div class="chart-card__header">
          <div>
            <h3 class="chart-title">Prediction Uncertainty Trend</h3>
            <p class="chart-subtitle">model uncertainty over time — 0% = fully certain, 100% = maximally confused across top-3</p>
          </div>
          <div class="header-right-group">
            <div class="entropy-badge" :class="entropyBadgeClass">
              <span>AVG UNCERTAINTY {{ entropyPct.toFixed(1) }}%</span>
            </div>
            <div class="chart-info" @mouseenter="showEntropyInfo = true" @mouseleave="showEntropyInfo = false">
              <i class="pi pi-info-circle" />
              <div v-if="showEntropyInfo" class="chart-tooltip chart-tooltip--left">
                How spread the model's probability is across its top-3 predictions. 0% means it picked one letter with full certainty. 100% means all three options looked equally likely — the model had no idea. A rising trend over time suggests the model is seeing inputs it was not trained on. <strong>Lower is better.</strong>
              </div>
            </div>
          </div>
        </div>
        <div v-if="fetching && hasNoPredictions" class="chart-loading-sm">
          <div class="spinner" />
          <span>Loading data…</span>
        </div>
        <div v-else-if="hasNoPredictions" class="chart-empty-sm">
          <i class="pi pi-chart-line" />
          <span>No predictions in this time range</span>
        </div>
        <div v-else class="chart-body-entropy">
          <canvas ref="entropyCanvasRef" />
        </div>
      </section>

      <!-- ── Prediction distribution full width ────────────── -->
      <section class="chart-card">
        <div class="chart-card__header">
          <div>
            <h3 class="chart-title">Prediction Distribution</h3>
            <p class="chart-subtitle">predicted letter counts — {{ activePeriodLabel }}</p>
          </div>
          <div class="header-right-group">
            <span class="total-badge-sm">{{ metrics.total_predictions.toLocaleString() }} total</span>
            <div class="chart-info" @mouseenter="showDistInfo = true" @mouseleave="showDistInfo = false">
              <i class="pi pi-info-circle" />
              <div v-if="showDistInfo" class="chart-tooltip chart-tooltip--left">
                How many times each letter was predicted in the selected period. An uneven distribution is expected — some letters appear more in natural signing. A sudden shift where one letter dominates unexpectedly may indicate a stuck prediction or model issue.
              </div>
            </div>
          </div>
        </div>
        <div v-if="fetching && metrics.total_predictions === 0" class="chart-loading-sm">
          <div class="spinner" />
          <span>Loading data…</span>
        </div>
        <div v-else-if="metrics.total_predictions === 0" class="chart-empty-sm">
          <i class="pi pi-hand" />
          <span>No predictions in this time range</span>
        </div>
        <div v-else class="chart-body-letters">
          <canvas ref="letterCanvasRef" />
        </div>
      </section>

      <!-- ── Error breakdown + model signal monitor ─────────── -->
      <section class="lower-grid">
        <ErrorBreakdown :breakdown="metrics.error_breakdown" :total-errors="metrics.error_count" />

        <div class="drift-card">
  <div class="drift-header">
    <div class="drift-title-row">
      <div>
        <div class="drift-title-with-info">
          <h3 class="chart-title">Model Signal Monitor</h3>
          <div class="drift-info" @mouseenter="showDriftInfo = true" @mouseleave="showDriftInfo = false">
            <i class="pi pi-info-circle" />
            <div v-if="showDriftInfo" class="drift-tooltip">
              <div class="drift-tooltip-row">
                <span class="dot dot--good" />
                <span><strong>High conf, low uncertainty</strong> — model decisive, healthy state</span>
              </div>
              <div class="drift-tooltip-row">
                <span class="dot dot--bad" />
                <span><strong>Low conf + high uncertainty</strong> — confused across classes, consider retraining</span>
              </div>
              <div class="drift-tooltip-row">
                <span class="dot dot--warn" />
                <span><strong>Low conf, low uncertainty</strong> — confidently picking wrong class, concept drift</span>
              </div>
              <div class="drift-tooltip-row">
                <span class="dot dot--warn" />
                <span><strong>High conf + rising uncertainty</strong> — early warning, monitor closely</span>
              </div>
            </div>
          </div>
        </div>
        <p class="chart-subtitle">confidence & uncertainty proxy — not confirmed accuracy</p>
      </div>
    </div>

            <div class="drift-status-wrap">
              <span
                class="drift-status"
                :class="driftStatusClass"
                @mouseenter="showStatusInfo = true"
                @mouseleave="showStatusInfo = false"
              >{{ driftStatusLabel }}</span>
              <div v-if="showStatusInfo" class="status-tooltip">
                <div class="status-tooltip-row" :class="{ 'row--active': driftStatusLabel === 'STABLE' }">
                  <span class="dot dot--good" />
                  <div>
                    <strong>STABLE</strong> — confidence high, uncertainty low.
                    Model performing as expected. No action needed.
                  </div>
                </div>
                <div class="status-tooltip-row" :class="{ 'row--active': driftStatusLabel === 'MONITOR' }">
                  <span class="dot dot--warn" />
                  <div>
                    <strong>MONITOR</strong> — one signal degrading.
                    Check if a new user, lighting condition, or signing style is affecting predictions. Not critical yet.
                  </div>
                </div>
                <div class="status-tooltip-row" :class="{ 'row--active': driftStatusLabel === 'INVESTIGATE' }">
                  <span class="dot dot--bad" />
                  <div>
                    <strong>INVESTIGATE</strong> — both signals degraded.
                    Possible causes: model seeing unfamiliar inputs, data pipeline issue, or significant change in user signing style. Consider collecting feedback and retraining.
                  </div>
                </div>
                <div class="status-tooltip-row" :class="{ 'row--active': driftStatusLabel === 'NO DATA' }">
                  <span class="dot dot--idle" />
                  <div>
                    <strong>NO DATA</strong> — no predictions recorded in the selected time range.
                  </div>
                </div>
                <p class="status-tooltip-note">These are unsupervised signal proxies — not confirmed accuracy. Use the feedback system to collect ground truth labels.</p>
              </div>
            </div>
          </div>

          <div v-if="hasNoPredictions" class="drift-no-data">No prediction data in this time range</div>

          <template v-else>
            <div class="drift-metric-label">
              <span>Avg Confidence</span>
              <span class="drift-metric-value">{{ avgConfidencePct.toFixed(1) }}%</span>
            </div>
            <div class="drift-bar-track">
              <div class="drift-bar-fill" :class="driftBarClass" :style="{ width: `${Math.min(avgConfidencePct, 100)}%` }" />
            </div>

            <div class="drift-metric-label" style="margin-top: 1rem">
              <span>Avg Uncertainty Score</span>
              <span class="drift-metric-value">{{ entropyPct.toFixed(1) }}%</span>
            </div>
            <div class="drift-bar-track">
              <div
                class="drift-bar-fill"
                :class="entropyBarClass"
                :style="{ width: `${Math.min(entropyPct, 100)}%` }"
              />
            </div>
            <div class="drift-bar-sublabels">
              <span>certain (0%)</span>
              <span>maximally confused (100%)</span>
            </div>
          </template>

          <p class="drift-note">
            <strong>Confidence</strong> measures how certain the model is about its top prediction.
            <strong>Uncertainty</strong> measures how spread the probability is across the top-3 classes — rising uncertainty with falling confidence is a strong signal to investigate. Neither replaces labelled accuracy.
          </p>
        </div>
      </section>

    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import Button from 'primevue/button'
import {
  Chart,
  BarController, BarElement,
  LineController, LineElement, PointElement,
  LinearScale, CategoryScale, Filler, Tooltip,
} from 'chart.js'
import { useAuthStore } from '../stores/auth'
import { api } from '../composables/useApi'
import MetricCard from '../components/dashboard/MetricCard.vue'
import HistoryChart from '../components/dashboard/HistoryChart.vue'
import ErrorBreakdown from '../components/dashboard/ErrorBreakdown.vue'

Chart.register(BarController, BarElement, LineController, LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip)

// ── Types ────────────────────────────────────────────────────────
interface ErrorRow { path: string; status_code: number; count: number }

interface Metrics {
  total_requests: number
  error_count: number
  error_rate: number
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  avg_confidence: number
  avg_entropy: number
  total_predictions: number
  letter_counts: Record<string, number>
  error_breakdown: ErrorRow[]
}

interface HistoryPoint {
  timestamp: string
  request_count: number
  error_count: number
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  avg_entropy: number
}

interface Thresholds {
  errorRate: number
  p50: number
  p95: number
  confidence: number
}

const DEFAULT_THRESHOLDS: Thresholds = { errorRate: 5, p50: 500, p95: 1000, confidence: 60 }

function loadThresholds(): Thresholds {
  try {
    const saved = localStorage.getItem('dashboard_thresholds')
    if (saved) return { ...DEFAULT_THRESHOLDS, ...JSON.parse(saved) }
  } catch { /* ignore */ }
  return { ...DEFAULT_THRESHOLDS }
}

// ── State ────────────────────────────────────────────────────────
const auth = useAuthStore()
const router = useRouter()

const metrics = ref<Metrics>({
  total_requests: 0, error_count: 0, error_rate: 0,
  avg_latency_ms: 0, p50_latency_ms: 0, p95_latency_ms: 0,
  avg_confidence: 0, avg_entropy: 0, total_predictions: 0,
  letter_counts: {}, error_breakdown: [],
})

const historyData = ref<HistoryPoint[]>([])
const activeRange = ref('1h')
const customMode = ref(false)
const customFrom = ref('')
const customTo = ref('')
const fetching = ref(false)
const fetchError = ref('')
const lastUpdated = ref<Date | null>(null)
const autoRefresh = ref(true)
const editMode = ref(false)
const showDriftInfo = ref(false)
const showStatusInfo = ref(false)
const showEntropyInfo = ref(false)
const showDistInfo = ref(false)
const thresholds = ref<Thresholds>(loadThresholds())
const letterCanvasRef = ref<HTMLCanvasElement | null>(null)
const entropyCanvasRef = ref<HTMLCanvasElement | null>(null)
let letterChart: Chart | null = null
let entropyChart: Chart | null = null
let timer: ReturnType<typeof setInterval> | null = null

const ranges = [
  { label: '1H', value: '1h' }, { label: '6H', value: '6h' },
  { label: '1D', value: '1d' }, { label: '7D', value: '7d' },
  { label: '30D', value: '30d' }, { label: 'ALL', value: 'all' },
]

// ── Computed ─────────────────────────────────────────────────────
const errorRatePct = computed(() => +(metrics.value.error_rate * 100).toFixed(4))
const avgConfidencePct = computed(() => +(metrics.value.avg_confidence * 100).toFixed(2))
const hasNoPredictions = computed(() => metrics.value.total_predictions === 0)
const entropyPct = computed(() => +((metrics.value.avg_entropy / 1.585) * 100).toFixed(1))

const activePeriodLabel = computed(() => {
  if (customMode.value && customFrom.value && customTo.value) return `${customFrom.value} → ${customTo.value}`
  const map: Record<string, string> = {
    '1h': 'last 1 hour', '6h': 'last 6 hours', '1d': 'last 24 hours',
    '7d': 'last 7 days', '30d': 'last 30 days', 'all': 'all time',
  }
  return map[activeRange.value] ?? activeRange.value
})

const healthStatus = computed(() => {
  if (fetchError.value) return 'error'
  if (errorRatePct.value > thresholds.value.errorRate || metrics.value.p95_latency_ms > thresholds.value.p95) return 'degraded'
  if (metrics.value.total_requests === 0) return 'idle'
  return 'healthy'
})

const lastUpdatedLabel = computed(() => lastUpdated.value ? lastUpdated.value.toLocaleTimeString() : 'never')

const driftBarClass = computed(() => {
  if (avgConfidencePct.value >= 80) return 'drift-bar-fill--good'
  if (avgConfidencePct.value >= thresholds.value.confidence) return 'drift-bar-fill--warn'
  return 'drift-bar-fill--bad'
})

const entropyBarClass = computed(() => {
  const e = metrics.value.avg_entropy
  if (e < 0.3) return 'drift-bar-fill--good'
  if (e < 0.8) return 'drift-bar-fill--warn'
  return 'drift-bar-fill--bad'
})

const entropyBadgeClass = computed(() => {
  if (hasNoPredictions.value) return 'entropy-badge--idle'
  const e = metrics.value.avg_entropy
  if (e < 0.3) return 'entropy-badge--good'
  if (e < 0.8) return 'entropy-badge--warn'
  return 'entropy-badge--bad'
})

const driftStatusClass = computed(() => {
  if (hasNoPredictions.value) return 'drift-status--idle'
  if (avgConfidencePct.value >= 80 && metrics.value.avg_entropy < 0.3) return 'drift-status--good'
  if (avgConfidencePct.value >= thresholds.value.confidence && metrics.value.avg_entropy < 0.8) return 'drift-status--warn'
  return 'drift-status--bad'
})

const driftStatusLabel = computed(() => {
  if (hasNoPredictions.value) return 'NO DATA'
  if (avgConfidencePct.value >= 80 && metrics.value.avg_entropy < 0.3) return 'STABLE'
  if (avgConfidencePct.value >= thresholds.value.confidence && metrics.value.avg_entropy < 0.8) return 'MONITOR'
  return 'INVESTIGATE'
})

// ── Threshold management ─────────────────────────────────────────
function saveThreshold(key: keyof Thresholds, value: number) {
  thresholds.value = { ...thresholds.value, [key]: value }
  localStorage.setItem('dashboard_thresholds', JSON.stringify(thresholds.value))
}

// ── Entropy history chart ────────────────────────────────────────
function buildEntropyChart() {
  if (!entropyCanvasRef.value || historyData.value.length === 0) return

  if (entropyChart && entropyChart.canvas !== entropyCanvasRef.value) {
    entropyChart.destroy()
    entropyChart = null
  }

  const labels = historyData.value.map((d) => {
    const dt = new Date(d.timestamp)
    if (['1h', '6h', '1d'].includes(activeRange.value)) {
      return dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    return dt.toLocaleDateString([], { month: 'short', day: 'numeric' })
  })
  const data = historyData.value.map((d) => +((d.avg_entropy / 1.585) * 100).toFixed(2))

  if (entropyChart) {
    entropyChart.data.labels = labels
    entropyChart.data.datasets[0]?.data.splice(0, entropyChart.data.datasets[0].data.length, ...data)
    entropyChart.update('none')
    return
  }

  entropyChart = new Chart(entropyCanvasRef.value, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'uncertainty',
        data,
        borderColor: 'rgba(251, 191, 36, 0.85)',
        backgroundColor: 'rgba(251, 191, 36, 0.06)',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(20, 0, 40, 0.97)',
          borderColor: 'rgba(251, 191, 36, 0.35)',
          borderWidth: 1,
          titleColor: 'rgba(216, 180, 254, 0.9)',
          bodyColor: 'rgba(216, 180, 254, 0.7)',
          padding: 12,
          callbacks: {
            label: (ctx) => ` uncertainty: ${(ctx.parsed.y ?? 0).toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(251, 191, 36, 0.05)' },
          ticks: { color: 'rgba(216, 180, 254, 0.35)', font: { size: 11 }, maxTicksLimit: 8, maxRotation: 0 },
          border: { color: 'rgba(251, 191, 36, 0.1)' },
        },
        y: {
          grid: { color: 'rgba(251, 191, 36, 0.05)' },
          ticks: { color: 'rgba(216, 180, 254, 0.35)', font: { size: 11 }, callback: (v) => `${v}%` },
          border: { color: 'rgba(251, 191, 36, 0.1)' },
          beginAtZero: true,
          max: 100,
        },
      },
    },
  })
}

// ── Letter distribution chart ────────────────────────────────────
function buildLetterChart() {
  if (!letterCanvasRef.value) return
  const entries = Object.entries(metrics.value.letter_counts).sort((a, b) => b[1] - a[1])
  if (entries.length === 0) return

  // If the chart is bound to a stale (remounted) canvas, destroy it so
  // we rebuild against the live canvas element instead of updating a
  // detached one.
  if (letterChart && letterChart.canvas !== letterCanvasRef.value) {
    letterChart.destroy()
    letterChart = null
  }

  const labels = entries.map(([k]) => k)
  const data = entries.map(([, v]) => v)
  const colors = labels.map((_, i) => `hsla(${270 + i * 6}, 70%, ${60 - i * 0.5}%, 0.7)`)

  if (letterChart) {
    letterChart.data.labels = labels
    letterChart.data.datasets[0]?.data.splice(0, letterChart.data.datasets[0].data.length, ...data)
    if (Array.isArray(letterChart.data.datasets[0]?.backgroundColor)) {
      (letterChart.data.datasets[0].backgroundColor as string[]).splice(0, colors.length, ...colors)
    }
    letterChart.update('none')
    return
  }

  letterChart = new Chart(letterCanvasRef.value, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Predictions', data, backgroundColor: colors, borderColor: 'rgba(168, 85, 247, 0.2)', borderWidth: 1, borderRadius: 4 }],
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
          callbacks: { label: (ctx) => ` ${(ctx.parsed.y ?? 0).toLocaleString()} predictions` },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: 'rgba(216, 180, 254, 0.5)', font: { size: 11, weight: 'bold' } }, border: { color: 'rgba(168, 85, 247, 0.12)' } },
        y: { grid: { color: 'rgba(168, 85, 247, 0.06)' }, ticks: { color: 'rgba(216, 180, 254, 0.35)', font: { size: 11 } }, border: { color: 'rgba(168, 85, 247, 0.12)' }, beginAtZero: true },
      },
    },
  })
}

watch(() => metrics.value.letter_counts, async () => {
  if (metrics.value.total_predictions > 0) {
    await nextTick()
    buildLetterChart()
  }
}, { deep: true })

watch(() => historyData.value, async () => {
  if (historyData.value.length > 0) {
    if (entropyChart) {
      const currentLabels = entropyChart.data.labels as string[]
      const firstPoint = historyData.value[0]
      const firstNew = firstPoint
        ? new Date(firstPoint.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        : ''
      if (currentLabels[0] !== firstNew) {
        entropyChart.destroy()
        entropyChart = null
      }
    }
    await nextTick()
    buildEntropyChart()
  }
}, { deep: true })

// ── Data fetching ────────────────────────────────────────────────
function buildApiParams(): string {
  if (customMode.value && customFrom.value && customTo.value) {
    return `range=custom&date_from=${customFrom.value}&date_to=${customTo.value}`
  }
  return `range=${activeRange.value}`
}

async function fetchMetrics() {
  fetching.value = true
  fetchError.value = ''
  try {
    const params = buildApiParams()
    const [metricsRes, historyRes] = await Promise.all([
      api.get<Metrics>(`/admin/metrics?${params}`),
      api.get<HistoryPoint[]>(`/admin/metrics/history?${params}`),
    ])
    metrics.value = metricsRes.data
    historyData.value = historyRes.data
    lastUpdated.value = new Date()
  } catch (err: unknown) {
    fetchError.value = err instanceof Error ? err.message : 'Unknown error'
  } finally {
    fetching.value = false
  }
}

function onRangeChange(range: string) {
  customMode.value = false
  activeRange.value = range
  fetchMetrics()
}

function toggleCustomMode() {
  customMode.value = !customMode.value
  if (!customMode.value) fetchMetrics()
}

function applyCustomRange() {
  if (!customFrom.value || !customTo.value) return
  fetchMetrics()
}

function handleManualRefresh() { fetchMetrics() }

function resetTimer() {
  if (timer) clearInterval(timer)
  if (customMode.value) { autoRefresh.value = false; return }
  autoRefresh.value = true
  timer = setInterval(fetchMetrics, 30_000)
}

function handleVisibilityChange() {
  if (document.hidden) {
    if (timer) clearInterval(timer)
  } else if (!customMode.value) {
    fetchMetrics()
    resetTimer()
  }
}

watch(customMode, (isCustom) => {
  if (isCustom) { if (timer) clearInterval(timer); autoRefresh.value = false }
  else resetTimer()
})

// ── Return to the main app from the dashboard. ───────────────────
function goToApp() {
  router.push('/')
}

function handleLogout() {
  auth.logout()
  router.push({ name: 'login' })
}

onMounted(async () => {
  await fetchMetrics()
  resetTimer()
  document.addEventListener('visibilitychange', handleVisibilityChange)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
  if (letterChart) letterChart.destroy()
  if (entropyChart) entropyChart.destroy()
  document.removeEventListener('visibilitychange', handleVisibilityChange)
})
</script>

<style scoped>
.dashboard-root {
  min-height: 100%;
  background-color: var(--color-brand-purple);
  font-family: 'DM Mono', 'Fira Code', monospace;
  position: relative;
}

.bg-grid {
  position: fixed; inset: 0;
  background-image: radial-gradient(circle, rgba(168, 85, 247, 0.07) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none; z-index: 0;
}

.top-nav {
  position: sticky; top: 0; z-index: 100;
  display: grid; grid-template-columns: auto 1fr auto;
  align-items: center; padding: 0 2.5rem; min-height: 56px;
  background: rgba(30, 0, 51, 0.92);
  border-bottom: 1px solid rgba(168, 85, 247, 0.2);
  backdrop-filter: blur(16px); gap: 1.5rem;
}

.nav-left { display: flex; align-items: center; gap: 0.875rem; flex-shrink: 0; }
.nav-logo { width: 32px; height: 32px; border-radius: 8px; background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.3); display: flex; align-items: center; justify-content: center; color: var(--color-brand-vibrant); font-size: 0.85rem; flex-shrink: 0; }
.nav-title { display: block; font-size: 0.78rem; font-weight: 600; color: white; letter-spacing: 0.06em; text-transform: uppercase; }
.nav-breadcrumb { display: block; font-size: 0.62rem; color: rgba(216, 180, 254, 0.4); letter-spacing: 0.04em; }
.nav-center { display: flex; align-items: center; justify-content: center; gap: 0.75rem; flex-wrap: wrap; padding: 0.5rem 0; }
.range-label { font-size: 0.6rem; letter-spacing: 0.1em; color: rgba(216, 180, 254, 0.35); font-weight: 500; flex-shrink: 0; }

.range-selector { display: flex; gap: 0.2rem; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(168, 85, 247, 0.15); border-radius: 8px; padding: 0.2rem; }
.range-btn { background: none; border: none; border-radius: 6px; color: rgba(216, 180, 254, 0.45); font-size: 0.65rem; font-weight: 600; letter-spacing: 0.06em; padding: 0.25rem 0.6rem; cursor: pointer; font-family: inherit; transition: all 0.15s; display: flex; align-items: center; gap: 0.3rem; }
.range-btn:hover { color: rgba(216, 180, 254, 0.8); background: rgba(168, 85, 247, 0.1); }
.range-btn--active { background: rgba(168, 85, 247, 0.25); color: var(--color-brand-accent); }
.range-btn--custom { border-left: 1px solid rgba(168, 85, 247, 0.2); margin-left: 0.1rem; padding-left: 0.7rem; }

.date-picker-row { display: flex; align-items: center; gap: 0.5rem; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 8px; padding: 0.3rem 0.6rem; }
.date-input { background: transparent; border: none; color: rgba(216, 180, 254, 0.8); font-size: 0.7rem; font-family: inherit; outline: none; cursor: pointer; width: 110px; }
.date-input::-webkit-calendar-picker-indicator { filter: invert(0.6) sepia(1) saturate(3) hue-rotate(220deg); cursor: pointer; }
.date-sep { font-size: 0.65rem; color: rgba(216, 180, 254, 0.3); }
.date-apply-btn { background: rgba(168, 85, 247, 0.25); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 6px; color: var(--color-brand-accent); font-size: 0.65rem; font-weight: 600; font-family: inherit; padding: 0.2rem 0.6rem; cursor: pointer; transition: all 0.15s; letter-spacing: 0.04em; }
.date-apply-btn:hover:not(:disabled) { background: rgba(168, 85, 247, 0.4); }
.date-apply-btn:disabled { opacity: 0.35; cursor: not-allowed; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease, transform 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; transform: translateY(-4px); }

.live-badge { display: flex; align-items: center; gap: 0.35rem; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em; color: var(--color-brand-success); padding: 0.2rem 0.6rem; border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 20px; background: rgba(74, 222, 128, 0.08); flex-shrink: 0; }
.live-badge--paused { color: rgba(216, 180, 254, 0.4); border-color: rgba(216, 180, 254, 0.15); background: rgba(216, 180, 254, 0.04); }
.live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--color-brand-success); animation: pulse 1.5s ease-in-out infinite; }
.live-badge--paused .live-dot { background: rgba(216, 180, 254, 0.3); animation: none; }

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}

.nav-right { display: flex; align-items: center; justify-content: flex-end; gap: 0.5rem; flex-shrink: 0; }
.refresh-btn, .logout-btn { color: rgba(216, 180, 254, 0.5) !important; width: 30px !important; height: 30px !important; }
.refresh-btn:hover, .logout-btn:hover { color: var(--color-brand-accent) !important; background: rgba(168, 85, 247, 0.1) !important; }
.back-to-app-btn { color: rgba(216, 180, 254, 0.7) !important; font-family: inherit !important; font-size: 0.8rem !important; font-weight: 600 !important; padding: 0.4rem 0.7rem !important; }
.back-to-app-btn:hover { color: var(--color-brand-accent) !important; background: rgba(168, 85, 247, 0.1) !important; }
.user-chip { display: flex; align-items: center; gap: 0.4rem; font-size: 0.68rem; color: rgba(216, 180, 254, 0.5); background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(168, 85, 247, 0.15); border-radius: 20px; padding: 0.2rem 0.65rem; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.dashboard-body { position: relative; z-index: 1; width: 100%; padding: 2rem 2.5rem 4rem; display: flex; flex-direction: column; gap: 1.75rem; }

.fetch-error { display: flex; align-items: center; gap: 0.5rem; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 0.75rem 1rem; font-size: 0.78rem; color: #fca5a5; }
.fetch-error em { font-style: normal; color: #f87171; }
.error-dismiss { margin-left: auto; background: none; border: none; color: #f87171; cursor: pointer; padding: 0; line-height: 1; }

.slide-down-enter-active, .slide-down-leave-active { transition: all 0.25s ease; }
.slide-down-enter-from, .slide-down-leave-to { opacity: 0; transform: translateY(-8px); }

.page-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
.page-title { font-size: 1.375rem; font-weight: 700; color: white; margin: 0 0 0.25rem; letter-spacing: -0.01em; }
.page-subtitle { font-size: 0.72rem; color: rgba(216, 180, 254, 0.4); margin: 0; }
.range-highlight { color: var(--color-brand-accent); font-weight: 600; }
.last-updated { color: rgba(216, 180, 254, 0.7); font-variant-numeric: tabular-nums; }
.header-right { display: flex; align-items: center; gap: 0.75rem; flex-shrink: 0; }

.health-pill { display: flex; align-items: center; gap: 0.4rem; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; padding: 0.3rem 0.875rem; border-radius: 20px; white-space: nowrap; border: 1px solid; }
.health-pill--healthy  { color: var(--color-brand-success); border-color: rgba(74, 222, 128, 0.3); background: rgba(74, 222, 128, 0.08); }
.health-pill--degraded { color: #fbbf24; border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.08); }
.health-pill--error    { color: #f87171; border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.08); }
.health-pill--idle     { color: rgba(216, 180, 254, 0.5); border-color: rgba(168, 85, 247, 0.2); background: rgba(168, 85, 247, 0.06); }
.health-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

.edit-btn { display: flex; align-items: center; gap: 0.4rem; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em; padding: 0.3rem 0.875rem; border-radius: 8px; border: 1px solid rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.08); color: rgba(216, 180, 254, 0.6); cursor: pointer; font-family: inherit; transition: all 0.15s; white-space: nowrap; }
.edit-btn:hover { border-color: rgba(168, 85, 247, 0.6); color: var(--color-brand-accent); background: rgba(168, 85, 247, 0.15); }
.edit-btn--active { border-color: var(--color-brand-success); color: var(--color-brand-success); background: rgba(74, 222, 128, 0.08); }

.kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1rem; }
@media (max-width: 1300px) { .kpi-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 700px)  { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }

.charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media (max-width: 900px) { .charts-grid { grid-template-columns: 1fr; } }

.lower-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; align-items: start; }
@media (max-width: 900px) { .lower-grid { grid-template-columns: 1fr; } }

.chart-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(168, 85, 247, 0.15); border-radius: 14px; padding: 1.5rem; }
.chart-card__header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1.25rem; }
.chart-title { font-size: 0.875rem; font-weight: 600; color: white; margin: 0 0 0.2rem; letter-spacing: 0.02em; }
.chart-subtitle { font-size: 0.68rem; color: rgba(216, 180, 254, 0.4); margin: 0; letter-spacing: 0.04em; }

.header-right-group { display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }

.total-badge-sm { font-size: 0.68rem; background: rgba(168, 85, 247, 0.12); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 20px; padding: 0.2rem 0.6rem; color: var(--color-brand-accent); white-space: nowrap; }

.entropy-badge { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em; padding: 0.2rem 0.65rem; border-radius: 20px; border: 1px solid; white-space: nowrap; }
.entropy-badge--good { color: var(--color-brand-success); border-color: rgba(74, 222, 128, 0.3); background: rgba(74, 222, 128, 0.08); }
.entropy-badge--warn { color: #fbbf24; border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.08); }
.entropy-badge--bad  { color: #f87171; border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.08); }
.entropy-badge--idle { color: rgba(216, 180, 254, 0.4); border-color: rgba(168, 85, 247, 0.2); background: rgba(168, 85, 247, 0.06); }

.chart-empty-sm { height: 220px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.5rem; color: rgba(216, 180, 254, 0.25); font-size: 0.72rem; }
.chart-empty-sm .pi { font-size: 1.75rem; opacity: 0.3; }
.chart-body-entropy { height: 180px; position: relative; }
.chart-body-letters { height: 220px; position: relative; }

/* ── Shared info button + tooltip ───────────────────────────────── */
.chart-info {
  position: relative; cursor: help;
  color: rgba(216, 180, 254, 0.3); font-size: 0.75rem;
  flex-shrink: 0; transition: color 0.2s;
}
.chart-info:hover { color: rgba(216, 180, 254, 0.75); }

.chart-tooltip {
  position: absolute; top: calc(100% + 8px); right: 0;
  width: 260px;
  background: rgba(18, 0, 36, 0.98);
  border: 1px solid rgba(168, 85, 247, 0.35);
  border-radius: 8px; padding: 0.65rem 0.75rem;
  font-size: 0.68rem; color: rgba(216, 180, 254, 0.75);
  line-height: 1.6; z-index: 999; pointer-events: none;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5); white-space: normal;
}

/* ── Drift card ─────────────────────────────────────────────────── */
.drift-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(168, 85, 247, 0.15); border-radius: 14px; padding: 1.5rem; }
.drift-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1.25rem; gap: 0.75rem; }
.drift-title-row { display: flex; align-items: flex-start; gap: 0.5rem; }
.drift-title-with-info {display: flex; align-items: center; gap: 0.4rem; }

.drift-info {
  position: relative; cursor: help;
  color: rgba(216, 180, 254, 0.35); font-size: 0.75rem;
  flex-shrink: 0; transition: color 0.2s;
  display: inline-flex; align-items: center;
}
.drift-info:hover { color: rgba(216, 180, 254, 0.8); }

.drift-tooltip {
  position: absolute; top: calc(100% + 8px); left: 0;
  width: 280px;
  background: rgba(18, 0, 36, 0.98);
  border: 1px solid rgba(168, 85, 247, 0.35);
  border-radius: 10px; padding: 0.75rem;
  z-index: 999; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5); pointer-events: none;
}

.drift-tooltip-row { display: flex; align-items: flex-start; gap: 0.5rem; font-size: 0.68rem; color: rgba(216, 180, 254, 0.7); line-height: 1.5; padding: 0.3rem 0; border-bottom: 1px solid rgba(168, 85, 247, 0.1); }
.drift-tooltip-row:last-child { border-bottom: none; }
.drift-tooltip-row strong { color: white; }

.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 0.3rem; }
.dot--good { background: #4ade80; }
.dot--warn { background: #fbbf24; }
.dot--bad  { background: #f87171; }
.dot--idle { background: rgba(216, 180, 254, 0.3); }

/* ── Status badge with tooltip ──────────────────────────────────── */
.drift-status-wrap { position: relative; align-self: flex-start; flex-shrink: 0; }

.drift-status {
  display: block;
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
  padding: 0.2rem 0.65rem; border-radius: 20px; border: 1px solid;
  white-space: nowrap; cursor: help;
}
.drift-status--good { color: var(--color-brand-success); border-color: rgba(74, 222, 128, 0.3); background: rgba(74, 222, 128, 0.08); }
.drift-status--warn { color: #fbbf24; border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.08); }
.drift-status--bad  { color: #f87171; border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.08); }
.drift-status--idle { color: rgba(216, 180, 254, 0.4); border-color: rgba(168, 85, 247, 0.2); background: rgba(168, 85, 247, 0.06); }

.status-tooltip {
  position: absolute; top: calc(100% + 8px); right: 0;
  width: 300px;
  background: rgba(18, 0, 36, 0.98);
  border: 1px solid rgba(168, 85, 247, 0.35);
  border-radius: 10px; padding: 0.75rem;
  z-index: 999; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5); pointer-events: none;
}

.status-tooltip-row { display: flex; align-items: flex-start; gap: 0.5rem; font-size: 0.68rem; color: rgba(216, 180, 254, 0.5); line-height: 1.5; padding: 0.3rem 0; border-bottom: 1px solid rgba(168, 85, 247, 0.1); transition: color 0.15s; }
.status-tooltip-row:last-of-type { border-bottom: none; }
.status-tooltip-row strong { color: rgba(216, 180, 254, 0.6); }
.row--active { color: rgba(216, 180, 254, 0.9); }
.row--active strong { color: white; }

.status-tooltip-note { margin: 0.5rem 0 0; font-size: 0.63rem; color: rgba(216, 180, 254, 0.3); line-height: 1.5; font-style: italic; }

.drift-no-data { font-size: 0.72rem; color: rgba(216, 180, 254, 0.25); text-align: center; padding: 1rem 0; }

.drift-metric-label { display: flex; justify-content: space-between; font-size: 0.68rem; color: rgba(216, 180, 254, 0.5); margin-bottom: 0.4rem; }
.drift-metric-value { font-weight: 700; color: white; font-variant-numeric: tabular-nums; }

.drift-bar-track { height: 10px; background: rgba(255, 255, 255, 0.07); border-radius: 999px; overflow: hidden; margin-bottom: 0.5rem; }
.drift-bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
.drift-bar-fill--good { background: linear-gradient(90deg, #4ade80, #22c55e); }
.drift-bar-fill--warn { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
.drift-bar-fill--bad  { background: linear-gradient(90deg, #f87171, #ef4444); }

.drift-bar-sublabels { display: flex; justify-content: space-between; font-size: 0.62rem; color: rgba(216, 180, 254, 0.25); margin-top: 0.3rem; margin-bottom: 0.5rem; }

.drift-note { font-size: 0.7rem; color: rgba(216, 180, 254, 0.35); margin: 0.75rem 0 0; line-height: 1.65; }
.drift-note strong { color: rgba(216, 180, 254, 0.6); }

.chart-loading-sm {
  height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  color: rgba(216, 180, 254, 0.4);
  font-size: 0.72rem;
}

.spinner {
  width: 28px;
  height: 28px;
  border: 2px solid rgba(168, 85, 247, 0.15);
  border-top-color: var(--color-brand-accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
