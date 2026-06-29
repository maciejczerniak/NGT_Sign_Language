<template>
  <div class="metric-card" :class="{ 'metric-card--alert': isAlert }">
    <div class="metric-header">
      <span class="metric-label">{{ label }}</span>
      <div class="metric-header-right">
        <div v-if="description" class="info-icon" @mouseenter="showTip = true" @mouseleave="showTip = false">
          <i class="pi pi-info-circle" />
          <div v-if="showTip" class="tooltip">{{ description }}</div>
        </div>
        <div class="metric-icon" :class="`metric-icon--${variant ?? 'default'}`">
          <i :class="`pi ${icon}`" />
        </div>
      </div>
    </div>

    <div class="metric-value">
      <span v-if="noData" class="value-nodata">—</span>
      <template v-else>
        <span class="value-number">{{ formattedValue }}</span>
        <span v-if="unit" class="value-unit">{{ unit }}</span>
      </template>
    </div>

    <div v-if="subtext" class="metric-subtext">{{ subtext }}</div>

    <div v-if="isAlert && !noData" class="alert-indicator">
      <i class="pi pi-exclamation-triangle" />
      {{ higherIsBetter ? 'Below threshold' : 'Above threshold' }}
    </div>

    <!-- Threshold editor — shown in edit mode -->
    <div v-if="editMode && alertThreshold !== undefined" class="threshold-editor">
      <span class="threshold-label">Alert threshold</span>
      <div class="threshold-input-row">
        <input
          v-model.number="localThreshold"
          type="number"
          class="threshold-input"
          :min="0"
          @keydown.enter="saveThreshold"
        />
        <span class="threshold-unit">{{ unit ?? '' }}</span>
        <button class="threshold-save" @click="saveThreshold">
          <i class="pi pi-check" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const props = defineProps<{
  label: string
  value: number
  unit?: string
  icon: string
  variant?: 'default' | 'success' | 'warning' | 'danger'
  subtext?: string
  description?: string
  alertThreshold?: number
  higherIsBetter?: boolean
  noData?: boolean
  editMode?: boolean
}>()

const emit = defineEmits<{
  (e: 'threshold-change', value: number): void
}>()

const showTip = ref(false)
const localThreshold = ref(props.alertThreshold ?? 0)

watch(() => props.alertThreshold, (v) => {
  if (v !== undefined) localThreshold.value = v
})

function saveThreshold() {
  emit('threshold-change', localThreshold.value)
}

const isAlert = computed(() => {
  if (props.noData) return false
  if (props.alertThreshold === undefined) return false
  return props.higherIsBetter
    ? props.value < props.alertThreshold
    : props.value > props.alertThreshold
})

const formattedValue = computed(() => {
  if (props.noData) return '—'
  if (props.value === 0 && props.unit === '%') return '0.00'
  if (Number.isInteger(props.value)) return props.value.toLocaleString()
  return props.value.toFixed(2)
})
</script>

<style scoped>
.metric-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(168, 85, 247, 0.15);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  position: relative;
  overflow: visible;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, rgba(168, 85, 247, 0.5), transparent);
  border-radius: 12px 12px 0 0;
}

.metric-card:hover {
  border-color: rgba(168, 85, 247, 0.3);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.metric-card--alert { border-color: rgba(239, 68, 68, 0.4) !important; }
.metric-card--alert::before { background: linear-gradient(90deg, rgba(239, 68, 68, 0.7), transparent); }

.metric-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.875rem;
}

.metric-header-right {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.metric-label {
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: rgba(216, 180, 254, 0.5);
  font-weight: 500;
}

.info-icon {
  position: relative;
  cursor: help;
  color: rgba(216, 180, 254, 0.3);
  font-size: 0.72rem;
  display: flex;
  align-items: center;
  transition: color 0.2s;
}

.info-icon:hover { color: rgba(216, 180, 254, 0.75); }

.tooltip {
  position: absolute;
  bottom: calc(100% + 8px);
  right: 0;
  width: 220px;
  background: rgba(18, 0, 36, 0.98);
  border: 1px solid rgba(168, 85, 247, 0.35);
  border-radius: 8px;
  padding: 0.6rem 0.75rem;
  font-size: 0.68rem;
  color: rgba(216, 180, 254, 0.8);
  line-height: 1.55;
  z-index: 999;
  pointer-events: none;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  white-space: normal;
  text-transform: none;
  letter-spacing: 0;
  font-weight: 400;
}

.tooltip::after {
  content: '';
  position: absolute;
  top: 100%; right: 6px;
  border: 5px solid transparent;
  border-top-color: rgba(168, 85, 247, 0.35);
}

.metric-icon {
  width: 30px; height: 30px;
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
}

.metric-icon--default { background: rgba(168, 85, 247, 0.12); color: var(--color-brand-vibrant); }
.metric-icon--success { background: rgba(74, 222, 128, 0.12); color: var(--color-brand-success); }
.metric-icon--warning { background: rgba(251, 191, 36, 0.12); color: #fbbf24; }
.metric-icon--danger  { background: rgba(239, 68, 68, 0.12);  color: #f87171; }

.metric-value {
  display: flex;
  align-items: baseline;
  gap: 0.3rem;
  margin-bottom: 0.25rem;
}

.value-number {
  font-size: 1.875rem;
  font-weight: 700;
  color: white;
  line-height: 1;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}

.value-nodata {
  font-size: 1.875rem;
  font-weight: 300;
  color: rgba(216, 180, 254, 0.25);
  line-height: 1;
}

.value-unit {
  font-size: 0.8rem;
  color: rgba(216, 180, 254, 0.5);
  font-weight: 400;
}

.metric-subtext {
  font-size: 0.68rem;
  color: rgba(216, 180, 254, 0.35);
  margin-top: 0.2rem;
}

.alert-indicator {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.65rem;
  color: #f87171;
  margin-top: 0.5rem;
  letter-spacing: 0.04em;
}

/* ── Threshold editor ───────────────────────────────────────────── */
.threshold-editor {
  margin-top: 0.875rem;
  padding-top: 0.875rem;
  border-top: 1px solid rgba(168, 85, 247, 0.15);
}

.threshold-label {
  display: block;
  font-size: 0.6rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(216, 180, 254, 0.35);
  margin-bottom: 0.4rem;
}

.threshold-input-row {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.threshold-input {
  flex: 1;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(168, 85, 247, 0.35);
  border-radius: 6px;
  color: white;
  font-size: 0.78rem;
  font-family: inherit;
  padding: 0.3rem 0.5rem;
  outline: none;
  font-variant-numeric: tabular-nums;
  min-width: 0;
  transition: border-color 0.2s;
}

.threshold-input:focus {
  border-color: var(--color-brand-vibrant);
  box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.15);
}

/* Remove number input arrows */
.threshold-input::-webkit-outer-spin-button,
.threshold-input::-webkit-inner-spin-button { -webkit-appearance: none; }
.threshold-input[type=number] { -moz-appearance: textfield; }

.threshold-unit {
  font-size: 0.68rem;
  color: rgba(216, 180, 254, 0.4);
  min-width: 1rem;
}

.threshold-save {
  background: rgba(168, 85, 247, 0.2);
  border: 1px solid rgba(168, 85, 247, 0.4);
  border-radius: 6px;
  color: var(--color-brand-accent);
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 0.65rem;
  transition: all 0.15s;
  flex-shrink: 0;
}

.threshold-save:hover {
  background: rgba(168, 85, 247, 0.35);
  border-color: var(--color-brand-vibrant);
}
</style>
