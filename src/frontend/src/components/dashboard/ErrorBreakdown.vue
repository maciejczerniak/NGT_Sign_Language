<template>
  <div class="breakdown-card">
    <div class="card-header">
      <div>
        <h3 class="card-title">Error Breakdown</h3>
        <p class="card-subtitle">by endpoint and status code</p>
      </div>
      <span class="total-badge" :class="totalBadgeClass">
        {{ totalErrors }} total
      </span>
    </div>

    <!-- Empty state -->
    <div v-if="breakdown.length === 0" class="empty-state">
      <i class="pi pi-check-circle" />
      <span>No errors recorded — system healthy</span>
    </div>

    <!-- Table -->
    <div v-else class="table-scroll">
    <table class="breakdown-table">
      <thead>
        <tr>
          <th>Endpoint</th>
          <th>Status</th>
          <th>Type</th>
          <th class="col-count">Count</th>
          <th class="col-bar">Distribution</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, i) in breakdown" :key="i" class="breakdown-row">
          <td class="col-path">
            <span class="path-text">{{ row.path }}</span>
          </td>
          <td class="col-status">
            <span class="status-badge" :class="statusClass(row.status_code)">
              {{ row.status_code }}
            </span>
          </td>
          <td class="col-type">
            <span class="type-label">{{ errorType(row.path, row.status_code) }}</span>
          </td>
          <td class="col-count">
            <span class="count-value" :class="countClass(row.count)">{{ row.count }}</span>
          </td>
          <td class="col-bar">
            <div class="bar-track">
              <div
                class="bar-fill"
                :class="statusClass(row.status_code)"
                :style="{ width: `${(row.count / maxCount) * 100}%` }"
              />
            </div>
          </td>
        </tr>
      </tbody>
    </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface ErrorRow {
  path: string
  status_code: number
  count: number
}

const props = defineProps<{
  breakdown: ErrorRow[]
  totalErrors: number
}>()

const maxCount = computed(() =>
  props.breakdown.length > 0 ? Math.max(...props.breakdown.map((r) => r.count)) : 1
)

const totalBadgeClass = computed(() => {
  if (props.totalErrors === 0) return 'total-badge--ok'
  if (props.totalErrors < 10) return 'total-badge--warn'
  return 'total-badge--danger'
})

function statusClass(code: number): string {
  if (code >= 500) return 'status--5xx'
  if (code >= 400) return 'status--4xx'
  return 'status--ok'
}

function countClass(count: number): string {
  if (count >= 50) return 'count--high'
  if (count >= 10) return 'count--mid'
  return 'count--low'
}

function errorType(path: string, code: number): string {
  if (path.includes('auth/jwt/login') && code === 400) return 'Bad credentials'
  if (path.includes('auth/jwt/login') && code === 401) return 'Invalid token'
  if (path.includes('auth') && code === 401) return 'Unauthorized'
  if (path.includes('auth') && code === 422) return 'Validation error'
  if (path.includes('predict') && code >= 500) return 'Inference failure'
  if (path.includes('predict') && code === 400) return 'Bad image input'
  if (path.includes('admin') && code === 401) return 'Missing token'
  if (path.includes('admin') && code === 403) return 'Access denied'
  if (code === 404) return 'Not found'
  if (code === 422) return 'Validation error'
  if (code >= 500) return 'Server error'
  return `HTTP ${code}`
}
</script>

<style scoped>
.breakdown-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(168, 85, 247, 0.15);
  border-radius: 14px;
  padding: 1.5rem;
}

.table-scroll {
  max-height: 280px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba(168, 85, 247, 0.3) transparent;
}

.table-scroll::-webkit-scrollbar {
  width: 4px;
}

.table-scroll::-webkit-scrollbar-track {
  background: transparent;
}

.table-scroll::-webkit-scrollbar-thumb {
  background: rgba(168, 85, 247, 0.3);
  border-radius: 999px;
}

.card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 1.25rem;
}

.card-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: white;
  margin: 0 0 0.2rem;
  letter-spacing: 0.02em;
}

.card-subtitle {
  font-size: 0.68rem;
  color: rgba(216, 180, 254, 0.4);
  margin: 0;
  letter-spacing: 0.04em;
}

.total-badge {
  font-size: 0.68rem;
  border-radius: 20px;
  padding: 0.2rem 0.65rem;
  border: 1px solid;
  font-weight: 600;
  white-space: nowrap;
  align-self: flex-start;
}

.total-badge--ok     { color: var(--color-brand-success); border-color: rgba(74, 222, 128, 0.3); background: rgba(74, 222, 128, 0.08); }
.total-badge--warn   { color: #fbbf24; border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.08); }
.total-badge--danger { color: #f87171; border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.08); }

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 2.5rem;
  color: var(--color-brand-success);
  font-size: 0.78rem;
  opacity: 0.7;
}

.breakdown-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
}

thead th {
  text-align: left;
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: rgba(216, 180, 254, 0.35);
  padding: 0 0.75rem 0.75rem;
  font-weight: 500;
  border-bottom: 1px solid rgba(168, 85, 247, 0.1);
}

.breakdown-row td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid rgba(168, 85, 247, 0.06);
  vertical-align: middle;
}

.breakdown-row:last-child td {
  border-bottom: none;
}

.breakdown-row:hover td {
  background: rgba(168, 85, 247, 0.04);
}

.col-path { width: 35%; }
.col-status { width: 10%; }
.col-type { width: 25%; }
.col-count { width: 10%; text-align: right; }
.col-bar { width: 20%; }

.path-text {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: rgba(216, 180, 254, 0.7);
}

.status-badge {
  display: inline-block;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 700;
  font-family: 'DM Mono', monospace;
}

.status--5xx { background: rgba(239, 68, 68, 0.15); color: #f87171; }
.status--4xx { background: rgba(251, 191, 36, 0.15); color: #fbbf24; }
.status--ok  { background: rgba(74, 222, 128, 0.15); color: #4ade80; }

.type-label {
  color: rgba(216, 180, 254, 0.5);
  font-size: 0.72rem;
}

.count-value {
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  font-size: 0.875rem;
}

.count--high { color: #f87171; }
.count--mid  { color: #fbbf24; }
.count--low  { color: rgba(216, 180, 254, 0.7); }

.bar-track {
  height: 6px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 999px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.4s ease;
}

.bar-fill.status--5xx { background: rgba(239, 68, 68, 0.6); }
.bar-fill.status--4xx { background: rgba(251, 191, 36, 0.6); }
.bar-fill.status--ok  { background: rgba(74, 222, 128, 0.6); }
</style>
