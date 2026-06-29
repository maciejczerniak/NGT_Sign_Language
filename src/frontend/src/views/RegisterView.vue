<template>
  <div class="register-root">
    <div class="grid-overlay" aria-hidden="true" />

    <div class="register-card">
      <button class="back-btn" @click="router.push('/welcome')" aria-label="Back to start">
        <i class="pi pi-arrow-left" />
      </button>
      <div class="card-header">
        <div class="logo-ring">🐙</div>
        <h1 class="card-title">SignSee</h1>
        <p class="card-subtitle">Create an account to get started</p>
      </div>

      <form class="register-form" @submit.prevent="handleRegister">
        <div class="field-group">
          <label for="email" class="field-label">Email</label>
          <InputText
            id="email"
            v-model="email"
            type="email"
            placeholder="you@example.com"
            autocomplete="username"
            :disabled="loading"
            class="field-input"
          />
        </div>

        <div class="field-group">
          <label for="password" class="field-label">Password</label>
          <Password
            id="password"
            v-model="password"
            placeholder="At least 8 characters"
            :feedback="false"
            toggle-mask
            autocomplete="new-password"
            :disabled="loading"
            input-class="field-input"
          />
        </div>

        <div class="field-group">
          <label for="confirm" class="field-label">Confirm password</label>
          <Password
            id="confirm"
            v-model="confirmPassword"
            placeholder="Re-enter your password"
            :feedback="false"
            toggle-mask
            autocomplete="new-password"
            :disabled="loading"
            input-class="field-input"
          />
        </div>

        <div v-if="errorMsg" class="error-banner" role="alert">
          <i class="pi pi-exclamation-triangle" />
          <span>{{ errorMsg }}</span>
        </div>

        <Button
          type="submit"
          label="Create account"
          icon="pi pi-arrow-right"
          icon-pos="right"
          :loading="loading"
          class="submit-btn"
        />
      </form>

      <p class="login-note">
        Already have an account?
        <router-link to="/login" class="login-link">Log in</router-link>
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import InputText from 'primevue/inputtext'
import Password from 'primevue/password'
import Button from 'primevue/button'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const toast = useToast()
const auth = useAuthStore()

const email = ref('')
const password = ref('')
const confirmPassword = ref('')
const loading = ref(false)
const errorMsg = ref('')

// Validate inputs, register, then (auto-logged-in) send the user home.
async function handleRegister() {
  if (!email.value || !password.value || !confirmPassword.value) {
    errorMsg.value = 'Please fill in all fields.'
    return
  }
  if (password.value.length < 8) {
    errorMsg.value = 'Password must be at least 8 characters.'
    return
  }
  if (password.value !== confirmPassword.value) {
    errorMsg.value = 'Passwords do not match.'
    return
  }

  errorMsg.value = ''
  loading.value = true
  try {
    await auth.register(email.value, password.value)
    router.push('/')
  } catch (err: unknown) {
    errorMsg.value = registerErrorMessage(err)
    toast.add({
      severity: 'error',
      summary: 'Registration failed',
      detail: errorMsg.value,
      life: 4000,
    })
  } finally {
    loading.value = false
  }
}

// Turn a backend/registration error into a friendly, specific message.
function registerErrorMessage(err: unknown): string {
  // axios attaches the server response under err.response
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail

  if (typeof detail === 'string') {
    if (detail === 'REGISTER_USER_ALREADY_EXISTS') {
      return 'This email is already registered. Try logging in instead.'
    }
    // FastAPI-Users can also return a password-policy message here
    if (detail.toLowerCase().includes('password')) {
      return detail
    }
  }
  // Validation errors come back as an array of issues
  if (Array.isArray(detail) && detail.length > 0) {
    return 'Please check your email and password and try again.'
  }
  return 'Could not create your account. Please try again.'
}
</script>

<style scoped>
.register-root {
  width: 100%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--color-brand-purple);
  position: relative;
  overflow: hidden;
}

.grid-overlay {
  position: absolute;
  inset: 0;
  background-image: radial-gradient(circle, rgba(168, 85, 247, 0.12) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none;
}

.register-root::before {
  content: '';
  position: absolute;
  width: 600px; height: 600px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.18) 0%, transparent 70%);
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
}

.register-card {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 420px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(168, 85, 247, 0.25);
  border-radius: 16px;
  padding: 2.5rem 2rem;
  backdrop-filter: blur(12px);
  box-shadow:
    0 0 0 1px rgba(168, 85, 247, 0.08),
    0 24px 64px rgba(0, 0, 0, 0.5);
}

.back-btn {
  position: absolute;
  top: 1rem;
  left: 1rem;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(168, 85, 247, 0.3);
  color: var(--color-brand-accent);
  cursor: pointer;
  transition: all 0.2s;
}
.back-btn:hover {
  background: rgba(168, 85, 247, 0.15);
  border-color: var(--color-brand-vibrant);
}

.card-header {
  text-align: center;
  margin-bottom: 2rem;
}

.logo-ring {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px; height: 56px;
  border-radius: 16px;
  border: 1.5px solid rgba(168, 85, 247, 0.5);
  background: rgba(168, 85, 247, 0.1);
  margin-bottom: 1rem;
  font-size: 1.6rem;
}

.card-title {
  font-size: 1.5rem;
  font-weight: 800;
  color: white;
  margin: 0 0 0.25rem;
  letter-spacing: -0.01em;
}

.card-subtitle {
  font-size: 0.85rem;
  color: var(--color-brand-accent);
  margin: 0;
}

.register-form {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.field-group {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.field-label {
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  color: var(--color-brand-accent);
  font-weight: 600;
}

:deep(.field-input),
:deep(.p-password-input) {
  width: 100%;
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(168, 85, 247, 0.3) !important;
  border-radius: 8px !important;
  color: white !important;
  font-family: inherit !important;
  font-size: 0.9rem !important;
  padding: 0.65rem 0.875rem !important;
  transition: border-color 0.2s;
}

:deep(.field-input:focus),
:deep(.p-password-input:focus) {
  border-color: var(--color-brand-vibrant) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2) !important;
}

:deep(.p-password) { width: 100%; }

.error-banner {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  padding: 0.6rem 0.875rem;
  font-size: 0.8rem;
  color: #fca5a5;
}

.submit-btn {
  width: 100% !important;
  justify-content: center !important;
  background: var(--color-brand-vibrant) !important;
  border-color: var(--color-brand-vibrant) !important;
  border-radius: 8px !important;
  font-family: inherit !important;
  letter-spacing: 0.02em !important;
  padding: 0.7rem !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
  transition: all 0.2s !important;
}

.submit-btn:hover:not(:disabled) {
  background: #9333ea !important;
  border-color: #9333ea !important;
  box-shadow: 0 0 20px rgba(168, 85, 247, 0.4) !important;
}

.login-note {
  text-align: center;
  margin-top: 1.5rem;
  font-size: 0.85rem;
  color: var(--color-brand-accent);
}

.login-link {
  color: var(--color-brand-vibrant);
  font-weight: 600;
  text-decoration: none;
}

.login-link:hover {
  text-decoration: underline;
}
</style>
