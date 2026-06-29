import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api } from '../composables/useApi'

interface User {
  id: string
  email: string
  is_superuser: boolean
}

// Auth store: handles login, registration, and logout for all users.
// Any registered user can log in — admin status (is_superuser) only gates
// admin-only features like the dashboard, never login itself. The JWT is
// persisted in localStorage so a refresh keeps the user signed in.
export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(localStorage.getItem('auth_token'))
  const user = ref<User | null>(null)
  // Guest mode is session-only (not persisted): reopening the app returns
  // a guest to the landing page.
  const guestMode = ref<boolean>(false)

  const isAuthenticated = computed(() => !!token.value)
  const isAdmin = computed(() => user.value?.is_superuser === true)

  // True when the user chose to continue without an account.
  const isGuest = computed(() => guestMode.value)

  // ── Token persistence ────────────────────────────────────────────
  function setToken(t: string): void {
    token.value = t
    localStorage.setItem('auth_token', t)
    // Logging in supersedes guest mode.
    guestMode.value = false
  }

  function clearAuth(): void {
    token.value = null
    user.value = null
    localStorage.removeItem('auth_token')
  }

  // ── Current user ─────────────────────────────────────────────────
  // Loads the signed-in user from the API. Only clears auth on a genuine
  // auth failure (401/403) — i.e. a missing, invalid, or expired token.
  // Transient errors (network blip, brief server outage, 5xx) leave the
  // session intact and simply return false, so the user is not logged out
  // by a temporary problem. Returns true if the user loaded.
  async function fetchCurrentUser(): Promise<boolean> {
    if (!token.value) return false
    try {
      const res = await api.get<User>('/users/me')
      user.value = res.data
      return true
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status
      if (status === 401 || status === 403) {
        clearAuth()
      }
      return false
    }
  }

  // ── Login ────────────────────────────────────────────────────────
  // Logs in any registered user regardless of admin status. Stores the JWT
  // and loads the profile. Throws if credentials are invalid.
  async function login(email: string, password: string): Promise<void> {
    const params = new URLSearchParams()
    params.append('username', email)
    params.append('password', password)

    const res = await api.post<{ access_token: string }>('/auth/jwt/login', params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })

    setToken(res.data.access_token)

    const ok = await fetchCurrentUser()
    if (!ok) {
      clearAuth()
      throw new Error('Could not load your account. Please try again.')
    }
  }

  // ── Register ─────────────────────────────────────────────────────
  // Creates a new account, then logs in automatically so the user is
  // signed in immediately. Throws if the email is already in use.
  async function register(email: string, password: string): Promise<void> {
    await api.post('/auth/register', { email, password })
    await login(email, password)
  }

  // ── Guest mode ───────────────────────────────────────────────────
  // Let the user into the app without an account, for this session only.
  // Not persisted — reopening the app returns them to the landing page.
  function continueAsGuest(): void {
    guestMode.value = true
  }

  function clearGuestMode(): void {
    guestMode.value = false
  }

  // ── Logout ───────────────────────────────────────────────────────
  function logout(): void {
    clearAuth()
    clearGuestMode()
  }

  return {
    token,
    user,
    isAuthenticated,
    isAdmin,
    isGuest,
    login,
    register,
    logout,
    continueAsGuest,
    clearGuestMode,
    fetchCurrentUser,
  }
})
