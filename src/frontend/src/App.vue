<template>
  <!-- Chromeless routes (landing, auth, admin) get a clean full-page wrapper -->
  <div v-if="isChromeless" class="min-h-screen bg-brand-purple text-white overflow-y-auto">
    <RouterView />
    <Toast position="top-right" />
  </div>

  <!-- Main app routes keep the header + flex layout -->
  <div v-else class="h-screen w-screen flex flex-col overflow-hidden bg-brand-purple text-white">
    <header class="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-black/10">
      <div class="flex items-center gap-3">
        <button class="flex items-center gap-3 bg-transparent border-none cursor-pointer text-white" @click="goHome">
          <div class="w-8 h-8 bg-brand-vibrant rounded-lg flex items-center justify-center text-lg">
            🐙
          </div>
          <h1 class="font-bold text-lg tracking-tight">SignSee</h1>
        </button>
      </div>
      <div class="flex items-center gap-4">
        <template v-if="route.name !== 'home'">
          <Button label="Learn" icon="pi pi-book" severity="help" size="small" @click="router.push('/learn')" />
          <Button label="Play Game" icon="pi pi-play" severity="help" size="small" @click="router.push('/play')" />
        </template>

        <!-- User menu: contents adapt to auth state -->
        <Button
          icon="pi pi-user"
          rounded
          severity="help"
          aria-haspopup="true"
          aria-controls="user-menu"
          @click="toggleMenu"
        />
        <Menu id="user-menu" ref="userMenu" :model="menuItems" :popup="true" />
      </div>
    </header>
    <RouterView class="flex flex-1 overflow-hidden" />
    <Toast position="top-right" />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterView, useRoute, useRouter } from 'vue-router'
import Button from 'primevue/button'
import Menu from 'primevue/menu'
import Toast from 'primevue/toast'
import type { MenuItem } from 'primevue/menuitem'
import { useAuthStore } from './stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

// Pages that should render without the app header (landing, auth, admin).
const isChromeless = computed(() => {
  const chromelessNames = ['welcome', 'login', 'register']
  return chromelessNames.includes(route.name as string) || route.path.startsWith('/admin')
})

// Logo click: into the app if signed in or browsing as guest, else to the front door.
function goHome() {
  if (auth.isAuthenticated || auth.isGuest) {
    router.push('/')
  } else {
    router.push('/welcome')
  }
}

// ── User dropdown menu ───────────────────────────────────────────
const userMenu = ref()

function toggleMenu(event: Event) {
  userMenu.value.toggle(event)
}

// Menu items depend on whether the user is signed in.
const menuItems = computed<MenuItem[]>(() => {
  if (auth.isAuthenticated) {
    const items: MenuItem[] = [
      { label: auth.user?.email ?? 'Account', disabled: true },
      { separator: true },
    ]
    // Admins get a link to the monitoring dashboard.
    if (auth.isAdmin) {
      items.push({
        label: 'Dashboard',
        icon: 'pi pi-chart-line',
        command: () => router.push('/admin/dashboard'),
      })
    }
    items.push({
      label: 'Log out',
      icon: 'pi pi-sign-out',
      command: () => {
        auth.logout()
        router.push('/welcome')
      },
    })
    return items
  }

  // Guest or not signed in: offer ways to authenticate.
  return [
    { label: 'Log in', icon: 'pi pi-sign-in', command: () => router.push('/login') },
    { label: 'Create account', icon: 'pi pi-user-plus', command: () => router.push('/register') },
  ]
})
</script>

<style scoped>
.animate-fadein {
  animation: fadeIn 0.3s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
