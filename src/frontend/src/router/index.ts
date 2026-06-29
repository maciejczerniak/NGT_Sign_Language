import { createRouter, createWebHistory } from 'vue-router';
import { useAuthStore } from '@/stores/auth';

const router = createRouter({
  // createWebHistory uses clean URLs like /play instead of /#/play
  history: createWebHistory(),
  routes: [
    {
      path: '/welcome',
      name: 'welcome',
      component: () => import('@/views/LandingView.vue'),
      meta: { requiresAuth: false },
    },
    {
      path: '/',
      name: 'home',
      component: () => import('@/views/HomeView.vue'),
    },
    {
      path: '/learn',
      name: 'learn',
      component: () => import('@/views/LearnView.vue'),
    },
    {
      path: '/play',
      name: 'play',
      component: () => import('@/views/PlayView.vue'),
    },
    {
      path: '/play/random-letters',
      name: 'random-letters',
      component: () => import('@/views/LevelEasyGame.vue'),
    },
    {
      path: '/collect',
      name: 'collect',
      component: () => import('@/views/CollectView.vue'),
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { requiresAuth: false },
    },
    {
      path: '/register',
      name: 'register',
      component: () => import('@/views/RegisterView.vue'),
      meta: { requiresAuth: false },
    },
    {
      path: '/admin/dashboard',
      name: 'dashboard',
      component: () => import('@/views/DashboardView.vue'),
      meta: { requiresAuth: true, requiresAdmin: true },
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/',
    },
  ],
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()

  // Pages anyone can see without being signed in or choosing guest mode
  // (the landing page and the auth pages themselves).
  const publicNames = ['welcome', 'login', 'register']
  if (publicNames.includes(to.name as string)) return true

  // Fresh visitor — not signed in and hasn't chosen guest mode → front door.
  if (!auth.isAuthenticated && !auth.isGuest) {
    return { name: 'welcome' }
  }

  // Admin-only routes: must be a signed-in admin (guests never qualify).
  if (to.meta.requiresAdmin) {
    // Load the profile if a token exists but it hasn't been fetched yet
    // (e.g. after a hard refresh) so the admin check is reliable.
    if (auth.isAuthenticated && !auth.user) {
      await auth.fetchCurrentUser()
    }
    if (!auth.isAuthenticated) {
      return { name: 'login', query: { redirect: to.fullPath } }
    }
    if (!auth.isAdmin) {
      return { name: 'home' }
    }
  }

  return true
})

export default router;
