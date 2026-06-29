<template>
  <div class="landing-root">
    <!-- Ambient glows -->
    <div class="glow glow-left" aria-hidden="true" />
    <div class="glow glow-right" aria-hidden="true" />

    <!-- Scattered background letters -->
    <div class="letters" aria-hidden="true">
      <span class="letter letter-1">S</span>
      <span class="letter letter-2">G</span>
      <span class="letter letter-3">N</span>
      <span class="letter letter-4">I</span>
      <span class="letter letter-5">A</span>
      <span class="letter letter-6">K</span>
    </div>

    <div class="landing-content">
      <!-- Otto -->
      <div class="otto-wrap">
        <div class="otto-shadow" />
        <img
          src="/Otto_1.png"
          alt="Otto the signing octopus"
          class="otto"
        />
      </div>

      <!-- Branding -->
      <h1 class="brand-title">SignSee</h1>
      <div class="brand-sub">
        <span class="divider-line" />
        <span class="brand-sub-text">NGT Fingerspelling</span>
        <span class="divider-line" />
      </div>
      <p class="tagline">Learn Dutch sign language, one sign at a time</p>

      <!-- Actions -->
      <div class="actions">
        <Button
          label="Create account"
          icon="pi pi-user-plus"
          class="action-btn action-primary"
          @click="router.push('/register')"
        />
        <Button
          label="Log in"
          icon="pi pi-sign-in"
          class="action-btn action-secondary"
          @click="router.push('/login')"
        />
        <button class="guest-link" @click="enterAsGuest">
          Continue without an account
          <i class="pi pi-arrow-right" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import Button from 'primevue/button'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const auth = useAuthStore()

// Mark the session as guest, then enter the app. Progress won't be saved.
function enterAsGuest() {
  auth.continueAsGuest()
  router.push('/')
}
</script>

<style scoped>
.landing-root {
  width: 100%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--color-brand-purple);
  position: relative;
  overflow: hidden;
}

.glow {
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
  filter: blur(120px);
}
.glow-left {
  width: 600px; height: 600px;
  background: rgba(168, 85, 247, 0.12);
  top: 50%; left: 25%;
  transform: translate(-50%, -50%);
}
.glow-right {
  width: 400px; height: 400px;
  background: rgba(168, 85, 247, 0.06);
  top: 50%; right: 25%;
  transform: translate(50%, -50%);
}

.letters {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
  user-select: none;
}
.letter {
  position: absolute;
  font-weight: 900;
  color: rgba(168, 85, 247, 0.06);
  line-height: 1;
}
.letter-1 { font-size: 14rem; top: -3%; left: -2%; transform: rotate(-15deg); }
.letter-2 { font-size: 11rem; top: 45%; left: -1%; transform: rotate(8deg); }
.letter-3 { font-size: 9rem; bottom: 0%; left: 4%; transform: rotate(12deg); }
.letter-4 { font-size: 8rem; top: 8%; right: 6%; transform: rotate(-10deg); }
.letter-5 { font-size: 8rem; bottom: 12%; right: 4%; transform: rotate(15deg); }
.letter-6 { font-size: 6rem; top: 55%; right: 16%; transform: rotate(-8deg); }

.landing-content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 2rem;
}

.otto-wrap {
  position: relative;
  margin-bottom: 0.5rem;
}
.otto {
  width: 280px;
  height: 280px;
  object-fit: contain;
  position: relative;
  z-index: 1;
  filter: drop-shadow(0 20px 40px rgba(0, 0, 0, 0.4));
  animation: ottoFloat 3s ease-in-out infinite;
}
.otto-shadow {
  position: absolute;
  bottom: 10px; left: 50%;
  transform: translateX(-50%);
  width: 180px; height: 24px;
  background: rgba(168, 85, 247, 0.25);
  filter: blur(24px);
  border-radius: 50%;
}
@keyframes ottoFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-16px); }
}

.brand-title {
  font-size: 4rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0;
  background: linear-gradient(135deg, #ffffff 0%, #d8b4fe 50%, #a855f7 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  padding-bottom: 4px;
}

.brand-sub {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  margin-top: 0.5rem;
}
.divider-line {
  height: 1px;
  width: 2rem;
  background: rgba(168, 85, 247, 0.4);
}
.brand-sub-text {
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: rgba(216, 180, 254, 0.6);
}

.tagline {
  font-size: 0.95rem;
  color: rgba(216, 180, 254, 0.5);
  margin: 0.75rem 0 0;
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 0.875rem;
  margin-top: 2.5rem;
  width: 280px;
}

.action-btn {
  width: 100% !important;
  justify-content: center !important;
  border-radius: 10px !important;
  font-family: inherit !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  padding: 0.8rem !important;
  transition: all 0.2s !important;
}

.action-primary {
  background: var(--color-brand-vibrant) !important;
  border-color: var(--color-brand-vibrant) !important;
}
.action-primary:hover {
  background: #9333ea !important;
  border-color: #9333ea !important;
  box-shadow: 0 0 24px rgba(168, 85, 247, 0.45) !important;
}

.action-secondary {
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(168, 85, 247, 0.4) !important;
  color: white !important;
}
.action-secondary:hover {
  background: rgba(168, 85, 247, 0.15) !important;
  border-color: var(--color-brand-vibrant) !important;
}

.guest-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  margin-top: 0.5rem;
  background: none;
  border: none;
  cursor: pointer;
  color: rgba(216, 180, 254, 0.55);
  font-family: inherit;
  font-size: 0.85rem;
  transition: color 0.2s;
}
.guest-link:hover {
  color: var(--color-brand-accent);
}
.guest-link .pi {
  font-size: 0.7rem;
}
</style>
