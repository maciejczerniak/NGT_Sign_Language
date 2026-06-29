<script setup lang="ts">
import { useRouter } from 'vue-router';
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue';
import { useAuthStore } from '../stores/auth';
import { api } from '../composables/useApi';

const router = useRouter();
const auth = useAuthStore();

// Guests browse without an account — progress isn't tracked or shown.
const isGuest = computed(() => auth.isGuest);

// Progress shown on the home page. Starts at zero and is filled from the
// stats API for signed-in users (see fetchStats). Guests never fetch — their
// progress isn't tracked, so the zero values are simply never displayed
// (the whole progress UI is gated behind v-if="!isGuest").
interface HomeStats {
  streak: number;
  lettersLearned: number;
  totalPoints: number;
  lastPlayed: string | null;
  level: number;
  levelName: string;
  practicedToday: number;
  dailyGoal: number;
}
const stats = reactive<HomeStats>({
  streak: 0,
  lettersLearned: 0,
  totalPoints: 0,
  lastPlayed: null,
  level: 1,
  levelName: 'Beginner',
  practicedToday: 0,
  dailyGoal: 5,
});

// Daily-goal display: bar width caps at 100% even if the user overshoots
// (e.g. 6/5), while the text still shows the real count. goalReached drives
// the celebratory state and confetti.
const dailyGoalPercent = computed(() =>
  Math.min((stats.practicedToday / stats.dailyGoal) * 100, 100)
);
const goalReached = computed(() => stats.practicedToday >= stats.dailyGoal);

// Pull the signed-in user's real stats from the backend. On any failure we
// keep the zero values rather than blocking the page — the dashboard simply
// shows a fresh-start state.
async function fetchStats(): Promise<void> {
  try {
    const res = await api.get('/stats');
    stats.streak = res.data.streak ?? 0;
    stats.lettersLearned = res.data.letters_learned ?? 0;
    stats.totalPoints = res.data.points ?? 0;
    stats.lastPlayed = res.data.last_played ?? null;
    stats.level = res.data.level ?? 1;
    stats.levelName = res.data.level_name ?? 'Beginner';
    stats.practicedToday = res.data.practiced_today ?? 0;
    stats.dailyGoal = res.data.daily_goal ?? 5;
  } catch {
    // Leave zeros in place; a transient error shouldn't break the home page.
  }
}

// Per-letter practice progress, shown as a list on the home page. Teaches the
// "sign a letter 3x to learn it" rule by simply showing each letter filling up
// — no separate instructions needed. Only fetched for signed-in users.
interface LetterProgress {
  letter: string;
  correct_count: number;
  learned: boolean;
}
const letterProgress = ref<LetterProgress[]>([]);
const learnedThreshold = ref(3);
const totalLetters = 22;  // NGT alphabet size

// The per-letter list is collapsed by default (a tall list of bars is a lot on
// the page); the header shows a one-line summary and expands on click.
const lettersExpanded = ref(false);
const lettersLearnedCount = computed(
  () => letterProgress.value.filter((lp) => lp.learned).length
);

async function fetchLetterProgress(): Promise<void> {
  try {
    const res = await api.get('/stats/letters');
    letterProgress.value = res.data.letters ?? [];
    learnedThreshold.value = res.data.threshold ?? 3;
  } catch {
    // Non-critical: if it fails, the list just doesn't show.
  }
}

const greeting = computed(() => {
  if (isGuest.value) return 'Welcome!';
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning!';
  if (hour < 18) return 'Good afternoon!';
  return 'Good evening!';
});

const ottoMessages = [
  "Let's learn today! 🐙",
  "You're doing great!",
  "Ready to sign?",
  "Practice makes perfect!",
  "I believe in you! 💜",
  "Let's go! 🚀",
  "Sign language is fun!",
];
const surpriseMessages = [
  "Woohoo! Let's go! 🎉",
  "You clicked me! 😄",
  "Eight arms, zero excuses! 💪",
  "Keep signing! 🐙✨",
  "You're my favourite human!",
];
const currentMessage = ref(ottoMessages[0]!);
const isOttoBouncing = ref(false);
let messageTimer: number | null = null;

// Track all loose timers for cleanup
let activeTimeouts: number[] = [];
let activeIntervals: number[] = [];

function trackedTimeout(fn: () => void, delay: number): number {
  const id = window.setTimeout(() => {
    fn();
    activeTimeouts = activeTimeouts.filter(t => t !== id);
  }, delay);
  activeTimeouts.push(id);
  return id;
}

function cycleMessage() {
  const idx = ottoMessages.indexOf(currentMessage.value);
  currentMessage.value = ottoMessages[(idx + 1) % ottoMessages.length]!;
}

function clickOtto() {
  const msg = surpriseMessages[Math.floor(Math.random() * surpriseMessages.length)]!;
  currentMessage.value = msg;
  isOttoBouncing.value = true;
  trackedTimeout(() => { isOttoBouncing.value = false; }, 600);
}

const confettiPieces = ref<{ id: number; left: number; color: string; delay: number; duration: number }[]>([]);

function spawnConfetti(force = false) {
  // Guests never get confetti. Otherwise the streak-based celebration only
  // fires from a 3+ streak, but an explicit force=true (e.g. hitting the daily
  // goal) celebrates regardless of streak.
  if (isGuest.value || (!force && stats.streak < 3)) return;
  const colors = ['#a855f7', '#d8b4fe', '#fbbf24', '#4ade80', '#60a5fa', '#f472b6'];
  confettiPieces.value = Array.from({ length: 30 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    color: colors[Math.floor(Math.random() * colors.length)]!,
    delay: Math.random() * 2,
    duration: 2 + Math.random() * 2,
  }));
  trackedTimeout(() => { confettiPieces.value = []; }, 4000);
}

const displayStreak = ref(0);
// Streak ring fill: empty at 0, full at a 7-day week (175.9 = 2*pi*28, the circle's circumference)
const streakRingOffset = computed(() => 175.9 * (1 - Math.min(displayStreak.value, 7) / 7));
const displayLetters = ref(0);
const displayPoints = ref(0);

function animateCounter(target: typeof displayStreak, end: number, duration: number) {
  const steps = 30;
  const increment = end / steps;
  const interval = duration / steps;
  let current = 0;
  const timer = window.setInterval(() => {
    current += increment;
    if (current >= end) {
      target.value = end;
      clearInterval(timer);
      activeIntervals = activeIntervals.filter(t => t !== timer);
    } else {
      target.value = Math.floor(current);
    }
  }, interval);
  activeIntervals.push(timer);
}

onMounted(async () => {
  // Counters only animate for signed-in users (guests have no saved progress).
  // Fetch the real stats first so the animation counts up to true values.
  if (!isGuest.value) {
    await fetchStats();
    await fetchLetterProgress();
    // Celebrate if today's goal is already complete when the page loads.
    if (goalReached.value) {
      trackedTimeout(() => spawnConfetti(true), 600);
    }
    animateCounter(displayStreak, stats.streak, 800);
    animateCounter(displayLetters, stats.lettersLearned, 1000);
    animateCounter(displayPoints, stats.totalPoints, 1200);
  }
  messageTimer = window.setInterval(cycleMessage, 6000);
  trackedTimeout(spawnConfetti, 800);
});

onUnmounted(() => {
  if (messageTimer) clearInterval(messageTimer);
  activeTimeouts.forEach(id => clearTimeout(id));
  activeTimeouts = [];
  activeIntervals.forEach(id => clearInterval(id));
  activeIntervals = [];
});
</script>

<template>
  <div class="flex-1 relative overflow-y-auto">

    <div class="absolute top-1/2 left-1/4 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-brand-vibrant/[0.12] blur-[120px] pointer-events-none" />
    <div class="absolute top-1/2 right-1/4 translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full bg-brand-vibrant/[0.06] blur-[100px] pointer-events-none" />

    <!-- Confetti -->
    <div class="absolute inset-0 pointer-events-none overflow-hidden z-50" aria-hidden="true">
      <div
        v-for="piece in confettiPieces"
        :key="piece.id"
        class="confetti-piece absolute top-0"
        :style="{
          left: `${piece.left}%`,
          backgroundColor: piece.color,
          animationDelay: `${piece.delay}s`,
          animationDuration: `${piece.duration}s`,
        }"
      />
    </div>

    <!-- Particles -->
    <div class="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
      <div class="particle" style="left: 10%; animation-delay: 0s; animation-duration: 8s;" />
      <div class="particle" style="left: 20%; animation-delay: 1s; animation-duration: 10s;" />
      <div class="particle" style="left: 35%; animation-delay: 2s; animation-duration: 7s;" />
      <div class="particle" style="left: 50%; animation-delay: 0.5s; animation-duration: 9s;" />
      <div class="particle" style="left: 65%; animation-delay: 1.5s; animation-duration: 11s;" />
      <div class="particle" style="left: 80%; animation-delay: 3s; animation-duration: 8s;" />
      <div class="particle" style="left: 90%; animation-delay: 2.5s; animation-duration: 10s;" />
    </div>

    <!-- Scattered background letters -->
    <div class="absolute inset-0 pointer-events-none overflow-hidden select-none" aria-hidden="true">
      <span class="absolute text-[14rem] font-black text-brand-vibrant/[0.07] top-[-3%] left-[-2%] letter-float-1">S</span>
      <span class="absolute text-[11rem] font-black text-brand-vibrant/[0.06] top-[45%] left-[-1%] letter-float-2">G</span>
      <span class="absolute text-[9rem] font-black text-brand-vibrant/[0.07] bottom-[0%] left-[3%] letter-float-3">N</span>
      <span class="absolute text-[7rem] font-black text-brand-vibrant/[0.05] top-[10%] left-[35%] letter-float-4">I</span>
      <span class="absolute text-[8rem] font-black text-brand-vibrant/[0.04] bottom-[15%] right-[3%] letter-float-2">A</span>
      <span class="absolute text-[6rem] font-black text-brand-vibrant/[0.05] top-[5%] right-[2%] letter-float-1">K</span>
      <span class="absolute text-[5rem] font-black text-brand-vibrant/[0.04] top-[60%] right-[15%] letter-float-3">V</span>
      <span class="absolute text-[7rem] font-black text-brand-vibrant/[0.03] bottom-[25%] left-[20%] letter-float-4">W</span>
    </div>

    <!-- Two columns side by side, full width, grows with content -->
    <div class="flex items-start w-full min-h-full">

    <!-- Left side -->
    <div class="w-1/2 flex flex-col items-center justify-center gap-2 p-8 relative z-10 min-h-full">
      <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full bg-brand-vibrant/[0.07] blur-[80px] pointer-events-none" />

      <div class="relative z-10 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl px-5 py-3 speech-bubble">
        <p class="text-white/80 text-sm font-medium">{{ currentMessage }}</p>
      </div>

      <div class="relative z-10 cursor-pointer" @click="clickOtto">
        <div class="absolute bottom-4 left-1/2 -translate-x-1/2 w-64 h-8 bg-brand-vibrant/25 blur-2xl rounded-full" />
        <img
          src="/Otto_1.png"
          alt="Otto the signing octopus"
          class="object-contain drop-shadow-2xl relative z-10"
          :class="isOttoBouncing ? 'otto-bounce' : 'otto-float'"
          style="width: 480px; height: 480px;"
        />
      </div>

      <div class="relative z-10 text-center -mt-8">
        <h1 class="text-6xl font-black tracking-tight gradient-text">SignSee</h1>
        <div class="flex items-center justify-center gap-3 mt-4">
          <div class="h-px w-8 bg-brand-vibrant/40" />
          <p class="text-brand-accent/60 text-sm font-semibold uppercase tracking-[0.2em]">NGT Fingerspelling</p>
          <div class="h-px w-8 bg-brand-vibrant/40" />
        </div>
        <p class="text-brand-accent/30 text-xs mt-2 tracking-widest uppercase">one sign at a time</p>
      </div>

      <!-- Streak pill: signed-in users only -->
      <div v-if="!isGuest" class="relative z-10 flex items-center gap-2 bg-amber-500/10 border border-amber-500/25 rounded-full px-6 py-2.5 streak-glow mt-2">
        <span class="streak-fire">🔥</span>
        <span class="text-amber-400 font-semibold">{{ stats.streak }} day streak</span>
      </div>
    </div>

    <div class="w-px bg-white/5 shrink-0 relative z-10" />

    <!-- Right side -->
    <div class="w-1/2 flex flex-col justify-start gap-6 px-16 py-10 relative z-10 glass-panel min-h-full">

      <div class="flex items-start justify-between">
        <div>
          <h2 class="text-4xl font-black text-white">{{ greeting }}</h2>
          <p class="text-white/40 mt-2">Otto is ready to sign with you today</p>
        </div>
        <!-- Badges: signed-in users only -->
        <div v-if="!isGuest" class="flex flex-col gap-2 items-end shrink-0 ml-4">
          <div v-if="stats.streak >= 3" class="flex items-center gap-2 bg-amber-500/10 border border-amber-500/20 rounded-xl px-3 py-2">
            <span>🏆</span>
            <span class="text-amber-400 text-xs font-bold">On a roll!</span>
          </div>
          <div class="flex items-center gap-2 bg-brand-vibrant/10 border border-brand-vibrant/20 rounded-xl px-3 py-2">
            <span>⚡</span>
            <span class="text-brand-accent text-xs font-bold">Level {{ stats.level }} · {{ stats.levelName }}</span>
          </div>
        </div>
      </div>

      <!-- Guest nudge: shown instead of progress when browsing without an account.
           The whole card is clickable, styled like a softer version of the mode cards. -->
      <button
        v-if="isGuest"
        @click="router.push('/login')"
        class="w-full text-left bg-brand-vibrant/10 border border-brand-vibrant/30 rounded-2xl p-5 flex items-center gap-4 cursor-pointer transition-all duration-200 guest-nudge"
      >
        <div class="w-12 h-12 rounded-xl bg-brand-vibrant/20 border border-brand-vibrant/40 flex items-center justify-center shrink-0" style="font-size: 1.5rem;">
          💾
        </div>
        <div class="flex-1">
          <p class="text-white font-bold">Log in to save your progress</p>
          <p class="text-sm text-white/50 mt-0.5">Track your streak, points, and letters learned.</p>
        </div>
        <div class="w-8 h-8 rounded-full bg-brand-vibrant/20 border border-brand-vibrant/40 flex items-center justify-center shrink-0">
          <i class="pi pi-chevron-right text-brand-accent text-xs" />
        </div>
      </button>

      <!-- Stats: signed-in users only -->
      <div v-if="!isGuest" class="grid grid-cols-3 gap-4">
        <div class="bg-white/5 border border-white/8 rounded-2xl p-5 text-center stat-card">
          <div class="relative w-16 h-16 mx-auto">
            <svg class="w-16 h-16 -rotate-90" viewBox="0 0 64 64">
              <circle cx="32" cy="32" r="28" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="4" />
              <circle cx="32" cy="32" r="28" fill="none" stroke="#fbbf24" stroke-width="4" stroke-linecap="round" stroke-dasharray="175.9" :stroke-dashoffset="streakRingOffset" class="ring-progress" />
            </svg>
            <span class="absolute inset-0 flex items-center justify-center text-3xl font-black text-amber-400">{{ displayStreak }}</span>
          </div>
          <p class="text-xs text-white/30 uppercase tracking-widest h-8 flex items-center justify-center">Streak</p>
        </div>
        <div class="bg-white/5 border border-white/8 rounded-2xl p-5 text-center stat-card">
          <div class="h-16 flex items-center justify-center">
            <p class="font-black text-brand-success leading-none">
              <span class="text-4xl">{{ displayLetters }}</span><span class="text-2xl text-white/40">/{{ totalLetters }}</span>
            </p>
          </div>
          <p class="text-xs text-white/30 uppercase tracking-widest h-8 flex items-center justify-center">Letters learned</p>
        </div>
        <div class="bg-white/5 border border-white/8 rounded-2xl p-5 text-center stat-card">
          <div class="h-16 flex items-center justify-center">
            <p class="text-4xl font-black text-brand-accent">{{ displayPoints }}</p>
          </div>
          <p class="text-xs text-white/30 uppercase tracking-widest h-8 flex items-center justify-center">Points</p>
        </div>
      </div>

      <!-- Daily goal: signed-in users only -->
      <div v-if="!isGuest" class="bg-white/5 border border-white/8 rounded-2xl p-4">
        <div class="flex justify-between text-xs mb-2">
          <span class="text-white/40 uppercase tracking-widest">Daily goal</span>
          <span :class="goalReached ? 'text-brand-goal font-bold' : 'text-brand-accent/60'">
            {{ stats.practicedToday }} / {{ stats.dailyGoal }} letters
          </span>
        </div>
        <div class="h-3 bg-white/10 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all duration-700"
            :class="goalReached ? 'bg-brand-goal' : 'bg-brand-vibrant'"
            :style="{ width: `${dailyGoalPercent}%` }"
          />
        </div>
        <p v-if="goalReached" class="text-[10px] text-brand-goal mt-2 font-semibold">
          Daily goal complete — nice work! 🎉
        </p>
        <p v-else class="text-[10px] text-white/20 mt-2">
          Practice {{ stats.dailyGoal }} letters today to keep your streak!
        </p>
      </div>

      <!-- Per-letter progress: signed-in users who've practiced at least one letter.
           Shows learned letters and progress toward learning the rest, which makes
           the "3x correct to learn a letter" rule self-evident. -->
      <div v-if="!isGuest && letterProgress.length" class="bg-white/5 border border-white/8 rounded-2xl p-4">
        <button
          @click="lettersExpanded = !lettersExpanded"
          class="w-full flex items-center justify-between cursor-pointer group"
        >
          <span class="text-xs text-white/30 uppercase tracking-widest group-hover:text-white/50 transition-colors">
            Your letters
          </span>
          <span class="flex items-center gap-2">
            <span class="text-xs text-brand-accent/60">{{ lettersLearnedCount }} of {{ totalLetters }} learned</span>
            <i
              class="pi text-white/40 text-xs transition-transform"
              :class="lettersExpanded ? 'pi-chevron-up' : 'pi-chevron-down'"
            />
          </span>
        </button>
        <div v-if="lettersExpanded" class="flex flex-col gap-2.5 mt-3">
          <div
            v-for="lp in letterProgress"
            :key="lp.letter"
            class="flex items-center gap-3"
          >
            <span
              class="w-5 text-center font-black text-sm"
              :class="lp.learned ? 'text-brand-learned' : 'text-white/70'"
            >{{ lp.letter }}</span>
            <div class="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                class="h-full rounded-full transition-all duration-500"
                :class="lp.learned ? 'bg-brand-learned' : 'bg-brand-vibrant'"
                :style="{ width: `${Math.min((lp.correct_count / learnedThreshold) * 100, 100)}%` }"
              />
            </div>
            <span
              class="text-[11px] min-w-[58px] text-right"
              :class="lp.learned ? 'text-brand-learned font-semibold' : 'text-white/40'"
            >{{ lp.learned ? '✓ Learned' : `${lp.correct_count} / ${learnedThreshold}` }}</span>
          </div>
        </div>
      </div>

      <!-- Navigation cards -->
      <div class="flex flex-col gap-4">
        <p class="text-xs text-white/30 uppercase tracking-widest">Choose your mode</p>

        <button
          @click="router.push('/learn')"
          class="w-full rounded-2xl p-6 text-left
                 card-hover transition-all duration-200 cursor-pointer flex items-center gap-5 learn-card-glow"
        >
          <div class="w-16 h-16 rounded-2xl bg-blue-500/25 border border-blue-500/40 flex items-center justify-center shrink-0" style="font-size: 2rem;">
            📚
          </div>
          <div class="flex-1">
            <p class="text-xl font-black text-white">Learn</p>
            <p class="text-sm text-white/40 mt-0.5">Practice letters with live camera feedback</p>
          </div>
          <div class="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500/30 flex items-center justify-center shrink-0">
            <i class="pi pi-chevron-right text-blue-400 text-xs" />
          </div>
        </button>

        <button
          @click="router.push('/play')"
          class="w-full rounded-2xl p-6 text-left
                 card-hover-purple transition-all duration-200 cursor-pointer flex items-center gap-5 play-card-glow"
        >
          <div class="w-16 h-16 rounded-2xl bg-brand-vibrant/30 border border-brand-vibrant/50 flex items-center justify-center shrink-0" style="font-size: 2rem;">
            🎮
          </div>
          <div class="flex-1">
            <p class="text-xl font-black text-white">Play</p>
            <p class="text-sm text-white/40 mt-0.5">Test your skills with game challenges</p>
          </div>
          <div class="w-8 h-8 rounded-full bg-brand-vibrant/20 border border-brand-vibrant/40 flex items-center justify-center shrink-0">
            <i class="pi pi-chevron-right text-brand-accent text-xs" />
          </div>
        </button>

        <button
          @click="router.push('/collect')"
          class="w-full rounded-2xl p-6 text-left
                 transition-all duration-200 cursor-pointer flex items-center gap-5 collect-card-glow"
        >
          <div class="w-16 h-16 rounded-2xl bg-emerald-500/25 border border-emerald-500/40 flex items-center justify-center shrink-0" style="font-size: 2rem;">
            📸
          </div>
          <div class="flex-1">
            <p class="text-xl font-black text-white">Collect</p>
            <p class="text-sm text-white/40 mt-0.5">Contribute signs to help SignSee learn</p>
          </div>
          <div class="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center shrink-0">
            <i class="pi pi-chevron-right text-emerald-400 text-xs" />
          </div>
        </button>
      </div>

      <!-- Last played: signed-in users only -->
      <div v-if="!isGuest" class="flex items-center justify-between bg-white/3 border border-white/5 rounded-xl px-5 py-4">
        <div class="flex items-center gap-3">
          <span class="text-lg">🎯</span>
          <div>
            <p class="text-xs text-white/25 uppercase tracking-widest">Last played</p>
            <p class="text-sm text-white/60 font-medium mt-0.5">{{ stats.lastPlayed ?? 'Nothing yet' }}</p>
          </div>
        </div>
        <button
          @click="router.push('/play/random-letters')"
          class="text-sm font-semibold text-brand-vibrant hover:text-brand-accent transition-colors cursor-pointer bg-brand-vibrant/10 border border-brand-vibrant/20 rounded-xl px-4 py-2"
        >
          Play again →
        </button>
      </div>

    </div>
    </div>
  </div>
</template>

<style scoped>
.gradient-text {
  background: linear-gradient(135deg, #ffffff 0%, #d8b4fe 50%, #a855f7 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  padding-bottom: 4px;
}

.glass-panel {
  background: rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(20px);
  border-left: 1px solid rgba(168, 85, 247, 0.15);
  box-shadow: inset 1px 0 0 rgba(255, 255, 255, 0.05);
}

.guest-nudge:hover {
  background: rgba(168, 85, 247, 0.18);
  border-color: rgba(168, 85, 247, 0.6);
  box-shadow: 0 0 24px rgba(168, 85, 247, 0.2);
  transform: translateY(-2px);
}

.confetti-piece {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  animation: confettiFall linear forwards;
}
@keyframes confettiFall {
  0% { top: -10px; opacity: 1; transform: rotate(0deg) translateX(0); }
  100% { top: 100vh; opacity: 0; transform: rotate(720deg) translateX(50px); }
}

.particle {
  position: absolute;
  bottom: -10px;
  width: 4px;
  height: 4px;
  background: rgba(168, 85, 247, 0.4);
  border-radius: 50%;
  animation: particleRise linear infinite;
}
@keyframes particleRise {
  0% { bottom: -10px; opacity: 0; transform: translateX(0) scale(1); }
  10% { opacity: 1; }
  90% { opacity: 0.5; }
  100% { bottom: 100%; opacity: 0; transform: translateX(30px) scale(0.3); }
}

.otto-float {
  animation: ottoFloat 3s ease-in-out infinite;
}
@keyframes ottoFloat {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-20px); }
}

.otto-bounce {
  animation: ottoBounce 0.6s ease-out;
}
@keyframes ottoBounce {
  0% { transform: scale(1) translateY(0); }
  20% { transform: scale(1.1) translateY(-15px); }
  40% { transform: scale(0.95) translateY(5px); }
  60% { transform: scale(1.05) translateY(-8px); }
  80% { transform: scale(0.98) translateY(2px); }
  100% { transform: scale(1) translateY(0); }
}

.speech-bubble {
  position: relative;
  animation: bubblePop 0.3s ease-out;
}
.speech-bubble::after {
  content: '';
  position: absolute;
  bottom: -10px;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: rgba(255,255,255,0.2);
}
@keyframes bubblePop {
  from { transform: scale(0.8); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}

.streak-fire {
  animation: fireWiggle 0.8s ease-in-out infinite alternate;
  display: inline-block;
}
@keyframes fireWiggle {
  from { transform: rotate(-10deg) scale(1); }
  to { transform: rotate(10deg) scale(1.2); }
}

.streak-glow {
  position: relative;
  animation: streakPulse 2s ease-in-out infinite;
}
@keyframes streakPulse {
  0%, 100% { box-shadow: 0 0 8px rgba(251, 191, 36, 0.2); }
  50% { box-shadow: 0 0 20px rgba(251, 191, 36, 0.5); }
}
.streak-glow::before,
.streak-glow::after {
  content: '✦';
  position: absolute;
  color: rgba(251, 191, 36, 0.6);
  font-size: 12px;
  animation: sparkle 2s ease-in-out infinite;
}
.streak-glow::before { top: -8px; left: 10px; animation-delay: 0s; }
.streak-glow::after { top: -6px; right: 10px; animation-delay: 1s; }
@keyframes sparkle {
  0%, 100% { opacity: 0; transform: scale(0.5) rotate(0deg); }
  50% { opacity: 1; transform: scale(1.2) rotate(180deg); }
}

.ring-progress {
  transition: stroke-dashoffset 1s ease-out;
}

.stat-card {
  transition: all 0.2s ease;
}
.stat-card:hover {
  background: rgba(255,255,255,0.08);
  border-color: rgba(168,85,247,0.3);
  box-shadow: 0 0 20px rgba(168,85,247,0.1);
  transform: translateY(-2px);
}

.letter-float-1 { animation: letterDrift1 8s ease-in-out infinite; }
.letter-float-2 { animation: letterDrift2 10s ease-in-out infinite; }
.letter-float-3 { animation: letterDrift3 12s ease-in-out infinite; }
.letter-float-4 { animation: letterDrift4 9s ease-in-out infinite; }
@keyframes letterDrift1 {
  0%, 100% { transform: rotate(-15deg) translateY(0px); }
  50% { transform: rotate(-12deg) translateY(-15px); }
}
@keyframes letterDrift2 {
  0%, 100% { transform: rotate(8deg) translateY(0px); }
  50% { transform: rotate(11deg) translateY(-20px); }
}
@keyframes letterDrift3 {
  0%, 100% { transform: rotate(12deg) translateY(0px); }
  50% { transform: rotate(8deg) translateY(-10px); }
}
@keyframes letterDrift4 {
  0%, 100% { transform: rotate(15deg) translateY(0px); }
  50% { transform: rotate(18deg) translateY(-18px); }
}

.xp-fill {
  width: 40%;
  animation: xpFill 1.5s ease-out forwards;
}
@keyframes xpFill {
  from { width: 0%; }
  to { width: 40%; }
}

.learn-card-glow {
  border: 1px solid rgba(59, 130, 246, 0.5);
  background: rgba(59, 130, 246, 0.1);
  animation: learnCardPulse 2.5s ease-in-out infinite;
}
@keyframes learnCardPulse {
  0%, 100% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.2), inset 0 0 15px rgba(59, 130, 246, 0.05); }
  50% { box-shadow: 0 0 30px rgba(59, 130, 246, 0.4), inset 0 0 20px rgba(59, 130, 246, 0.1); }
}

.play-card-glow {
  border: 1px solid rgba(168, 85, 247, 0.5);
  background: rgba(168, 85, 247, 0.12);
  animation: playCardPulse 2.5s ease-in-out infinite;
}
@keyframes playCardPulse {
  0%, 100% { box-shadow: 0 0 15px rgba(168, 85, 247, 0.2), inset 0 0 15px rgba(168, 85, 247, 0.05); }
  50% { box-shadow: 0 0 30px rgba(168, 85, 247, 0.4), inset 0 0 20px rgba(168, 85, 247, 0.1); }
}

.collect-card-glow {
  border: 1px solid rgba(16, 185, 129, 0.5);
  background: rgba(16, 185, 129, 0.1);
  animation: playCardPulse 2.5s ease-in-out infinite;
}
.collect-card-glow:hover {
  background: rgba(16, 185, 129, 0.22) !important;
  border-color: rgba(16, 185, 129, 0.6) !important;
  box-shadow: 0 0 28px rgba(16, 185, 129, 0.25) !important;
  transform: translateY(-2px) scale(1.01);
}

.card-hover:hover {
  background: rgba(59, 130, 246, 0.18) !important;
  border-color: rgba(59, 130, 246, 0.6) !important;
  box-shadow: 0 0 28px rgba(59, 130, 246, 0.25) !important;
  transform: translateY(-2px) scale(1.01);
}
.card-hover-purple:hover {
  background: rgba(168,85,247,0.22) !important;
  border-color: rgba(168,85,247,0.6) !important;
  box-shadow: 0 0 28px rgba(168,85,247,0.25) !important;
  transform: translateY(-2px) scale(1.01);
}
</style>
