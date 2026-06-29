<script setup lang="ts">
import { ref, computed, watch, onUnmounted, nextTick } from 'vue';
import { useRouter } from 'vue-router';
import { useCamera } from '@/composables/useCamera';
import { VALID_LETTERS, LETTER_HINTS } from '@/constants/letters';
import { api } from '@/composables/useApi';

// ── Router ─────────────────────────────────────────────────────────────
const router = useRouter();

// ── Template refs ──────────────────────────────────────────────────────
const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);

// ── Camera composable ──────────────────────────────────────────────────
const {
  cameraError,
  connectionStatus,
  activeHands,
  enableCamera,
  disableCamera,
  handColor,
  handOverlayPosition,
} = useCamera(videoRef, canvasRef);

// ── Game state ─────────────────────────────────────────────────────────
const gameStarted = ref(false);
const letters = ref<string[]>([]);
const currentIndex = ref(0);
const results = ref<Map<string, 'completed' | 'completed_hint' | 'completed_image' | 'skipped'>>(new Map());
const confirmationProgress = ref(0);
const confirmationTimerSeconds = ref(2);

// The letter the model currently recognizes (prefers the stabilized reading),
// and whether it matches the letter the player has been asked to sign. Mirrors
// the Learn page: we watch this boolean so the hold timer reliably starts/stops
// on the match transition, rather than watching the raw hands array (which may
// update by mutation and not trigger the watcher).
const predictedLetter = computed(() => {
  const first = activeHands.value[0];
  return (first?.stable_letter ?? first?.predicted_letter)?.toUpperCase() ?? null;
});
const isMatch = computed(() => predictedLetter.value === currentTarget.value);

// ── Scoring state ──────────────────────────────────────────────────────
const score = ref(0);
const currentStreak = ref(0);
const bestStreak = ref(0);

// ── Hint state ─────────────────────────────────────────────────────────
const letterElapsed = ref(0);
const hintButtonVisible = ref(false);
const textHintVisible = ref(false);
const stillStuckVisible = ref(false);
const imageHintVisible = ref(false);
let letterTimer: number | null = null;

const HINT_BUTTON_DELAY = 15;
const STILL_STUCK_DELAY = 10;

let textHintShownAt = 0;

function startLetterTimer() {
  letterElapsed.value = 0;
  hintButtonVisible.value = false;
  textHintVisible.value = false;
  stillStuckVisible.value = false;
  imageHintVisible.value = false;
  textHintShownAt = 0;

  if (letterTimer) clearInterval(letterTimer);
  letterTimer = window.setInterval(() => {
    letterElapsed.value += 1;

    if (!hintButtonVisible.value && letterElapsed.value >= HINT_BUTTON_DELAY) {
      hintButtonVisible.value = true;
    }

    if (textHintVisible.value && !stillStuckVisible.value) {
      const timeSinceTextHint = letterElapsed.value - textHintShownAt;
      if (timeSinceTextHint >= STILL_STUCK_DELAY) {
        stillStuckVisible.value = true;
      }
    }
  }, 1000);
}

function clearLetterTimer() {
  if (letterTimer) {
    clearInterval(letterTimer);
    letterTimer = null;
  }
}

function showTextHint() {
  textHintVisible.value = true;
  textHintShownAt = letterElapsed.value;
}

function showImageHint() {
  imageHintVisible.value = true;
  clearConfirmationTimer();
  // Mark letter as revealed — no points, must skip to continue
  if (currentTarget.value) {
    results.value.set(currentTarget.value, 'completed_image');
    currentStreak.value = 0;
  }
}

// ── Pop-up message state ───────────────────────────────────────────────
const popupMessage = ref('');
const popupVisible = ref(false);
const popupType = ref<'success' | 'streak' | 'skip' | 'hint'>('success');
let popupTimeout: number | null = null;

const SUCCESS_MESSAGES = ['Nice!', 'Nailed it!', 'Great sign!', 'Keep going!', 'Well done!', 'Perfect!'];
const STREAK_MESSAGES = ['3 in a row!', 'On fire!', 'Unstoppable!', 'Incredible streak!'];
const SKIP_MESSAGES = ['No worries', "You'll get it next time", 'Moving on!'];
const HINT_MESSAGES = ['Good learning!', 'Now you know!', 'Getting there!'];

function randomFrom(arr: string[]): string {
  return arr[Math.floor(Math.random() * arr.length)]!;
}

function showPopup(message: string, type: 'success' | 'streak' | 'skip' | 'hint') {
  if (popupTimeout) clearTimeout(popupTimeout);
  popupMessage.value = message;
  popupType.value = type;
  popupVisible.value = true;
  popupTimeout = window.setTimeout(() => {
    popupVisible.value = false;
  }, 1800);
}

// ── Card flash state ───────────────────────────────────────────────────
const flashingLetter = ref<string | null>(null);
let flashTimeout: number | null = null;

function flashCard(letter: string) {
  flashingLetter.value = letter;
  if (flashTimeout) clearTimeout(flashTimeout);
  flashTimeout = window.setTimeout(() => {
    flashingLetter.value = null;
  }, 800);
}

// ── Derived state ──────────────────────────────────────────────────────
const currentTarget = computed(() => letters.value[currentIndex.value] ?? null);
const isRoundOver = computed(() => letters.value.length > 0 && currentIndex.value >= letters.value.length);

const skippedCount = computed(() =>
  [...results.value.values()].filter(v => v === 'skipped').length
);
const noHelpCount = computed(() =>
  [...results.value.values()].filter(v => v === 'completed').length
);
const hintAssistedCount = computed(() =>
  [...results.value.values()].filter(v => v === 'completed_hint').length
);
const imageRevealedCount = computed(() =>
  [...results.value.values()].filter(v => v === 'completed_image').length
);

// ── Confirmation timer internals ───────────────────────────────────────
let confirmationInterval: number | null = null;
let confirmationStartTime: number | null = null;

function startConfirmationTimer() {
  if (confirmationInterval) return;
  confirmationStartTime = performance.now();
  confirmationProgress.value = 0;

  confirmationInterval = window.setInterval(() => {
    if (!confirmationStartTime) return;
    const elapsed = (performance.now() - confirmationStartTime) / 1000;
    confirmationProgress.value = Math.min(elapsed / confirmationTimerSeconds.value, 1);

    if (confirmationProgress.value >= 1) {
      clearConfirmationTimer();
      markCompleted();
    }
  }, 50);
}

function clearConfirmationTimer() {
  if (confirmationInterval) {
    clearInterval(confirmationInterval);
    confirmationInterval = null;
  }
  confirmationStartTime = null;
  confirmationProgress.value = 0;
}

// ── Game actions ───────────────────────────────────────────────────────
function markCompleted() {
  if (!currentTarget.value) return;

  let resultType: 'completed' | 'completed_hint';
  let points: number;

  if (textHintVisible.value) {
    resultType = 'completed_hint';
    points = 5;
    showPopup(randomFrom(HINT_MESSAGES), 'hint');
    currentStreak.value = 0;
  } else {
    resultType = 'completed';
    points = 10;
    currentStreak.value += 1;

    if (currentStreak.value >= 3 && currentStreak.value % 3 === 0) {
      points += 3;
      showPopup(randomFrom(STREAK_MESSAGES), 'streak');
    } else {
      showPopup(randomFrom(SUCCESS_MESSAGES), 'success');
    }
  }

  if (currentStreak.value > bestStreak.value) {
    bestStreak.value = currentStreak.value;
  }

  score.value += points;

  // Report this correct sign to the backend so the user's stats (points,
  // streak, letters-learned) update. Fire-and-forget — a failed report must
  // never interrupt gameplay.
  api.post('/stats/progress', {
    letter: currentTarget.value,
    correct: true,
    points,
    activity: 'Random Letters',
  }).catch(() => { /* offline or guest token expired — ignore */ });

  results.value.set(currentTarget.value, resultType);
  flashCard(currentTarget.value);
  advanceToNext();
}

function skipLetter() {
  if (!currentTarget.value) return;
  // Don't overwrite if already marked as completed_image
  if (!results.value.has(currentTarget.value)) {
    results.value.set(currentTarget.value, 'skipped');
    showPopup(randomFrom(SKIP_MESSAGES), 'skip');
  }
  currentStreak.value = 0;
  advanceToNext();
}

function advanceToNext() {
  clearConfirmationTimer();
  clearLetterTimer();
  currentIndex.value += 1;

  if (isRoundOver.value) {
    disableCamera();
  } else {
    nextTick(() => {
      startLetterTimer();
    });
  }
}

// ── Fisher-Yates shuffle ───────────────────────────────────────────────
function shuffleArray(arr: string[]): string[] {
  const shuffled = [...arr];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
  }
  return shuffled;
}

// ── Start / restart game ───────────────────────────────────────────────
async function startGame() {
  letters.value = shuffleArray([...VALID_LETTERS]);
  currentIndex.value = 0;
  results.value = new Map();
  score.value = 0;
  currentStreak.value = 0;
  bestStreak.value = 0;
  clearConfirmationTimer();
  clearLetterTimer();
  popupVisible.value = false;
  flashingLetter.value = null;
  gameStarted.value = true;
  await enableCamera();
  startLetterTimer();
}

// ── Card state helper ──────────────────────────────────────────────────
function cardState(letter: string): 'active' | 'completed' | 'skipped' | 'failed' | 'pending' {
  if (letter === currentTarget.value && !isRoundOver.value && !imageHintVisible.value) return 'active';
  const result = results.value.get(letter);
  if (result === 'skipped') return 'skipped';
  if (result === 'completed_image') return 'failed';
  if (result) return 'completed';
  return 'pending';
}

// ── Prediction matching (watch activeHands) ────────────────────────────
watch(isMatch, (matched) => {
  // Don't run the hold timer once the round is over, there's no target, or
  // the image hint was revealed (the player must skip in that case).
  if (isRoundOver.value || !currentTarget.value || imageHintVisible.value) {
    clearConfirmationTimer();
    return;
  }

  if (matched) {
    if (!confirmationInterval) {
      startConfirmationTimer();
    }
  } else {
    clearConfirmationTimer();
  }
});

// If the hand already matches when the target changes (the watcher above won't
// fire because isMatch didn't transition), start the timer immediately —
// mirrors the same guard on the Learn page.
watch(currentTarget, () => {
  clearConfirmationTimer();
  if (isMatch.value && !isRoundOver.value && !imageHintVisible.value) {
    startConfirmationTimer();
  }
});

// ── Cleanup ────────────────────────────────────────────────────────────
onUnmounted(() => {
  clearConfirmationTimer();
  clearLetterTimer();
  if (popupTimeout) clearTimeout(popupTimeout);
  if (flashTimeout) clearTimeout(flashTimeout);
  disableCamera();
});
</script>

<template>
  <!-- Ready screen (before game starts) -->
  <div v-if="!gameStarted" class="flex-1 flex items-center justify-center p-6 relative overflow-hidden">
    <!-- Background glow -->
    <div class="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-brand-vibrant/[0.07] blur-[120px] pointer-events-none" />

    <!-- Scattered background letters -->
    <div class="absolute inset-0 pointer-events-none overflow-hidden select-none" aria-hidden="true">
      <span class="absolute text-6xl font-black text-brand-vibrant/[0.04] top-[8%] left-[5%] rotate-[-15deg]">A</span>
      <span class="absolute text-8xl font-black text-brand-vibrant/[0.05] top-[4%] right-[12%] rotate-[20deg]">K</span>
      <span class="absolute text-5xl font-black text-brand-vibrant/[0.03] top-[18%] left-[18%] rotate-[8deg]">S</span>
      <span class="absolute text-7xl font-black text-brand-vibrant/[0.06] top-[12%] right-[3%] rotate-[-25deg]">F</span>
      <span class="absolute text-9xl font-black text-brand-vibrant/[0.04] top-[35%] left-[2%] rotate-[30deg]">W</span>
      <span class="absolute text-5xl font-black text-brand-vibrant/[0.05] top-[30%] right-[5%] rotate-[-12deg]">B</span>
      <span class="absolute text-6xl font-black text-brand-vibrant/[0.03] top-[50%] left-[8%] rotate-[-22deg]">M</span>
      <span class="absolute text-8xl font-black text-brand-vibrant/[0.05] top-[45%] right-[8%] rotate-[18deg]">D</span>
      <span class="absolute text-7xl font-black text-brand-vibrant/[0.04] top-[65%] left-[4%] rotate-[12deg]">P</span>
      <span class="absolute text-5xl font-black text-brand-vibrant/[0.06] top-[60%] right-[2%] rotate-[-30deg]">G</span>
      <span class="absolute text-6xl font-black text-brand-vibrant/[0.03] top-[75%] left-[15%] rotate-[25deg]">T</span>
      <span class="absolute text-9xl font-black text-brand-vibrant/[0.04] top-[72%] right-[10%] rotate-[-8deg]">R</span>
      <span class="absolute text-5xl font-black text-brand-vibrant/[0.05] top-[85%] left-[6%] rotate-[-18deg]">N</span>
      <span class="absolute text-7xl font-black text-brand-vibrant/[0.03] top-[88%] right-[6%] rotate-[15deg]">E</span>
      <span class="absolute text-6xl font-black text-brand-vibrant/[0.04] top-[22%] left-[85%] rotate-[35deg]">I</span>
      <span class="absolute text-8xl font-black text-brand-vibrant/[0.03] top-[55%] left-[88%] rotate-[-20deg]">V</span>
    </div>
    <div class="max-w-md w-full text-center relative z-10">

      <!-- Welcome to -->
      <p class="text-4xl text-white mb-4" style="font-family: 'Playfair Display', serif; font-weight: 700; font-style: italic;">It's time for . . .</p>

      <!-- Icon -->
      <div class="w-40 h-40 mx-auto mb-4">
        <svg width="100%" viewBox="0 0 160 160" role="img">
          <circle cx="80" cy="80" r="80" fill="#a855f7" fill-opacity="0.1" stroke="#a855f7" stroke-width="0.5"/>
          <text x="50" y="60" font-family="sans-serif" font-size="18" font-weight="700" fill="#d8b4fe" fill-opacity="0.25" transform="rotate(-20,50,60)">F</text>
          <text x="105" y="48" font-family="sans-serif" font-size="16" font-weight="700" fill="#a855f7" fill-opacity="0.3" transform="rotate(15,105,48)">M</text>
          <text x="38" y="92" font-family="sans-serif" font-size="14" font-weight="700" fill="#d8b4fe" fill-opacity="0.2" transform="rotate(-10,38,92)">R</text>
          <text x="118" y="88" font-family="sans-serif" font-size="15" font-weight="700" fill="#a855f7" fill-opacity="0.25" transform="rotate(10,118,88)">G</text>
          <text x="62" y="118" font-family="sans-serif" font-size="14" font-weight="700" fill="#d8b4fe" fill-opacity="0.2" transform="rotate(6,62,118)">T</text>
          <text x="102" y="120" font-family="sans-serif" font-size="13" font-weight="700" fill="#a855f7" fill-opacity="0.2" transform="rotate(-8,102,120)">P</text>
          <text x="80" y="92" font-family="sans-serif" font-size="48" font-weight="700" fill="#d8b4fe" fill-opacity="0.9" text-anchor="middle">?</text>
        </svg>
      </div>

      <!-- Game mode title -->
      <h2 class="text-3xl font-black text-white mb-2">Random Letters</h2>
      <p class="text-white/40 mb-8">Sign all 22 letters in random order</p>

      <!-- Rules -->
      <div class="text-left bg-white/5 border border-white/10 rounded-2xl p-5 mb-8 space-y-3">
        <div class="flex items-start gap-3">
          <i class="pi pi-camera text-brand-accent mt-0.5" />
          <p class="text-sm text-white/60">Your camera will activate to detect hand signs</p>
        </div>
        <div class="flex items-start gap-3">
          <i class="pi pi-clock text-brand-accent mt-0.5" />
          <p class="text-sm text-white/60">Hold each sign steady for 2 seconds to confirm</p>
        </div>
        <div class="flex items-start gap-3">
          <i class="pi pi-star text-brand-accent mt-0.5" />
          <p class="text-sm text-white/60">Earn 10 pts per sign — hints reduce points, streaks give bonuses</p>
        </div>
        <div class="flex items-start gap-3">
          <i class="pi pi-question-circle text-brand-accent mt-0.5" />
          <p class="text-sm text-white/60">Stuck? A hint will appear after 15 seconds</p>
        </div>
      </div>

      <!-- Start button -->
      <button
        @click="startGame"
        class="w-full py-4 rounded-xl font-bold text-lg
               bg-brand-vibrant text-white
               hover:bg-brand-vibrant/80 transition-colors cursor-pointer"
      >
        Start game
      </button>

      <!-- Back link -->
      <button
        @click="router.push('/play')"
        class="mt-4 text-sm text-white/30 hover:text-white/50 transition-colors cursor-pointer"
      >
        <i class="pi pi-arrow-left mr-1 text-xs" />
        Back to game modes
      </button>
    </div>
  </div>

  <!-- Game UI -->
  <div v-else-if="!isRoundOver" class="flex-1 flex flex-col h-full overflow-hidden relative">

    <!-- Pop-up message toast -->
    <Transition name="popup">
      <div
        v-if="popupVisible"
        class="absolute top-6 left-1/2 -translate-x-1/2 z-50 px-6 py-3 rounded-2xl text-sm font-bold backdrop-blur-md border shadow-lg"
        :class="{
          'bg-brand-success/20 border-brand-success/40 text-brand-success': popupType === 'success',
          'bg-amber-500/20 border-amber-500/40 text-amber-400': popupType === 'streak',
          'bg-white/10 border-white/20 text-white/60': popupType === 'skip',
          'bg-blue-500/20 border-blue-500/40 text-blue-400': popupType === 'hint',
        }"
      >
        {{ popupMessage }}
      </div>
    </Transition>

    <!-- Top section: Camera + Target -->
    <div class="flex-1 flex flex-col lg:flex-row gap-4 p-4 min-h-0">

      <!-- Camera feed -->
      <div class="relative flex-1 rounded-2xl overflow-hidden bg-black/40 border border-white/10 min-h-[280px]">
        <video
          ref="videoRef"
          autoplay
          playsinline
          muted
          class="w-full h-full object-cover scale-x-[-1]"
        />
        <canvas
          ref="canvasRef"
          class="absolute inset-0 w-full h-full scale-x-[-1] pointer-events-none"
        />

        <!-- Per-hand prediction overlays -->
        <div
          v-for="hand in activeHands"
          :key="hand.hand_id"
          class="absolute flex flex-col gap-2 pointer-events-none"
          :style="handOverlayPosition(hand)"
        >
          <div class="bg-black/60 backdrop-blur-md border border-white/10 p-3 rounded-2xl flex flex-col items-center min-w-[80px]">
            <span class="text-[10px] font-bold uppercase opacity-50 tracking-tighter mb-1">{{ hand.label }}</span>
            <span class="text-4xl font-black" :style="{ color: handColor(hand.hand_id) }">
              {{ hand.predicted_letter?.toUpperCase() }}
            </span>
            <span class="text-xs font-bold mt-1" :class="hand.confidence > 0.9 ? 'text-green-400' : 'text-red-400'">
              {{ (hand.confidence * 100).toFixed(0) }}%
            </span>
          </div>
          <div class="flex gap-1">
            <div
              v-for="alt in hand.alternatives"
              :key="alt.letter"
              class="bg-black/40 backdrop-blur-md border border-white/5 px-2 py-1 rounded-xl flex items-baseline gap-1"
            >
              <span class="text-sm font-bold opacity-80">{{ alt.letter }}</span>
              <span class="text-[10px] font-medium opacity-40">{{ (alt.confidence * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>

        <!-- Score badge -->
        <div class="absolute top-3 right-3 px-3 py-1.5 rounded-xl bg-black/60 backdrop-blur-md border border-white/10 text-sm font-bold text-brand-accent">
          {{ score }} pts
          <span v-if="currentStreak >= 3" class="ml-1 text-amber-400">{{ currentStreak }}x</span>
        </div>

        <!-- Connection status badge -->
        <div
          v-if="connectionStatus !== 'connected' && connectionStatus !== 'idle'"
          class="absolute top-3 left-3 px-3 py-1 rounded-full text-xs font-medium bg-black/60 backdrop-blur-sm"
          :class="{
            'text-yellow-400': connectionStatus === 'connecting' || connectionStatus === 'disconnected',
            'text-red-400': connectionStatus === 'error',
          }"
        >
          {{ connectionStatus === 'connecting' ? 'Connecting...' : connectionStatus === 'error' ? 'Connection error' : 'Reconnecting...' }}
        </div>

        <!-- Camera error -->
        <div
          v-if="cameraError"
          class="absolute inset-0 flex items-center justify-center bg-black/80 p-6"
        >
          <p class="text-red-400 text-sm text-center max-w-xs">{{ cameraError }}</p>
        </div>
      </div>

      <!-- Target + controls panel -->
      <div class="lg:w-72 flex flex-col gap-4 shrink-0 overflow-y-auto">

        <!-- Current target letter -->
        <div class="rounded-2xl bg-white/5 border border-white/10 p-6 text-center">
          <p class="text-xs uppercase tracking-widest text-brand-accent/60 mb-2">Sign this letter</p>
          <p class="text-7xl font-black text-brand-accent leading-none">
            {{ currentTarget }}
          </p>
          <p class="text-xs text-white/30 mt-3">
            {{ currentIndex + 1 }} / {{ letters.length }}
          </p>
        </div>

        <!-- Confirmation progress bar -->
        <div class="rounded-2xl bg-white/5 border border-white/10 p-4">
          <div class="flex justify-between text-xs text-white/40 mb-2">
            <span>Hold steady</span>
            <span>{{ (confirmationProgress * confirmationTimerSeconds).toFixed(1) }}s / {{ confirmationTimerSeconds }}s</span>
          </div>
          <div class="h-3 rounded-full bg-white/10 overflow-hidden">
            <div
              class="h-full rounded-full transition-all duration-100 ease-linear"
              :class="confirmationProgress >= 1 ? 'bg-brand-success' : 'bg-brand-vibrant'"
              :style="{ width: `${confirmationProgress * 100}%` }"
            />
          </div>
        </div>

        <!-- Hint button (appears after 15s) -->
        <div v-if="hintButtonVisible && !textHintVisible" class="animate-fade-in">
          <button
            @click="showTextHint"
            class="w-full py-3 rounded-xl text-sm font-semibold
                   bg-blue-500/10 border border-blue-500/30 text-blue-400
                   hover:bg-blue-500/20 transition-colors cursor-pointer"
          >
            <i class="pi pi-question-circle mr-2" />
            Need a hint? (5 pts)
          </button>
        </div>

        <!-- Text hint display -->
        <div v-if="textHintVisible" class="rounded-2xl bg-blue-500/10 border border-blue-500/30 p-4 animate-fade-in">
          <p class="text-xs uppercase tracking-widest text-blue-400/60 mb-2">Hint</p>
          <p class="text-sm text-blue-200/80 leading-relaxed">
            {{ LETTER_HINTS[currentTarget ?? ''] }}
          </p>
        </div>

        <!-- Still stuck button (appears 10s after text hint) -->
        <div v-if="stillStuckVisible && !imageHintVisible" class="animate-fade-in">
          <button
            @click="showImageHint"
            class="w-full py-3 rounded-xl text-sm font-semibold
                   bg-orange-500/10 border border-orange-500/30 text-orange-400
                   hover:bg-orange-500/20 transition-colors cursor-pointer"
          >
            <i class="pi pi-image mr-2" />
            Still stuck? Show me (0 pts)
          </button>
        </div>

        <!-- Image hint display -->
        <div v-if="imageHintVisible" class="rounded-2xl bg-orange-500/10 border border-orange-500/30 p-4 animate-fade-in">
          <p class="text-xs uppercase tracking-widest text-orange-400/60 mb-2">Reference</p>
          <img
            :src="`/signs/${currentTarget?.toLowerCase()}_illustration.png`"
            :alt="`Sign for ${currentTarget}`"
            class="w-full aspect-square rounded-xl object-contain bg-white/5 border border-white/10"
          />
          <p class="text-[10px] text-orange-400/40 mt-2 text-center">No points for this letter</p>
        </div>

        <!-- Skip / Next button -->
        <button
          @click="skipLetter"
          class="w-full py-3 rounded-xl text-sm font-semibold transition-colors cursor-pointer"
          :class="imageHintVisible
            ? 'bg-orange-500/10 border border-orange-500/30 text-orange-400 hover:bg-orange-500/20'
            : 'bg-white/5 border border-white/10 text-white/50 hover:bg-white/10 hover:text-white/70'"
        >
          {{ imageHintVisible ? 'Next letter' : 'Skip letter' }}
          <i class="pi pi-forward ml-1 text-xs" />
        </button>
      </div>
    </div>

    <!-- Bottom section: Letter card grid -->
    <div class="px-4 pb-4">
      <div class="flex flex-wrap justify-center gap-2">
        <div
          v-for="letter in VALID_LETTERS"
          :key="letter"
          class="w-11 h-11 rounded-lg flex items-center justify-center text-sm font-bold transition-all duration-200 select-none"
          :class="{
            'bg-brand-vibrant/20 text-brand-accent border-2 border-brand-vibrant shadow-[0_0_12px_rgba(168,85,247,0.4)] scale-110':
              cardState(letter) === 'active',
            'bg-brand-success/40 text-brand-success border-2 border-brand-success shadow-[0_0_16px_rgba(74,222,128,0.5)] scale-115 animate-card-flash':
              cardState(letter) === 'completed' && flashingLetter === letter,
            'bg-brand-success/20 text-brand-success border border-brand-success/40':
              cardState(letter) === 'completed' && flashingLetter !== letter,
            'bg-red-500/15 text-red-400/60 border border-red-500/30':
              cardState(letter) === 'failed',
            'bg-orange-500/10 text-orange-400/50 border border-orange-500/20 line-through':
              cardState(letter) === 'skipped',
            'bg-white/5 text-white/25 border border-white/5':
              cardState(letter) === 'pending',
          }"
        >
          {{ letter }}
        </div>
      </div>
    </div>
  </div>

  <!-- Summary screen -->
  <div v-else class="flex-1 flex items-center justify-center p-6">
    <div class="max-w-md w-full text-center">

      <!-- Trophy / done icon -->
      <div class="w-20 h-20 mx-auto mb-6 rounded-full bg-brand-vibrant/10 border border-brand-vibrant/30 flex items-center justify-center">
        <i class="pi pi-trophy text-4xl text-brand-accent" />
      </div>

      <h2 class="text-3xl font-black text-white mb-1">Round complete</h2>
      <p class="text-5xl font-black text-brand-accent mb-2">{{ score }} pts</p>
      <p class="text-white/40 mb-8">Here's how you did</p>

      <!-- Stats -->
      <div class="flex justify-center gap-4 mb-8">
        <div class="text-center">
          <p class="text-3xl font-black text-brand-success">{{ noHelpCount }}</p>
          <p class="text-[10px] text-white/40 mt-1">No help</p>
        </div>
        <div class="w-px bg-white/10" />
        <div class="text-center">
          <p class="text-3xl font-black text-blue-400">{{ hintAssistedCount }}</p>
          <p class="text-[10px] text-white/40 mt-1">With hint</p>
        </div>
        <div class="w-px bg-white/10" />
        <div class="text-center">
          <p class="text-3xl font-black text-red-400">{{ imageRevealedCount }}</p>
          <p class="text-[10px] text-white/40 mt-1">Failed</p>
        </div>
        <div class="w-px bg-white/10" />
        <div class="text-center">
          <p class="text-3xl font-black text-white/30">{{ skippedCount }}</p>
          <p class="text-[10px] text-white/40 mt-1">Skipped</p>
        </div>
      </div>

      <!-- Streak badge -->
      <div v-if="bestStreak >= 3" class="mb-8">
        <span class="px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 font-bold text-sm inline-block">
          Best streak: {{ bestStreak }} in a row
        </span>
      </div>

      <!-- Letter results grid -->
      <div class="flex flex-wrap justify-center gap-2 mb-10">
        <div
          v-for="letter in VALID_LETTERS"
          :key="letter"
          class="w-10 h-10 rounded-lg flex items-center justify-center text-xs font-bold"
          :class="{
            'bg-brand-success/20 text-brand-success border border-brand-success/40':
              results.get(letter) === 'completed',
            'bg-blue-500/15 text-blue-400 border border-blue-500/30':
              results.get(letter) === 'completed_hint',
            'bg-red-500/15 text-red-400 border border-red-500/30':
              results.get(letter) === 'completed_image',
            'bg-white/5 text-white/20 border border-white/5 line-through':
              results.get(letter) === 'skipped',
            'bg-white/5 text-white/10 border border-white/5':
              !results.get(letter),
          }"
        >
          {{ letter }}
        </div>
      </div>

      <!-- Action buttons -->
      <div class="flex gap-3 justify-center">
        <button
          @click="startGame"
          class="px-6 py-3 rounded-xl font-semibold text-sm
                 bg-brand-vibrant text-white
                 hover:bg-brand-vibrant/80 transition-colors cursor-pointer"
        >
          <i class="pi pi-refresh mr-2" />
          Play again
        </button>
        <button
          @click="router.push('/play')"
          class="px-6 py-3 rounded-xl font-semibold text-sm
                 bg-white/5 border border-white/10 text-white/60
                 hover:bg-white/10 hover:text-white/80 transition-colors cursor-pointer"
        >
          <i class="pi pi-arrow-left mr-2" />
          Game modes
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.popup-enter-active {
  transition: all 0.3s ease-out;
}
.popup-leave-active {
  transition: all 0.4s ease-in;
}
.popup-enter-from {
  opacity: 0;
  transform: translateX(-50%) translateY(-20px);
}
.popup-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(-10px);
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}

.animate-card-flash {
  animation: cardFlash 0.8s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes cardFlash {
  0% { transform: scale(1.3); box-shadow: 0 0 24px rgba(74, 222, 128, 0.7); }
  50% { transform: scale(1.15); box-shadow: 0 0 16px rgba(74, 222, 128, 0.4); }
  100% { transform: scale(1); box-shadow: none; }
}
</style>
