<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue';
import { useRouter } from 'vue-router';
import { useCamera } from '@/composables/useCamera';
import { VALID_LETTERS, LETTER_HINTS } from '@/constants/letters';
import { api } from '@/composables/useApi';

const router = useRouter();

const HOLD_DURATION = 2000; // ms

const selectedLetter = ref<string>('A');
const showPhoto = ref(false);
const isCorrect = ref(false);
const practicedLetters = ref<string[]>([]);
const confettiPieces = ref<{ id: number; left: number; color: string; delay: number; duration: number }[]>([]);

// Hold timer
const holdProgress = ref(0); // 0-100
let holdInterval: number | null = null;
let holdStart: number | null = null;
let confettiTimeout: number | null = null;
let celebrationTimeouts: number[] = [];

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);

const {
  isCameraActive,
  cameraError,
  connectionStatus,
  activeHands,
  enableCamera,
  disableCamera,
  handColor,
  handOverlayPosition,
} = useCamera(videoRef, canvasRef);

const illustrationSrc = computed(() =>
  `/signs/${selectedLetter.value.toLowerCase()}_illustration.png`
);
const photoSrc = computed(() =>
  `/signs/${selectedLetter.value.toLowerCase()}_photo.jpg`
);

const currentHint = computed(() => LETTER_HINTS[selectedLetter.value] ?? '');

const progressPercent = computed(() =>
  Math.round((practicedLetters.value.length / VALID_LETTERS.length) * 100)
);

const isCompleted = computed(() =>
  practicedLetters.value.length === VALID_LETTERS.length
);

const predictedLetter = computed(() => {
  const first = activeHands.value[0];
  return (first?.stable_letter ?? first?.predicted_letter)?.toUpperCase() ?? null;
});

const isMatch = computed(() =>
  predictedLetter.value === selectedLetter.value
);

const currentLetterIndex = computed(() =>
  VALID_LETTERS.indexOf(selectedLetter.value as typeof VALID_LETTERS[number])
);

const nextLetter = computed(() => {
  const next = currentLetterIndex.value + 1;
  return next < VALID_LETTERS.length ? VALID_LETTERS[next] : null;
});

function spawnConfetti() {
  const colors = ['#a855f7', '#d8b4fe', '#fbbf24', '#4ade80', '#60a5fa', '#f472b6'];
  confettiPieces.value = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    left: 30 + Math.random() * 40,
    color: colors[Math.floor(Math.random() * colors.length)]!,
    delay: Math.random() * 0.5,
    duration: 1 + Math.random() * 1,
  }));
  if (confettiTimeout) clearTimeout(confettiTimeout);
  confettiTimeout = window.setTimeout(() => {
    confettiPieces.value = [];
  }, 2000);
}

function startHoldTimer() {
  if (holdInterval) return;
  holdStart = Date.now();
  holdInterval = window.setInterval(() => {
    const elapsed = Date.now() - (holdStart ?? Date.now());
    holdProgress.value = Math.min((elapsed / HOLD_DURATION) * 100, 100);
    if (holdProgress.value >= 100) {
      stopHoldTimer();
      if (!isCorrect.value) {
        isCorrect.value = true;
        if (!practicedLetters.value.includes(selectedLetter.value)) {
          practicedLetters.value.push(selectedLetter.value);
        }

        // Report the correct sign to the backend. Learn is practice, not a
        // scored mode, so no points are awarded (points: 0) — but the sign
        // still counts toward letters-learned and keeps the daily streak.
        // Fire-and-forget so a failed report never disrupts practice.
        api.post('/stats/progress', {
          letter: selectedLetter.value,
          correct: true,
          points: 0,
          activity: 'Learn',
        }).catch(() => { /* offline or guest — ignore */ });
        spawnConfetti();
        if (practicedLetters.value.length === VALID_LETTERS.length) {
          celebrationTimeouts.push(
            window.setTimeout(() => spawnConfetti(), 500),
            window.setTimeout(() => spawnConfetti(), 1000),
            window.setTimeout(() => spawnConfetti(), 1500),
          );
        }
      }
    }
  }, 30);
}

function stopHoldTimer() {
  if (holdInterval) {
    clearInterval(holdInterval);
    holdInterval = null;
  }
  holdStart = null;
  if (!isCorrect.value) {
    holdProgress.value = 0;
  }
}

watch(isMatch, (val) => {
  if (val && !isCorrect.value) {
    startHoldTimer();
  } else {
    stopHoldTimer();
  }
});

function selectLetter(letter: string) {
  selectedLetter.value = letter;
  isCorrect.value = false;
  showPhoto.value = false;
  holdProgress.value = 0;
  stopHoldTimer();

  // If the hand already matches the new target, start immediately
  // (the watcher won't fire because isMatch never transitioned)
  if (isMatch.value) {
    startHoldTimer();
  }
}

function goToNext() {
  if (nextLetter.value) {
    selectLetter(nextLetter.value);
  }
}

function resetPractice() {
  practicedLetters.value = [];
  selectedLetter.value = 'A';
  isCorrect.value = false;
  holdProgress.value = 0;
  stopHoldTimer();
  celebrationTimeouts.forEach(id => clearTimeout(id));
  celebrationTimeouts = [];
}

onUnmounted(() => {
  disableCamera();
  stopHoldTimer();
  if (confettiTimeout) clearTimeout(confettiTimeout);
  celebrationTimeouts.forEach(id => clearTimeout(id));
  celebrationTimeouts = [];
});
</script>

<template>
  <div class="flex-1 flex overflow-hidden relative">

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

    <!-- Completion overlay -->
    <div
      v-if="isCompleted"
      class="absolute inset-0 z-40 flex items-center justify-center backdrop-blur-sm bg-black/60"
    >
      <div class="bg-black/80 border border-brand-vibrant/30 rounded-3xl p-10 text-center max-w-md mx-4 relative overflow-hidden">
        <div class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-brand-vibrant/50 to-transparent" />
        <div class="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-brand-vibrant/30 to-transparent" />
        <div class="text-6xl mb-4">🎉</div>
        <h2 class="text-3xl font-black text-white mb-2">Amazing work!</h2>
        <p class="text-brand-accent/60 mb-1">You signed all 22 letters!</p>
        <p class="text-white/30 text-sm mb-8">You've completed the full NGT fingerspelling alphabet.</p>
        <div class="grid grid-cols-3 gap-3 mb-8">
          <div class="bg-white/5 border border-white/8 rounded-2xl p-3 text-center">
            <p class="text-2xl font-black text-amber-400">22</p>
            <p class="text-[10px] text-white/30 mt-1 uppercase tracking-widest">Letters</p>
          </div>
          <div class="bg-white/5 border border-white/8 rounded-2xl p-3 text-center">
            <p class="text-2xl font-black text-brand-success">100%</p>
            <p class="text-[10px] text-white/30 mt-1 uppercase tracking-widest">Complete</p>
          </div>
          <div class="bg-white/5 border border-white/8 rounded-2xl p-3 text-center">
            <p class="text-2xl font-black">🏆</p>
            <p class="text-[10px] text-white/30 mt-1 uppercase tracking-widest">Trophy</p>
          </div>
        </div>
        <div class="flex flex-col gap-3">
          <button
            @click="router.push('/play')"
            class="w-full py-3 rounded-xl bg-brand-vibrant text-white font-bold hover:bg-brand-vibrant/80 transition-all hover:shadow-[0_0_20px_rgba(168,85,247,0.4)] cursor-pointer"
          >
            🎮 Try the game now!
          </button>
          <button
            @click="resetPractice"
            class="w-full py-3 rounded-xl bg-white/5 border border-white/10 text-white/50 font-semibold hover:bg-white/10 hover:text-white/70 transition-colors cursor-pointer"
          >
            Practice again
          </button>
          <button
            @click="router.push('/')"
            class="text-xs text-white/25 hover:text-white/40 transition-colors cursor-pointer"
          >
            Back to home
          </button>
        </div>
      </div>
    </div>

    <!-- Left side: letter grid -->
    <div class="w-64 shrink-0 flex flex-col overflow-hidden" style="background: rgba(0,0,0,0.3); border-right: 1px solid rgba(168,85,247,0.12);">

      <!-- Back + title -->
      <div class="p-5 border-b border-white/5">
        <button
          @click="router.push('/')"
          class="flex items-center gap-2 text-white/30 hover:text-brand-accent transition-colors text-xs mb-4 cursor-pointer group"
        >
          <div class="w-5 h-5 rounded-full bg-white/5 border border-white/10 flex items-center justify-center group-hover:border-brand-vibrant/40 transition-colors">
            <i class="pi pi-arrow-left text-[8px]" />
          </div>
          Back to home
        </button>
        <div class="flex items-center gap-2 mb-1">
          <div class="w-6 h-6 rounded-lg bg-brand-vibrant/20 border border-brand-vibrant/30 flex items-center justify-center text-sm">📚</div>
          <h2 class="text-lg font-black text-white">Learn</h2>
        </div>
        <p class="text-[11px] text-white/30 uppercase tracking-widest">Select a letter</p>
      </div>

      <!-- Selected letter box -->
      <div
        class="mx-4 mt-4 rounded-2xl border p-4 text-center transition-all duration-300 relative overflow-hidden shrink-0"
        :class="isCorrect
          ? 'bg-brand-success/15 border-brand-success/40 shadow-[0_0_20px_rgba(74,222,128,0.2)]'
          : isMatch && !isCorrect
            ? 'bg-brand-vibrant/10 border-brand-vibrant/30'
            : 'bg-white/5 border-white/8'"
      >
        <div class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-brand-vibrant/30 to-transparent" />
        <p class="text-[10px] uppercase tracking-[0.2em] text-white/25 mb-1">Signing</p>
        <p
          class="text-6xl font-black leading-none transition-all duration-300"
          :class="isCorrect ? 'text-brand-success' : 'gradient-letter'"
        >
          {{ selectedLetter }}
        </p>

        <!-- Hold timer bar -->
        <div v-if="isCameraActive && !isCorrect" class="mt-3">
          <div class="flex justify-between text-[10px] text-white/25 mb-1">
            <span>Hold steady</span>
            <span>{{ ((holdProgress / 100) * 2).toFixed(1) }}s / 2s</span>
          </div>
          <div class="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-none"
              :class="holdProgress > 0 ? 'bg-brand-vibrant' : 'bg-white/10'"
              :style="{ width: `${holdProgress}%` }"
            />
          </div>
        </div>

        <!-- Correct state -->
        <div v-if="isCorrect" class="mt-3 flex flex-col gap-2">
          <div class="flex items-center justify-center gap-1 bg-brand-success/10 rounded-xl py-1.5">
            <i class="pi pi-check-circle text-brand-success text-xs" />
            <span class="text-brand-success text-xs font-bold">Great sign!</span>
          </div>
          <button
            v-if="nextLetter"
            @click="goToNext"
            class="w-full py-1.5 rounded-xl bg-brand-vibrant/20 border border-brand-vibrant/30 text-brand-accent text-xs font-semibold hover:bg-brand-vibrant/30 transition-colors cursor-pointer flex items-center justify-center gap-1"
          >
            Next: {{ nextLetter }} <i class="pi pi-arrow-right text-xs" />
          </button>
        </div>

        <p v-else-if="!isCameraActive" class="text-[10px] text-white/20 mt-2">Start camera to practice</p>
        <p v-else-if="holdProgress === 0" class="text-[10px] text-white/20 mt-2">Sign this letter</p>
      </div>

      <!-- Letter grid -->
      <div class="flex-1 overflow-y-auto p-4">
        <div class="grid grid-cols-4 gap-2">
          <button
            v-for="letter in VALID_LETTERS"
            :key="letter"
            @click="selectLetter(letter)"
            class="aspect-square rounded-xl flex items-center justify-center text-base font-black transition-all duration-200 cursor-pointer relative"
            :class="selectedLetter === letter
              ? 'bg-brand-vibrant text-white shadow-[0_0_16px_rgba(168,85,247,0.5)] scale-110 border border-brand-vibrant'
              : practicedLetters.includes(letter)
                ? 'bg-brand-success/20 border border-brand-success/40 text-brand-success hover:scale-105'
                : 'bg-white/5 border border-white/8 text-white/40 hover:bg-white/10 hover:text-white hover:border-brand-vibrant/40 hover:scale-105'"
          >
            {{ letter }}
            <span v-if="practicedLetters.includes(letter) && selectedLetter !== letter" class="absolute top-0.5 right-0.5 w-2 h-2 rounded-full bg-brand-success" />
          </button>
        </div>
      </div>

      <!-- Progress -->
      <div class="p-4 border-t border-white/5">
        <div class="flex justify-between items-center mb-2">
          <p class="text-[10px] text-white/20 uppercase tracking-widest">Progress</p>
          <p class="text-[10px] text-brand-accent/50">{{ practicedLetters.length }} / 22</p>
        </div>
        <div class="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div
            class="h-full bg-brand-vibrant rounded-full transition-all duration-500"
            :style="{ width: `${progressPercent}%` }"
          />
        </div>
      </div>
    </div>

    <!-- Center: camera feed -->
    <div class="flex-1 relative p-4 min-w-0">
      <div class="w-full h-full relative rounded-2xl overflow-hidden bg-black/40 border border-white/8">

        <video
          ref="videoRef"
          autoplay
          playsinline
          muted
          class="w-full h-full object-cover scale-x-[-1]"
          :class="{ 'hidden': !isCameraActive }"
        />
        <canvas
          ref="canvasRef"
          class="absolute inset-0 w-full h-full scale-x-[-1] pointer-events-none"
          :class="{ 'hidden': !isCameraActive }"
        />

        <!-- Per-hand overlays -->
        <div
          v-for="hand in activeHands"
          :key="hand.hand_id"
          class="absolute flex flex-col gap-2 pointer-events-none"
          :style="handOverlayPosition(hand)"
        >
          <div
            class="backdrop-blur-md border p-3 rounded-2xl flex flex-col items-center min-w-[80px] transition-all duration-300"
            :class="hand.predicted_letter?.toUpperCase() === selectedLetter
              ? 'bg-brand-success/20 border-brand-success/50'
              : 'bg-black/60 border-white/10'"
          >
            <span class="text-[10px] font-bold uppercase opacity-50 tracking-tighter mb-1">{{ hand.label }}</span>
            <span
              class="text-4xl font-black"
              :style="{ color: hand.predicted_letter?.toUpperCase() === selectedLetter ? '#4ade80' : handColor(hand.hand_id) }"
            >
              {{ hand.predicted_letter?.toUpperCase() }}
            </span>
            <span class="text-xs font-bold mt-1" :class="hand.confidence > 0.9 ? 'text-green-400' : 'text-red-400'">
              {{ (hand.confidence * 100).toFixed(0) }}%
            </span>
          </div>
        </div>

        <!-- Match flash overlay -->
        <div
          v-if="isCorrect"
          class="absolute inset-0 border-4 border-brand-success rounded-2xl pointer-events-none correct-flash"
        />

        <!-- Camera offline -->
        <div v-if="!isCameraActive" class="absolute inset-0 flex flex-col items-center justify-center gap-4">
          <div class="w-20 h-20 rounded-full bg-white/5 border border-white/10 flex items-center justify-center mb-2">
            <i class="pi pi-camera text-3xl text-white/20" />
          </div>
          <p class="text-white/50 font-semibold text-lg">Camera offline</p>
          <p class="text-white/25 text-sm">Start the camera to practice signing</p>
          <p v-if="cameraError" class="text-red-400 text-sm text-center max-w-xs px-4">{{ cameraError }}</p>
          <button
            @click="enableCamera"
            class="mt-2 px-8 py-3 rounded-xl bg-brand-vibrant text-white font-bold hover:bg-brand-vibrant/80 transition-all hover:shadow-[0_0_20px_rgba(168,85,247,0.4)] cursor-pointer"
          >
            <i class="pi pi-camera mr-2" />Start Camera
          </button>
        </div>

        <!-- Connection status -->
        <div
          v-if="isCameraActive && connectionStatus !== 'connected' && connectionStatus !== 'idle'"
          class="absolute top-3 left-3 px-3 py-1 rounded-full text-xs font-medium bg-black/60 backdrop-blur-sm border border-white/10"
          :class="{
            'text-yellow-400': connectionStatus === 'connecting' || connectionStatus === 'disconnected',
            'text-red-400': connectionStatus === 'error',
          }"
        >
          {{ connectionStatus === 'connecting' ? 'Connecting...' : connectionStatus === 'error' ? 'Connection error' : 'Reconnecting...' }}
        </div>

        <!-- Stop camera -->
        <button
          v-if="isCameraActive"
          @click="disableCamera"
          class="absolute top-3 right-3 px-3 py-1.5 rounded-xl bg-black/60 backdrop-blur-md border border-white/10 text-xs text-white/50 hover:text-white/80 transition-colors cursor-pointer flex items-center gap-1"
        >
          <i class="pi pi-video-slash text-xs" /> Stop
        </button>

        <!-- Current letter badge -->
        <div
          v-if="isCameraActive"
          class="absolute bottom-4 left-1/2 -translate-x-1/2 px-5 py-2 rounded-full backdrop-blur-md border text-sm font-bold transition-all duration-300"
          :class="isCorrect
            ? 'bg-brand-success/20 border-brand-success/50 text-brand-success'
            : 'bg-black/60 border-white/10 text-white/60'"
        >
          {{ isCorrect ? '✓ Great sign!' : `Sign: ${selectedLetter}` }}
        </div>

      </div>
    </div>

    <!-- Right side: reference panel -->
    <div class="w-72 shrink-0 flex flex-col gap-4 p-4 overflow-y-auto" style="background: rgba(0,0,0,0.3); border-left: 1px solid rgba(168,85,247,0.12);">

      <!-- Hint -->
      <div class="rounded-2xl border border-white/8 bg-white/5 p-4">
        <div class="flex items-center gap-2 mb-2">
          <i class="pi pi-lightbulb text-brand-accent/60 text-xs" />
          <p class="text-[10px] uppercase tracking-widest text-white/25">How to sign {{ selectedLetter }}</p>
        </div>
        <p class="text-xs text-white/50 leading-relaxed">{{ currentHint }}</p>
      </div>

      <!-- Image toggle -->
      <div class="flex gap-2 p-1 bg-white/5 border border-white/8 rounded-xl">
        <button
          @click="showPhoto = false"
          class="flex-1 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer"
          :class="!showPhoto
            ? 'bg-brand-vibrant text-white shadow-[0_0_12px_rgba(168,85,247,0.3)]'
            : 'text-white/40 hover:text-white/60'"
        >
          <i class="pi pi-image mr-1" /> Illustration
        </button>
        <button
          @click="showPhoto = true"
          class="flex-1 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer"
          :class="showPhoto
            ? 'bg-brand-vibrant text-white shadow-[0_0_12px_rgba(168,85,247,0.3)]'
            : 'text-white/40 hover:text-white/60'"
        >
          <i class="pi pi-camera mr-1" /> Photo
        </button>
      </div>

      <!-- Main reference image -->
      <div class="rounded-2xl border border-white/8 bg-white/5 overflow-hidden flex items-center justify-center" style="min-height: 200px;">
        <img
          :src="showPhoto ? photoSrc : illustrationSrc"
          :alt="`Sign for letter ${selectedLetter}`"
          class="w-full object-contain p-3 transition-opacity duration-200"
        />
      </div>

      <!-- Both references -->
      <div>
        <p class="text-[10px] uppercase tracking-widest text-white/20 mb-2">Both references</p>
        <div class="flex gap-2">
          <div
            class="flex-1 rounded-xl border overflow-hidden flex items-center justify-center p-2 cursor-pointer transition-all"
            :class="!showPhoto ? 'border-brand-vibrant/40 bg-brand-vibrant/5' : 'border-white/8 bg-white/5 hover:border-white/15'"
            @click="showPhoto = false"
          >
            <img :src="illustrationSrc" :alt="`Illustration ${selectedLetter}`" class="w-full object-contain" />
          </div>
          <div
            class="flex-1 rounded-xl border overflow-hidden flex items-center justify-center p-2 cursor-pointer transition-all"
            :class="showPhoto ? 'border-brand-vibrant/40 bg-brand-vibrant/5' : 'border-white/8 bg-white/5 hover:border-white/15'"
            @click="showPhoto = true"
          >
            <img :src="photoSrc" :alt="`Photo ${selectedLetter}`" class="w-full object-contain" />
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<style scoped>
.gradient-letter {
  background: linear-gradient(135deg, #ffffff 0%, #d8b4fe 60%, #a855f7 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
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

.correct-flash {
  animation: correctFlash 1.5s ease-out;
}
@keyframes correctFlash {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 0; }
}
</style>
