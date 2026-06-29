<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useCamera } from '../composables/useCamera';
import { api } from '../composables/useApi';
import { LETTER_HINTS } from '@/constants/letters';

const router = useRouter();

// ── Capture modes ──────────────────────────────────────────────────────
// camera  : webcam frame + the user picks the letter themselves
// upload  : an uploaded image file + the user picks the letter
// auto    : webcam frame + the label comes from the live model prediction
type Mode = 'camera' | 'upload' | 'auto';
const mode = ref<Mode>('camera');

const modes: { id: Mode; label: string; icon: string; hint: string }[] = [
  { id: 'camera', label: 'Camera', icon: '📸', hint: 'Sign live, you pick the letter' },
  { id: 'upload', label: 'Upload', icon: '🖼️', hint: 'Upload a photo, you pick the letter' },
  { id: 'auto', label: 'Auto-label', icon: '✨', hint: 'The model picks the letter' },
];

// ── The alphabet (NGT fingerspelling labels) ───────────────────────────
const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const selectedLetter = ref<string | null>(null);

// Reference panel (mirrors the Learn page) for the currently selected letter.
// Collapsed by default so it doesn't dominate the rail; expand to see the full
// illustration/photo + hint while collecting.
const showPhoto = ref(false);
const referenceExpanded = ref(false);
// In camera/upload the user picks the letter; in auto mode the model's live
// prediction is the subject, so the reference follows whichever applies.
const referenceLetter = computed(() =>
  mode.value === 'auto' ? liveLetter.value : selectedLetter.value
);
const illustrationSrc = computed(() =>
  referenceLetter.value ? `/signs/${referenceLetter.value.toLowerCase()}_illustration.png` : ''
);
const photoSrc = computed(() =>
  referenceLetter.value ? `/signs/${referenceLetter.value.toLowerCase()}_photo.jpg` : ''
);
const currentHint = computed(() =>
  referenceLetter.value ? (LETTER_HINTS[referenceLetter.value] ?? '') : ''
);

// ── Camera (reuses the shared composable) ───────────────────────────────
const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const {
  isCameraActive,
  cameraError,
  activeHands,
  enableCamera,
  disableCamera,
} = useCamera(videoRef, canvasRef);

// The live model prediction for auto-label mode = first detected hand.
const livePrediction = computed(() => activeHands.value[0] ?? null);
const liveLetter = computed(() => livePrediction.value?.predicted_letter ?? null);
const liveConfidence = computed(() => livePrediction.value?.confidence ?? 0);

// Whether a hand is currently detected (drives the smart-crop indicator).
const handDetected = computed(() => !!livePrediction.value?.landmarks?.length);

// ── Uploaded image ──────────────────────────────────────────────────────
const uploadedImage = ref<string | null>(null);
const fileInputRef = ref<HTMLInputElement | null>(null);

function onFilePicked(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    uploadedImage.value = typeof reader.result === 'string' ? reader.result : null;
  };
  reader.readAsDataURL(file);
}

// ── Collected samples ───────────────────────────────────────────────────
// A sample = a small thumbnail (base64) + its label + how it was captured.
interface Sample {
  id: number;
  thumbnail: string;
  letter: string;
  source: Mode;
  at: string;
  stored: boolean;
}

const samples = ref<Sample[]>([]);
const STORAGE_KEY = 'collect_samples';

// Persist across reloads (best-effort). We store reasonably-sized cropped
// images, but stay mindful of the browser storage size limit. If storage is
// unavailable the samples simply live in memory for the session.
function loadSamples() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) samples.value = JSON.parse(saved);
  } catch {
    /* storage unavailable — fall back to in-memory only */
  }
}

function persistSamples() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples.value));
  } catch {
    /* over quota or unavailable — keep going with in-memory state */
  }
}

// ── The single storage seam ─────────────────────────────────────────────
// Each captured sample is (1) added to the local gallery so the user sees it
// immediately, and (2) sent to the backend collect endpoint to be stored for
// training. The backend call is best-effort: if it fails, the local gallery
// still works and clearly marks the sample as local-only. Auth is optional
// server-side — the token (if any) is attached
// automatically, so both signed-in users and guests can contribute.
async function saveSample(image: string, letter: string, source: Mode): Promise<boolean> {
  const sample: Sample = {
    id: Date.now(),
    thumbnail: image,
    letter,
    source,
    at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    stored: false,
  };
  samples.value.unshift(sample);
  if (samples.value.length > 40) samples.value.pop();
  persistSamples();

  try {
    await api.post('/collect', { image, letter, source });
    sample.stored = true;
    persistSamples();
    return true;
  } catch {
    return false;
  }
}

// ── The hand guide box ───────────────────────────────────────────────────
// A centered square region: the capture always crops to this box, and the
// overlay shows the user where to place their hand. Expressed as
// fractions of the frame so it maps cleanly onto any resolution.
const GUIDE = { cx: 0.5, cy: 0.5, size: 0.7 }; // centred square, 70% of the shorter side

// ── Capturing + smart crop ───────────────────────────────────────────────
// Preview is mirrored (selfie view); we draw the frame mirrored too, so the
// saved image matches what the user saw. Then we crop:
//   • around the detected hand landmarks, with generous padding, when a hand
//     is found — gives a tight, hand-dominant sample. The padding is large so
//     the hand's outer silhouette (fingertips, edges) isn't clipped.
//   • to the fixed guide box otherwise (no hand detected).
// Output is a square image at good quality for the dataset.
const OUTPUT_SIZE = 640;   // saved image is OUTPUT_SIZE × OUTPUT_SIZE
const HAND_PADDING = 0.65; // margin around the hand bbox; generous to avoid clipping fingers

function captureCropped(): string | null {
  const video = videoRef.value;
  if (!video || video.videoWidth === 0) return null;
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  // 1) Draw the full frame, mirrored, onto a working canvas.
  const full = document.createElement('canvas');
  full.width = vw;
  full.height = vh;
  const fctx = full.getContext('2d');
  if (!fctx) return null;
  fctx.translate(vw, 0);
  fctx.scale(-1, 1);
  fctx.drawImage(video, 0, 0);

  // 2) Work out the crop square (in pixels) on the mirrored frame.
  // Prefer a tight crop around the detected hand (with generous padding so the
  // full hand is kept); fall back to the guide box when no hand is detected.
  let sx: number, sy: number, side: number;
  const hand = activeHands.value[0];

  if (hand?.landmarks?.length) {
    // Landmark x/y are normalised 0..1 in the mirrored preview space, which
    // matches our mirrored canvas — so we can use them directly.
    const xs = hand.landmarks.map((p) => p.x);
    const ys = hand.landmarks.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const boxW = (maxX - minX) * vw;
    const boxH = (maxY - minY) * vh;
    const cx = ((minX + maxX) / 2) * vw;
    const cy = ((minY + maxY) / 2) * vh;
    // Square side = larger hand dimension + generous padding on both sides,
    // so fingertips and the hand's outer edge aren't clipped.
    side = Math.max(boxW, boxH) * (1 + HAND_PADDING * 2);
    sx = cx - side / 2;
    sy = cy - side / 2;
  } else {
    // Fallback: the fixed guide box.
    side = Math.min(vw, vh) * GUIDE.size;
    sx = GUIDE.cx * vw - side / 2;
    sy = GUIDE.cy * vh - side / 2;
  }

  // Keep the crop square fully inside the frame.
  side = Math.min(side, vw, vh);
  sx = Math.max(0, Math.min(sx, vw - side));
  sy = Math.max(0, Math.min(sy, vh - side));

  // 3) Render the crop into a square output canvas at good quality.
  const out = document.createElement('canvas');
  out.width = OUTPUT_SIZE;
  out.height = OUTPUT_SIZE;
  const octx = out.getContext('2d');
  if (!octx) return null;
  octx.imageSmoothingQuality = 'high';
  octx.drawImage(full, sx, sy, side, side, 0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  return out.toDataURL('image/jpeg', 0.95);
}

// For uploads we don't have landmarks, so just resize to a square-ish, good
// quality image (object-fit contain onto a square canvas).
function prepareUpload(dataUrl: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = OUTPUT_SIZE;
      canvas.height = OUTPUT_SIZE;
      const ctx = canvas.getContext('2d');
      if (!ctx) return resolve(dataUrl);
      // letterbox onto a dark square
      ctx.fillStyle = '#0b0014';
      ctx.fillRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
      const scale = Math.min(OUTPUT_SIZE / img.width, OUTPUT_SIZE / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, (OUTPUT_SIZE - w) / 2, (OUTPUT_SIZE - h) / 2, w, h);
      resolve(canvas.toDataURL('image/jpeg', 0.95));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}

// ── Submit (per mode) ───────────────────────────────────────────────────
const flash = ref('');
let flashTimer: number | null = null;

function showFlash(msg: string) {
  flash.value = msg;
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = window.setTimeout(() => { flash.value = ''; }, 2200);
}

const canSubmit = computed(() => {
  if (mode.value === 'camera') return isCameraActive.value && !!selectedLetter.value;
  if (mode.value === 'upload') return !!uploadedImage.value && !!selectedLetter.value;
  if (mode.value === 'auto') return isCameraActive.value && !!liveLetter.value && liveConfidence.value > 0;
  return false;
});

async function submit() {
  if (!canSubmit.value) return;

  if (mode.value === 'camera') {
    const img = captureCropped();
    if (!img || !selectedLetter.value) return;
    const stored = await saveSample(img, selectedLetter.value, 'camera');
    showFlash(stored
      ? `Saved "${selectedLetter.value}" to Azure`
      : `Saved "${selectedLetter.value}" locally; Azure upload failed`);
  } else if (mode.value === 'upload') {
    if (!uploadedImage.value || !selectedLetter.value) return;
    const img = await prepareUpload(uploadedImage.value);
    const stored = await saveSample(img, selectedLetter.value, 'upload');
    showFlash(stored
      ? `Saved "${selectedLetter.value}" to Azure`
      : `Saved "${selectedLetter.value}" locally; Azure upload failed`);
    uploadedImage.value = null;
    if (fileInputRef.value) fileInputRef.value.value = '';
  } else if (mode.value === 'auto') {
    const img = captureCropped();
    if (!img || !liveLetter.value) return;
    const stored = await saveSample(img, liveLetter.value, 'auto');
    showFlash(stored
      ? `Saved model guess "${liveLetter.value}" to Azure`
      : `Saved model guess "${liveLetter.value}" locally; Azure upload failed`);
  }
}

// ── Per-letter tally ─────────────────────────────────────────────────────
const tally = computed(() => {
  const counts: Record<string, number> = {};
  for (const s of samples.value) counts[s.letter] = (counts[s.letter] ?? 0) + 1;
  return counts;
});

function clearAll() {
  samples.value = [];
  persistSamples();
}

function removeSample(id: number) {
  samples.value = samples.value.filter((s) => s.id !== id);
  persistSamples();
}

// ── Enlarge a saved sample ───────────────────────────────────────────────
const preview = ref<Sample | null>(null);

const sourceLabel = (s: Mode) =>
  s === 'auto' ? 'Auto-labelled' : s === 'upload' ? 'Uploaded' : 'Camera';
const sourceIcon = (s: Mode) =>
  s === 'auto' ? '✨' : s === 'upload' ? '🖼️' : '📸';

// ── Switching modes: turn the camera on/off as needed ───────────────────
watch(mode, (m) => {
  selectedLetter.value = null;
  if (m === 'upload' && isCameraActive.value) disableCamera();
});

onMounted(loadSamples);

onUnmounted(() => {
  if (flashTimer) clearTimeout(flashTimer);
  // useCamera already stops the stream on unmount.
});
</script>

<template>
  <div class="flex-1 flex flex-col relative overflow-hidden">

    <!-- Ambient glow -->
    <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full bg-brand-vibrant/[0.08] blur-[150px] pointer-events-none" />

    <!-- Save flash -->
    <Transition name="flash">
      <div
        v-if="flash"
        class="fixed top-20 left-1/2 -translate-x-1/2 z-50 bg-brand-success/15 border border-brand-success/40 rounded-xl px-5 py-3 backdrop-blur-md"
      >
        <p class="text-brand-success font-semibold text-sm">{{ flash }}</p>
      </div>
    </Transition>

    <!-- ═══════════════════ TOP HEADER (centered) ═══════════════════ -->
    <header class="relative z-10 shrink-0 px-8 pt-6 pb-5 border-b border-white/5">
      <button
        @click="router.push('/')"
        class="absolute left-8 top-6 text-white/40 hover:text-brand-accent transition-colors text-sm flex items-center gap-2 cursor-pointer"
      >
        <i class="pi pi-arrow-left text-xs" /> Back to home
      </button>
      <div class="text-center">
        <div class="flex items-center justify-center gap-3">
          <span class="text-3xl">📸</span>
          <h1 class="text-4xl font-black gradient-text">Collect Data</h1>
        </div>
        <p class="text-white/40 mt-2 max-w-xl mx-auto">
          Help SignSee recognise NGT fingerspelling better. Capture a sign, tell us the letter,
          and add it to the training set.
        </p>
      </div>
    </header>

    <!-- ═══════════════════ THREE COLUMNS ═══════════════════ -->
    <div class="flex-1 flex min-h-0 relative z-10">

      <!-- LEFT RAIL -->
      <aside class="w-[280px] shrink-0 flex flex-col p-7 border-r border-white/5 overflow-y-auto">
        <!-- Mode selector (made prominent so it isn't missed) -->
        <div class="flex items-center gap-2 mb-1">
          <span class="text-base">🎬</span>
          <p class="text-sm font-bold text-white uppercase tracking-wide">Choose a capture mode</p>
        </div>
        <p class="text-xs text-white/40 mb-3">Pick how you'd like to add a sample:</p>
        <div class="flex flex-col gap-2.5 mb-8 p-2.5 rounded-2xl bg-brand-vibrant/[0.06] border border-brand-vibrant/20">
          <button
            v-for="m in modes"
            :key="m.id"
            @click="mode = m.id"
            class="rounded-xl px-4 py-3 text-left border transition-all duration-200 cursor-pointer"
            :class="mode === m.id
              ? 'bg-brand-vibrant/25 border-brand-vibrant/60 shadow-[0_0_18px_rgba(168,85,247,0.25)]'
              : 'bg-white/5 border-white/10 hover:border-brand-vibrant/40 hover:bg-white/[0.07]'"
          >
            <div class="flex items-center gap-3">
              <span class="text-xl">{{ m.icon }}</span>
              <div>
                <p class="font-bold text-sm" :class="mode === m.id ? 'text-white' : 'text-white/70'">{{ m.label }}</p>
                <p class="text-[11px] text-white/35 mt-0.5">{{ m.hint }}</p>
              </div>
              <i v-if="mode === m.id" class="pi pi-check-circle text-brand-accent text-sm ml-auto" />
            </div>
          </button>
        </div>

        <!-- Letter picker (camera + upload; auto uses the prediction) -->
        <template v-if="mode !== 'auto'">
          <p class="text-xs text-white/30 uppercase tracking-widest mb-3">Which letter?</p>
          <div class="grid grid-cols-5 gap-2">
            <button
              v-for="letter in alphabet"
              :key="letter"
              @click="selectedLetter = letter"
              class="aspect-square rounded-lg font-bold text-sm transition-all cursor-pointer"
              :class="selectedLetter === letter
                ? 'bg-brand-vibrant text-white shadow-[0_0_12px_rgba(168,85,247,0.5)] scale-105'
                : 'bg-white/5 text-white/50 hover:bg-white/10 hover:text-white/80'"
            >
              {{ letter }}
            </button>
          </div>
        </template>
        <div v-else class="bg-brand-vibrant/10 border border-brand-vibrant/25 rounded-xl p-4">
          <p class="text-xs text-brand-accent/80 leading-relaxed">
            ✨ The letter comes from the live model prediction. Hold a sign steady, then save it.
          </p>
        </div>

      </aside>

      <!-- CENTER STAGE -->
      <main class="flex-1 flex flex-col p-7 min-w-0 overflow-y-auto">

        <!-- Camera + reference, side by side -->
        <div class="flex gap-6 min-h-0" style="min-height: 560px;">

          <!-- Camera column -->
          <div class="flex-1 flex flex-col min-w-0">

            <!-- Camera / capture surface fills the available height -->
            <div class="flex-1 relative rounded-2xl overflow-hidden bg-black/40 border border-white/10 flex items-center justify-center min-h-0">
              <!-- Camera + Auto -->
              <template v-if="mode === 'camera' || mode === 'auto'">
                <video
                  ref="videoRef"
                  autoplay
                  playsinline
                  muted
                  class="w-full h-full object-cover -scale-x-100"
                  :class="{ 'opacity-0': !isCameraActive }"
                />
                <canvas ref="canvasRef" class="absolute inset-0 w-full h-full object-cover -scale-x-100 pointer-events-none" />

                <div v-if="isCameraActive" class="absolute inset-0 pointer-events-none flex items-center justify-center">
                  <div class="guide-box" :class="{ 'guide-box--active': handDetected }">
                    <span class="guide-corner guide-corner--tl" />
                    <span class="guide-corner guide-corner--tr" />
                    <span class="guide-corner guide-corner--bl" />
                    <span class="guide-corner guide-corner--br" />
                    <span
                      class="absolute -bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap text-sm font-bold px-4 py-1.5 rounded-full shadow-lg backdrop-blur-md"
                      :class="handDetected
                        ? 'bg-brand-success/30 text-brand-success border border-brand-success/60'
                        : 'bg-black/80 text-white border border-white/30'"
                    >
                      {{ handDetected ? '✋ Hand detected — will crop around your hand' : 'Place your whole hand in the middle of the box' }}
                    </span>
                  </div>
                </div>

                <div v-if="!isCameraActive" class="absolute inset-0 flex flex-col items-center justify-center gap-4">
                  <span class="text-6xl opacity-25">📷</span>
                  <p class="text-white/40">Camera is off</p>
                  <button
                    @click="enableCamera"
                    class="mt-2 flex items-center gap-2 bg-brand-vibrant hover:bg-purple-600 text-white font-bold rounded-xl px-6 py-3 transition-colors cursor-pointer shadow-[0_0_20px_rgba(168,85,247,0.3)]"
                  >
                    <i class="pi pi-video" /> Turn on camera
                  </button>
                </div>

                <div
                  v-if="mode === 'auto' && isCameraActive && liveLetter"
                  class="absolute top-5 left-5 bg-black/70 backdrop-blur-md rounded-2xl px-5 py-3 border border-brand-vibrant/40"
                >
                  <p class="text-[10px] text-white/50 uppercase tracking-widest">Model sees</p>
                  <p class="text-4xl font-black text-brand-accent leading-tight">{{ liveLetter }}</p>
                  <p class="text-xs" :class="liveConfidence > 0.8 ? 'text-brand-success' : 'text-amber-400'">
                    {{ (liveConfidence * 100).toFixed(0) }}% confident
                  </p>
                </div>
              </template>

              <template v-else>
                <img v-if="uploadedImage" :src="uploadedImage" class="w-full h-full object-contain" alt="Uploaded sign" />
                <div v-else class="flex flex-col items-center gap-4">
                  <span class="text-6xl opacity-25">🖼️</span>
                  <p class="text-white/40">No image selected</p>
                  <input ref="fileInputRef" type="file" accept="image/*" class="hidden" @change="onFilePicked" />
                  <button
                    @click="fileInputRef?.click()"
                    class="mt-2 flex items-center gap-2 bg-brand-vibrant hover:bg-purple-600 text-white font-bold rounded-xl px-6 py-3 transition-colors cursor-pointer shadow-[0_0_20px_rgba(168,85,247,0.3)]"
                  >
                    <i class="pi pi-upload" /> Choose an image
                  </button>
                </div>
              </template>
            </div>

            <div v-if="cameraError" class="mt-3 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 text-xs text-red-300 shrink-0">
              {{ cameraError }}
            </div>

            <div class="mt-5 flex items-center gap-3 shrink-0">
              <template v-if="(mode === 'camera' || mode === 'auto') && isCameraActive">
                <button
                  @click="disableCamera"
                  class="flex items-center justify-center gap-2 bg-white/10 hover:bg-white/15 text-white/60 font-semibold rounded-xl px-5 py-3.5 transition-colors cursor-pointer shrink-0"
                >
                  <i class="pi pi-stop-circle text-sm" /> Stop camera
                </button>
              </template>
              <template v-else-if="mode === 'upload' && uploadedImage">
                <button
                  @click="fileInputRef?.click()"
                  class="flex items-center justify-center gap-2 bg-white/10 hover:bg-white/15 text-white/70 font-semibold rounded-xl px-5 py-3.5 transition-colors cursor-pointer shrink-0"
                >
                  <i class="pi pi-refresh text-sm" /> Change image
                </button>
              </template>

              <button
                @click="submit"
                :disabled="!canSubmit"
                class="flex-1 font-bold rounded-xl py-3.5 transition-all text-base"
                :class="canSubmit
                  ? 'bg-brand-success hover:bg-green-500 text-white cursor-pointer shadow-[0_0_24px_rgba(74,222,128,0.3)]'
                  : 'bg-white/5 text-white/25 cursor-not-allowed'"
              >
                <template v-if="mode === 'auto'">Save the model's guess ✨</template>
                <template v-else-if="!selectedLetter">Pick a letter first</template>
                <template v-else>Add sample for "{{ selectedLetter }}"</template>
              </button>
            </div>
          </div>

          <aside
            v-if="referenceLetter"
            class="w-[340px] shrink-0 flex flex-col gap-3 rounded-2xl border border-white/8 bg-white/5 p-4 overflow-y-auto"
          >
            <div class="flex items-center gap-2 text-xs uppercase tracking-widest text-white/40">
              <i class="pi pi-book text-brand-accent/60" />
              Reference · {{ referenceLetter }}
              <span v-if="mode === 'auto'" class="text-[10px] text-brand-accent/50 normal-case tracking-normal">(live guess)</span>
            </div>

            <div class="rounded-xl border border-white/8 bg-white/5 p-3">
              <div class="flex items-center gap-2 mb-1.5">
                <i class="pi pi-lightbulb text-brand-accent/60 text-xs" />
                <p class="text-[10px] uppercase tracking-widest text-white/25">How to sign {{ referenceLetter }}</p>
              </div>
              <p class="text-xs text-white/50 leading-relaxed">{{ currentHint }}</p>
            </div>

            <div class="flex gap-2 p-1 bg-white/5 border border-white/8 rounded-xl">
              <button
                @click="showPhoto = false"
                class="flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer"
                :class="!showPhoto
                  ? 'bg-brand-vibrant text-white shadow-[0_0_12px_rgba(168,85,247,0.3)]'
                  : 'text-white/40 hover:text-white/60'"
              >
                <i class="pi pi-image mr-1" /> Illustration
              </button>
              <button
                @click="showPhoto = true"
                class="flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer"
                :class="showPhoto
                  ? 'bg-brand-vibrant text-white shadow-[0_0_12px_rgba(168,85,247,0.3)]'
                  : 'text-white/40 hover:text-white/60'"
              >
                <i class="pi pi-camera mr-1" /> Photo
              </button>
            </div>

            <div class="flex-1 rounded-xl border border-white/8 bg-white/5 overflow-hidden flex items-center justify-center" style="min-height: 240px;">
              <img
                :src="showPhoto ? photoSrc : illustrationSrc"
                :alt="`Sign for letter ${referenceLetter}`"
                class="w-full object-contain p-3 transition-opacity duration-200"
              />
            </div>
          </aside>

          <aside
            v-else
            class="w-[340px] shrink-0 flex flex-col items-center justify-center gap-3 rounded-2xl border border-white/8 bg-white/5 p-4 text-center"
          >
            <span class="text-4xl opacity-25">👆</span>
            <p v-if="mode === 'auto'" class="text-white/35 text-sm">Sign a letter — the model will show its reference here</p>
            <p v-else class="text-white/35 text-sm">Pick a letter to see how to sign it</p>
          </aside>
        </div>

        <div class="mt-8 pt-6 border-t border-white/8">
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center gap-3">
              <p class="text-2xl font-black text-brand-accent leading-none">{{ samples.length }}</p>
              <p class="text-sm font-bold text-white">Your contributions</p>
              <span class="text-2xl opacity-40">🐙</span>
            </div>
            <button
              v-if="samples.length"
              @click="clearAll"
              class="text-xs text-white/30 hover:text-red-300 transition-colors cursor-pointer"
            >
              Clear all
            </button>
          </div>

          <div v-if="samples.length === 0" class="flex flex-col items-center justify-center text-center gap-3 py-10">
            <span class="text-5xl opacity-20">🐙</span>
            <p class="text-white/30 text-sm">No samples yet — capture your first sign and it'll appear here!</p>
          </div>

          <template v-else>
            <div class="flex flex-wrap gap-1.5 mb-4">
              <span
                v-for="(count, letter) in tally"
                :key="letter"
                class="bg-brand-vibrant/15 border border-brand-vibrant/30 rounded-lg px-2.5 py-1 text-xs text-brand-accent font-semibold"
              >
                {{ letter }} ×{{ count }}
              </span>
            </div>

            <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2.5">
              <button
                v-for="s in samples"
                :key="s.id"
                @click="preview = s"
                class="relative rounded-xl overflow-hidden bg-black/30 aspect-square group cursor-pointer border border-white/5"
              >
                <img :src="s.thumbnail" class="w-full h-full object-cover transition-transform group-hover:scale-105" :alt="`Sample for ${s.letter}`" />
                <div class="absolute inset-0 bg-black/0 group-hover:bg-black/25 transition-colors flex items-center justify-center">
                  <i class="pi pi-search-plus text-white/0 group-hover:text-white/80 transition-colors" />
                </div>
                <div class="absolute bottom-0 inset-x-0 bg-black/70 px-2 py-1 flex items-center justify-between">
                  <span class="text-xs font-bold text-white">{{ s.letter }}</span>
                  <span class="text-[10px]">{{ s.stored ? 'Cloud' : 'Local only' }} · {{ sourceIcon(s.source) }}</span>
                </div>
              </button>
            </div>
          </template>
        </div>
      </main>
    </div>

    <!-- ═══════════════════ ENLARGE MODAL ═══════════════════ -->
    <Transition name="modal">
      <div
        v-if="preview"
        class="fixed inset-0 z-[60] bg-black/80 backdrop-blur-sm flex items-center justify-center p-8"
        @click="preview = null"
      >
        <div class="relative max-w-lg w-full bg-brand-purple border border-brand-vibrant/30 rounded-2xl overflow-hidden" @click.stop>
          <img :src="preview.thumbnail" class="w-full object-contain bg-black/40" :alt="`Sample for ${preview.letter}`" />
          <div class="p-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
              <span class="text-3xl font-black text-brand-accent">{{ preview.letter }}</span>
              <div>
                <p class="text-xs text-white/60">{{ sourceLabel(preview.source) }} {{ sourceIcon(preview.source) }}</p>
                <p class="text-[10px] text-white/30">Captured at {{ preview.at }}</p>
                <p class="text-[10px] text-white/30">{{ preview.stored ? 'Stored in Azure' : 'Local only' }}</p>
              </div>
            </div>
            <button
              @click="removeSample(preview.id); preview = null"
              class="text-xs text-white/40 hover:text-red-300 transition-colors cursor-pointer flex items-center gap-1"
            >
              <i class="pi pi-trash text-xs" /> Delete
            </button>
          </div>
          <button
            @click="preview = null"
            class="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/50 hover:bg-black/70 flex items-center justify-center cursor-pointer transition-colors"
          >
            <i class="pi pi-times text-white/80 text-sm" />
          </button>
        </div>
      </div>
    </Transition>
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
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(20px);
}

/* ── Hand guide box ──────────────────────────────────────────────────── */
.guide-box {
  position: relative;
  /* square sized to ~60% of the camera's shorter side; aspect-video means
     height is the shorter side, so base it on height via vmin-like sizing */
  width: 46%;
  aspect-ratio: 1 / 1;
  max-height: 78%;
  border: 2px dashed rgba(255, 255, 255, 0.35);
  border-radius: 18px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.guide-box--active {
  border-color: rgba(74, 222, 128, 0.7);
  border-style: solid;
  box-shadow: 0 0 24px rgba(74, 222, 128, 0.25), inset 0 0 24px rgba(74, 222, 128, 0.08);
}

.guide-corner {
  position: absolute;
  width: 22px;
  height: 22px;
  border-color: rgba(168, 85, 247, 0.9);
  border-style: solid;
  border-width: 0;
}
.guide-box--active .guide-corner { border-color: rgba(74, 222, 128, 0.95); }
.guide-corner--tl { top: -2px; left: -2px; border-top-width: 3px; border-left-width: 3px; border-top-left-radius: 18px; }
.guide-corner--tr { top: -2px; right: -2px; border-top-width: 3px; border-right-width: 3px; border-top-right-radius: 18px; }
.guide-corner--bl { bottom: -2px; left: -2px; border-bottom-width: 3px; border-left-width: 3px; border-bottom-left-radius: 18px; }
.guide-corner--br { bottom: -2px; right: -2px; border-bottom-width: 3px; border-right-width: 3px; border-bottom-right-radius: 18px; }

.flash-enter-active, .flash-leave-active { transition: all 0.3s ease; }
.flash-enter-from, .flash-leave-to { opacity: 0; transform: translate(-50%, -10px); }

.modal-enter-active, .modal-leave-active { transition: opacity 0.2s ease; }
.modal-enter-from, .modal-leave-to { opacity: 0; }
</style>
