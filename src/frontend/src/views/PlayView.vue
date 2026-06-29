<script setup lang="ts">
import { useRouter } from 'vue-router';

const router = useRouter();

const gameModes = [
  {
    id: 'random-letters',
    title: 'Random Letters',
    description: 'Practice random NGT letters',
    difficulty: 'Easy',
    difficultyColor: 'text-green-400 bg-green-400/15 border-green-400/30',
    icon: 'pi-th-large',
    iconBg: 'bg-blue-500/20 text-blue-400',
    enabled: true,
    route: '/play/random-letters',
  },
  {
    id: 'spell-words',
    title: 'Spell Words',
    description: 'Sign complete words letter by letter',
    difficulty: 'Medium',
    difficultyColor: 'text-yellow-400 bg-yellow-400/15 border-yellow-400/30',
    icon: 'pi-file-edit',
    iconBg: 'bg-orange-500/20 text-orange-400',
    enabled: false,
    route: '/play/spell-words',
  },
  {
    id: 'speed-challenge',
    title: 'Speed Challenge',
    description: 'Race against the clock!',
    difficulty: 'Hard',
    difficultyColor: 'text-red-400 bg-red-400/15 border-red-400/30',
    icon: 'pi-bolt',
    iconBg: 'bg-amber-500/20 text-amber-400',
    enabled: false,
    route: '/play/speed-challenge',
  },
];

function selectMode(mode: (typeof gameModes)[number]) {
  if (mode.enabled) {
    router.push(mode.route);
  }
}
</script>

<template>
  <div class="flex-1 flex items-center justify-center p-6">
    <div class="max-w-3xl w-full">

      <!-- Heading -->
      <h2 class="text-2xl font-bold text-center text-white mb-8">Select a Game Mode</h2>

      <!-- Mode cards -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <button
          v-for="mode in gameModes"
          :key="mode.id"
          @click="selectMode(mode)"
          class="group relative rounded-2xl border p-6 text-center transition-all duration-200"
          :class="mode.enabled
            ? 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-brand-vibrant/40 hover:shadow-[0_0_20px_rgba(168,85,247,0.15)] cursor-pointer'
            : 'bg-white/[0.02] border-white/5 opacity-50 cursor-not-allowed'"
        >
          <!-- Icon -->
          <div
            class="w-14 h-14 rounded-full flex items-center justify-center mx-auto mb-4"
            :class="mode.iconBg"
          >
            <i class="pi text-xl" :class="mode.icon" />
          </div>

          <!-- Title -->
          <h3 class="text-lg font-bold text-white mb-1">{{ mode.title }}</h3>

          <!-- Description -->
          <p class="text-sm text-white/40 mb-4">{{ mode.description }}</p>

          <!-- Difficulty badge -->
          <span
            class="inline-block text-xs font-semibold px-3 py-1 rounded-full border"
            :class="mode.difficultyColor"
          >
            {{ mode.difficulty }}
          </span>

          <!-- Coming soon overlay for disabled modes -->
          <span
            v-if="!mode.enabled"
            class="absolute top-3 right-3 text-[10px] uppercase tracking-wider text-white/20 font-semibold"
          >
            Soon
          </span>
        </button>
      </div>
    </div>
  </div>
</template>
