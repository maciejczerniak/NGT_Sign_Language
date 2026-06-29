import { ref, onUnmounted } from 'vue';
import type { Ref } from 'vue';

// ── Types ──────────────────────────────────────────────────────────────
interface HandPrediction {
  hand_id: string;
  label: string;
  predicted_letter: string;
  stable_letter: string | null;
  confidence: number;
  alternatives: { letter: string; confidence: number }[];
  landmarks: { x: number; y: number; z: number }[] | null;
  wrist_x: number;
  wrist_y: number;
}

interface EndpointHandPrediction {
  hand_id: string;
  label: string;
  predicted_letter: string;
  stable_letter?: string | null;
  stable_confidence?: number;
  confidence: number;
  top_3?: { letter: string; confidence: number }[];
  committed_letter?: string | null;
  landmarks?: { x: number; y: number; z: number }[] | null;
}

interface PredictionResponse {
  error?: string;
  ok?: boolean;
  hands?: EndpointHandPrediction[];
}

interface HistoryEntry {
  letter: string;
  confidence: number;
  timestamp: string;
  hand_id: string;
  hand_label: string;
}

export function useCamera(
  videoRef: Ref<HTMLVideoElement | null>,
  canvasRef: Ref<HTMLCanvasElement | null>,
) {
  // ── State ────────────────────────────────────────────────────────────
  const isCameraActive = ref(false);
  const cameraError = ref('');
  const connectionStatus = ref<'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'>('idle');
  const connectionMessage = ref('');
  const activeHands = ref<HandPrediction[]>([]);
  const history = ref<HistoryEntry[]>([]);

  // ── Internal variables ───────────────────────────────────────────────
  let socket: WebSocket | null = null;
  let captureInterval: number | null = null;
  let reconnectTimeout: number | null = null;
  let reconnectAttempts = 0;
  let awaitingResponse = false;
  const captureCanvas = document.createElement('canvas');

  // ── Constants ────────────────────────────────────────────────────────
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
  ];

  const HAND_COLORS = ['#a78bfa', '#34d399', '#f472b6', '#fbbf24', '#60a5fa'];
  const DEFAULT_HAND_COLOR = '#a78bfa';

  const handColor = (handId: string): string => {
    const match = handId.match(/(\d+)$/);
    const idx = match?.[1] ? parseInt(match[1], 10) : 0;
    return HAND_COLORS[idx % HAND_COLORS.length] ?? DEFAULT_HAND_COLOR;
  };

  const handOverlayPosition = (hand: HandPrediction): Record<string, string> => {
    const wrist = hand.landmarks?.[0];
    if (!wrist) {
      return { top: '24px', left: '24px' };
    }
    const xPercent = (1 - wrist.x) * 100;
    const yPercent = wrist.y * 100;
    const clampedX = Math.max(2, Math.min(xPercent, 75));
    const clampedY = Math.max(2, Math.min(yPercent - 15, 70));
    return {
      left: `${clampedX}%`,
      top: `${clampedY}%`,
    };
  };

  // ── Drawing ──────────────────────────────────────────────────────────
  const drawAllSkeletons = (hands: EndpointHandPrediction[]) => {
    const canvas = canvasRef.value;
    const video = videoRef.value;
    if (!canvas || !video) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const hand of hands) {
      if (!hand.landmarks) continue;
      const color = handColor(hand.hand_id);
      drawSingleSkeleton(ctx, canvas.width, canvas.height, hand.landmarks, color, hand.confidence);
    }
  };

  const drawSingleSkeleton = (
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    landmarks: { x: number; y: number }[],
    color: string,
    confidence: number
  ) => {
    const strokeColor = confidence > 0.9 ? '#22c55e' : color;
    ctx.strokeStyle = strokeColor;
    ctx.fillStyle = 'white';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';

    HAND_CONNECTIONS.forEach(([s, e]) => {
      const start = landmarks[s as number];
      const end = landmarks[e as number];
      if (start && end) {
        ctx.beginPath();
        ctx.moveTo(start.x * w, start.y * h);
        ctx.lineTo(end.x * w, end.y * h);
        ctx.stroke();
      }
    });

    landmarks.forEach(p => {
      if (p) {
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  };

  // ── WebSocket ────────────────────────────────────────────────────────
  const connectWebSocket = () => {
    if (
      socket &&
      (socket.readyState === WebSocket.CONNECTING ||
        socket.readyState === WebSocket.OPEN)
    ) return;

    connectionStatus.value = 'connecting';
    connectionMessage.value = reconnectAttempts > 0 ? 'Reconnecting...' : 'Connecting...';
    socket = new WebSocket('/ws/predict');

    socket.onopen = () => {
      reconnectAttempts = 0;
      connectionStatus.value = 'connected';
      connectionMessage.value = '';
    };

    socket.onmessage = (event) => {
      awaitingResponse = false;
      const data = JSON.parse(event.data) as PredictionResponse;

      if (data.error || data.ok) return;

      const hands = data.hands ?? [];

      activeHands.value = hands.map(h => ({
        hand_id: h.hand_id,
        label: h.label,
        predicted_letter: h.predicted_letter,
        stable_letter: h.stable_letter ?? null,
        confidence: h.confidence,
        alternatives: h.top_3?.slice(1, 3) || [],
        landmarks: h.landmarks || null,
        wrist_x: h.landmarks?.[0]?.x ?? 0,
        wrist_y: h.landmarks?.[0]?.y ?? 0,
      }));

      if (hands.length === 0) {
        activeHands.value = [];
      }

      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

      for (const h of hands) {
        if (h.committed_letter) {
          history.value.unshift({
            letter: h.committed_letter,
            confidence: h.stable_confidence ?? h.confidence,
            timestamp: timeStr,
            hand_id: h.hand_id,
            hand_label: h.label,
          });

          if (history.value.length > 15) history.value.pop();
        }
      }

      drawAllSkeletons(hands);
    };

    socket.onerror = () => {
      connectionStatus.value = 'error';
      connectionMessage.value = 'Prediction connection error';
      socket?.close();
    };

    socket.onclose = () => {
      socket = null;
      if (!isCameraActive.value) return;
      scheduleReconnect();
    };
  };

  const scheduleReconnect = () => {
    if (reconnectTimeout) return;
    reconnectAttempts += 1;
    const delay = Math.min(1000 * 2 ** (reconnectAttempts - 1), 10000);
    connectionStatus.value = 'disconnected';
    connectionMessage.value = `Connection lost. Retrying in ${Math.round(delay / 1000)}s...`;
    reconnectTimeout = window.setTimeout(() => {
      reconnectTimeout = null;
      if (isCameraActive.value) connectWebSocket();
    }, delay);
  };

  // ── Frame capture ────────────────────────────────────────────────────
  const sendFrame = () => {
    if (!socket || socket.readyState !== WebSocket.OPEN || !videoRef.value) return;
    if (videoRef.value.videoWidth === 0) return;
    if (awaitingResponse) return;

    captureCanvas.width = videoRef.value.videoWidth;
    captureCanvas.height = videoRef.value.videoHeight;
    const ctx = captureCanvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.value, 0, 0);
      socket.send(JSON.stringify({ image: captureCanvas.toDataURL('image/jpeg', 0.5) }));
      awaitingResponse = true;
    }
  };

  // ── Camera lifecycle ─────────────────────────────────────────────────
  async function enableCamera() {
    cameraError.value = '';
    awaitingResponse = false;

    if (!navigator.mediaDevices?.getUserMedia) {
      cameraError.value = window.isSecureContext
        ? 'Camera access is not available in this browser.'
        : 'Camera access requires HTTPS. Open the app over HTTPS or use localhost for local testing.';
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } });
      if (videoRef.value) {
        videoRef.value.srcObject = stream;
        isCameraActive.value = true;
        connectWebSocket();
        captureInterval = window.setInterval(sendFrame, 100);
      }
    } catch (err) {
      console.error(err);
      const errorName = err instanceof DOMException ? err.name : '';
      if (errorName === 'NotAllowedError') {
        cameraError.value = 'Camera permission was denied. Allow camera access in the browser site settings.';
      } else if (errorName === 'NotFoundError') {
        cameraError.value = 'No camera was found on this device.';
      } else if (errorName === 'NotReadableError') {
        cameraError.value = 'The camera is already in use by another app or browser tab.';
      } else {
        cameraError.value = 'Camera could not be started. Check browser permissions and HTTPS.';
      }
    }
  }

  function disableCamera() {
    isCameraActive.value = false;
    cameraError.value = '';
    if (videoRef.value?.srcObject) {
      (videoRef.value.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      videoRef.value.srcObject = null;
    }
    if (captureInterval) clearInterval(captureInterval);
    if (reconnectTimeout) clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
    reconnectAttempts = 0;
    socket?.close();
    socket = null;
    activeHands.value = [];
    connectionStatus.value = 'idle';
    connectionMessage.value = '';
  }

  const deleteLastPrediction = () => {
    if (history.value.length > 0) {
      history.value.shift();
    }
  };

  // ── Cleanup ──────────────────────────────────────────────────────────
  onUnmounted(disableCamera);

  // ── Public API ───────────────────────────────────────────────────────
  return {
    // State
    isCameraActive,
    cameraError,
    connectionStatus,
    connectionMessage,
    activeHands,
    history,
    // Methods
    enableCamera,
    disableCamera,
    deleteLastPrediction,
    handColor,
    handOverlayPosition,
  };
}
