import { useEffect, useRef, useState, type ChangeEvent, type PointerEvent as ReactPointerEvent, type MutableRefObject } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { motion, AnimatePresence } from 'framer-motion'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { classifyGesture, type GestureClassification, type HandLandmark } from './gestures/gestureClassifier'

import ModelViewer from './components/viewport/ModelViewer'

import {
  Atom,
  ArrowLeft,
  ArrowRight,
  Box,
  ChevronDown,
  ChevronRight,
  Camera,
  GripVertical,
  Minus,
  Plus,
  VideoOff,
  Eye,
  FileBox,
  Layers3,
  MousePointer2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  RotateCcw,
  Rotate3D,
  Sparkles,
  Upload,
  X,
} from 'lucide-react'

interface ComponentInfo {
  id: string
  name: string
  geometry: string
  vertices: number
}

type ViewMode = 'Explore' | 'Inspect' | 'Disassemble'
type GestureActionState = 'idle' | 'rotating' | 'grabbing' | 'zooming' | 'locked'

/* -----------------------------
   LANDING PAGE
----------------------------- */

function LandingPage({ onStart }: { onStart: () => void }) {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[#05080d] text-white">
      <div className="absolute inset-0 grid-background opacity-50" />

      <div className="absolute left-1/2 top-1/3 h-[450px] w-[450px] -translate-x-1/2 rounded-full bg-cyan-500/[0.07] blur-[140px]" />

      <nav className="relative z-10 flex h-20 items-center justify-between border-b border-white/[0.06] px-6 lg:px-12">
        <div className="flex items-center gap-3">
          <Atom className="text-cyan-400" size={27} />
          <span className="text-xl font-bold tracking-[0.25em]">
            NEXUS<span className="text-cyan-400">.</span>
          </span>
        </div>

        <button
          onClick={onStart}
          className="flex items-center gap-2 rounded-lg border border-cyan-400/30 px-5 py-2.5 text-sm text-cyan-300 transition hover:bg-cyan-400/10"
        >
          Open Workspace <ArrowRight size={16} />
        </button>
      </nav>

      <section className="relative z-10 mx-auto flex min-h-[calc(100vh-80px)] max-w-7xl flex-col justify-center px-6 py-24">
        <div className="max-w-3xl">
          <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/[0.06] px-4 py-2 text-xs tracking-widest text-cyan-300">
            <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />
            THE NEXT DIMENSION OF EXPLORATION
          </div>

          <h1 className="text-6xl font-semibold leading-[1.1] tracking-tight sm:text-8xl">
            Don't just
            <br />
            <span className="text-gradient">look at it.</span>
            <br />
            Explore it.
          </h1>

          <p className="mt-8 max-w-xl text-base leading-8 text-slate-400 sm:text-lg">
            Step inside the structure of anything. Explore 3D models,
            uncover their components, and interact with digital objects
            using natural hand movements.
          </p>

          <button
            onClick={onStart}
            className="mt-10 flex items-center gap-3 rounded-xl bg-cyan-400 px-7 py-4 font-semibold text-[#031019] transition hover:bg-cyan-300"
          >
            Start Exploring <ArrowRight size={18} />
          </button>
        </div>

        <div className="mt-20 grid gap-5 border-t border-white/[0.07] pt-8 sm:grid-cols-3">
          {[
            ['01', '3D Exploration', 'Explore digital structures.'],
            ['02', 'Hand Interaction', 'Interact without controllers.'],
            ['03', 'Deep Inspection', 'Discover individual components.'],
          ].map(([number, title, description]) => (
            <div key={number}>
              <p className="text-xs tracking-widest text-cyan-400">{number}</p>
              <h3 className="mt-3 font-medium text-slate-200">{title}</h3>
              <p className="mt-2 text-sm text-slate-500">{description}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  )
}

/* -----------------------------
   WEBCAM + HAND TRACKING — PHASE 4D
----------------------------- */

interface SmoothedGestureLabel {
  handIndex: number
  gesture: string
  confidence: number
  isHeld: boolean
}

type TrackedGesture = GestureClassification['gesture'] | 'V-Zoom'

interface HandTemporalState {
  landmarks: HandLandmark[] | null
  activeGesture: TrackedGesture | null
  activeConfidence: number
  candidateGesture: TrackedGesture | null
  candidateFrames: number
  releaseFrames: number
  missingFrames: number
}

const createHandTemporalState = (): HandTemporalState => ({
  landmarks: null,
  activeGesture: null,
  activeConfidence: 0,
  candidateGesture: null,
  candidateFrames: 0,
  releaseFrames: 0,
  missingFrames: 0,
})

const smoothHandLandmarks = (
  previous: HandLandmark[] | null,
  current: readonly HandLandmark[],
  alpha: number,
): HandLandmark[] => {
  if (!previous || previous.length !== current.length) {
    return current.map((point) => ({ x: point.x, y: point.y, z: point.z ?? 0 }))
  }

  return current.map((point, index) => {
    const oldPoint = previous[index]
    return {
      x: oldPoint.x + alpha * (point.x - oldPoint.x),
      y: oldPoint.y + alpha * (point.y - oldPoint.y),
      z: oldPoint.z + alpha * ((point.z ?? 0) - oldPoint.z),
    }
  })
}

interface HandCursorPosition {
  x: number
  y: number
  visible: boolean
}

/** Pinch-selects a mesh, then moves it on a camera-facing plane while the pinch is held. */
function GestureRaycastController({
  scene,
  cursorRef,
  pinchActive,
  onSelect,
}: {
  scene: THREE.Object3D
  cursorRef: MutableRefObject<HandCursorPosition>
  pinchActive: boolean
  onSelect: (id: string) => void
}) {
  const { camera, size } = useThree()
  const raycaster = useRef(new THREE.Raycaster())
  const pointer = useRef(new THREE.Vector2())
  const dragPlane = useRef(new THREE.Plane())
  const planeHit = useRef(new THREE.Vector3())
  const dragOffset = useRef(new THREE.Vector3())
  const grabbedMesh = useRef<THREE.Mesh | null>(null)
  const wasPinching = useRef(false)
  const cameraDirection = useRef(new THREE.Vector3())
  const targetWorldPosition = useRef(new THREE.Vector3())

  useFrame(() => {
    const cursor = cursorRef.current
    if (!pinchActive || !cursor.visible || size.width <= 0 || size.height <= 0) {
      grabbedMesh.current = null
      wasPinching.current = false
      return
    }

    pointer.current.set(cursor.x * 2 - 1, 1 - cursor.y * 2)
    scene.updateWorldMatrix(true, true)
    raycaster.current.setFromCamera(pointer.current, camera)

    // A new pinch begins by selecting the mesh directly under the hand cursor.
    if (!wasPinching.current) {
      wasPinching.current = true
      const intersections = raycaster.current.intersectObject(scene, true)
      const hit = intersections.find(
        (intersection) => intersection.object instanceof THREE.Mesh,
      )
      if (!hit || !(hit.object instanceof THREE.Mesh)) return

      const mesh = hit.object
      grabbedMesh.current = mesh
      onSelect(mesh.uuid)

      // Keep the mesh origin at its existing offset from the cursor to avoid a jump.
      mesh.getWorldPosition(targetWorldPosition.current)
      camera.getWorldDirection(cameraDirection.current)
      dragPlane.current.setFromNormalAndCoplanarPoint(
        cameraDirection.current,
        targetWorldPosition.current,
      )
      if (raycaster.current.ray.intersectPlane(dragPlane.current, planeHit.current)) {
        dragOffset.current.copy(targetWorldPosition.current).sub(planeHit.current)
      } else {
        grabbedMesh.current = null
      }
      return
    }

    // Continue dragging the same mesh until the pinch is released or tracking is lost.
    const mesh = grabbedMesh.current
    if (!mesh || !mesh.parent) return
    if (!raycaster.current.ray.intersectPlane(dragPlane.current, planeHit.current)) return

    targetWorldPosition.current.copy(planeHit.current).add(dragOffset.current)
    mesh.parent.worldToLocal(targetWorldPosition.current)
    mesh.position.copy(targetWorldPosition.current)
  })

  return null
}

/** Detect the dedicated V-shaped zoom pose. Thumb position is intentionally ignored. */
function isVShapeZoomGesture(landmarks: readonly HandLandmark[]): boolean {
  if (landmarks.length !== 21) return false

  const wrist = landmarks[0]
  const middleMcp = landmarks[9]
  const handScale = Math.hypot(
    wrist.x - middleMcp.x,
    wrist.y - middleMcp.y,
    wrist.z - middleMcp.z,
  )
  if (!Number.isFinite(handScale) || handScale < 0.0001) return false

  const wristDistance = (index: number) => Math.hypot(
    landmarks[index].x - wrist.x,
    landmarks[index].y - wrist.y,
    landmarks[index].z - wrist.z,
  )

  const isExtended = (tip: number, pip: number) =>
    wristDistance(tip) > wristDistance(pip) + handScale * 0.12

  const isCurled = (tip: number, pip: number) =>
    wristDistance(tip) < wristDistance(pip) + handScale * 0.08

  return (
    isExtended(8, 6) &&
    isExtended(12, 10) &&
    isCurled(16, 14) &&
    isCurled(20, 18)
  )
}

function CameraPreview({
  onRotateDelta,
  onZoomDelta,
  onCursorUpdate,
  onPinchStateChange,
}: {
  onRotateDelta: (deltaX: number, deltaY: number) => void
  onZoomDelta: (delta: number) => void
  onCursorUpdate: (cursor: HandCursorPosition) => void
  onPinchStateChange: (active: boolean) => void
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const landmarkerRef = useRef<HandLandmarker | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const mountedRef = useRef(true)
  const lastVideoTimeRef = useRef(-1)
  const lastHandsCountRef = useRef(-1)
  const temporalHandsRef = useRef<[HandTemporalState, HandTemporalState]>([
    createHandTemporalState(),
    createHandTemporalState(),
  ])
  const smoothingStrengthRef = useRef(65)
  const actionStateRef = useRef<GestureActionState>('idle')
  const previousPrimaryGestureRef = useRef<TrackedGesture | null>(null)
  const previousPalmPositionRef = useRef<{ x: number; y: number } | null>(null)
  const previousVZoomRelativeDistanceRef = useRef<number | null>(null)
  const vZoomStartDistanceRef = useRef<number | null>(null)
  const filteredVZoomDistanceRef = useRef<number | null>(null)
  const rotationSensitivityRef = useRef(5)
  const zoomSensitivityRef = useRef(4)
  const lastCursorUpdateRef = useRef(0)
  // Rotation stays disabled until the raw hand landmarks show a fully open palm for 250 ms.
  const openPalmSinceRef = useRef<number | null>(null)

  const [actionState, setActionState] = useState<GestureActionState>('idle')
  const [cameraState, setCameraState] = useState<'off' | 'starting' | 'on'>('off')
  const [cameraError, setCameraError] = useState('')
  const [trackerError, setTrackerError] = useState('')
  const [trackerState, setTrackerState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const [detectedHands, setDetectedHands] = useState(0)
  const [fps, setFps] = useState(0)
  const [gestureLabels, setGestureLabels] = useState<SmoothedGestureLabel[]>([])
  const [smoothingStrength, setSmoothingStrength] = useState(65)
  const [rotationSensitivity, setRotationSensitivity] = useState(5)
  const [zoomSensitivity, setZoomSensitivity] = useState(4)
  const [position, setPosition] = useState({ x: 20, y: 12 })
  const [panelSize, setPanelSize] = useState<'compact' | 'normal' | 'large'>('normal')
  const [controlsCollapsed, setControlsCollapsed] = useState(false)

  smoothingStrengthRef.current = smoothingStrength
  rotationSensitivityRef.current = rotationSensitivity
  zoomSensitivityRef.current = zoomSensitivity

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false

      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
      }

      landmarkerRef.current?.close()
      landmarkerRef.current = null

      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null

      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    }
  }, [])

  const resetTemporalHands = () => {
    temporalHandsRef.current = [createHandTemporalState(), createHandTemporalState()]
  }

  const stopCamera = () => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    landmarkerRef.current?.close()
    landmarkerRef.current = null

    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null

    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
    }

    const canvas = overlayRef.current
    const context = canvas?.getContext('2d')
    if (canvas && context) {
      context.setTransform(1, 0, 0, 1, 0, 0)
      context.clearRect(0, 0, canvas.width, canvas.height)
    }

    lastVideoTimeRef.current = -1
    lastHandsCountRef.current = -1
    resetTemporalHands()
    actionStateRef.current = 'idle'
    previousPrimaryGestureRef.current = null
    previousPalmPositionRef.current = null
    openPalmSinceRef.current = null
    previousVZoomRelativeDistanceRef.current = null
    vZoomStartDistanceRef.current = null
    filteredVZoomDistanceRef.current = null
    setActionState('idle')
    setCameraState('off')
    setTrackerState('idle')
    setTrackerError('')
    setCameraError('')
    setDetectedHands(0)
    setGestureLabels([])
    setFps(0)
    onCursorUpdate({ x: 0.5, y: 0.5, visible: false })
    onPinchStateChange(false)
  }

  const startCamera = async () => {
    if (cameraState !== 'off') return

    setCameraError('')
    setTrackerError('')
    setGestureLabels([])
    resetTemporalHands()
    actionStateRef.current = 'idle'
    previousPrimaryGestureRef.current = null
    previousPalmPositionRef.current = null
    openPalmSinceRef.current = null
    previousVZoomRelativeDistanceRef.current = null
    vZoomStartDistanceRef.current = null
    filteredVZoomDistanceRef.current = null
    setActionState('idle')
    setTrackerState('loading')
    setCameraState('starting')

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        'Camera access is unavailable. Open NEXUS in Chrome on localhost or a secure HTTPS page.',
      )
      setCameraState('off')
      setTrackerState('idle')
      return
    }

    let stream: MediaStream | null = null

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      })

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }

      streamRef.current = stream

      const video = videoRef.current
      if (!video) {
        stream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        setCameraState('off')
        setTrackerState('idle')
        return
      }

      video.srcObject = stream
      setCameraState('on')
      await video.play()

      const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })

      if (!mountedRef.current || streamRef.current !== stream) {
        landmarker.close()
        return
      }

      landmarkerRef.current = landmarker
      setTrackerState('ready')

      let fpsFrameCount = 0
      let fpsWindowStart = performance.now()
      let lastGestureUiUpdate = 0

      const connections: Array<[number, number]> = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17],
      ]

      const processFrame = () => {
        const currentVideo = videoRef.current
        const currentCanvas = overlayRef.current
        const currentLandmarker = landmarkerRef.current

        if (!currentVideo || !currentCanvas || !currentLandmarker) return

        if (
          currentVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          currentVideo.currentTime !== lastVideoTimeRef.current
        ) {
          const videoW = currentVideo.videoWidth
          const videoH = currentVideo.videoHeight
          const cssW = currentCanvas.clientWidth
          const cssH = currentCanvas.clientHeight

          if (videoW > 0 && videoH > 0 && cssW > 0 && cssH > 0) {
            const dpr = window.devicePixelRatio || 1
            const bmpW = Math.round(cssW * dpr)
            const bmpH = Math.round(cssH * dpr)

            if (currentCanvas.width !== bmpW || currentCanvas.height !== bmpH) {
              currentCanvas.width = bmpW
              currentCanvas.height = bmpH
            }

            const context = currentCanvas.getContext('2d')
            if (context) {
              context.setTransform(1, 0, 0, 1, 0, 0)
              context.clearRect(0, 0, bmpW, bmpH)
              context.scale(dpr, dpr)
              context.lineCap = 'round'
              context.lineJoin = 'round'

              // 1. Draw the mirrored video onto the canvas FIRST (object-cover).
              const videoAspect = videoW / videoH
              const cssAspect = cssW / cssH
              let drawX = 0, drawY = 0, drawW = cssW, drawH = cssH
              if (cssAspect > videoAspect) {
                drawW = cssW
                drawH = cssW / videoAspect
                drawY = (cssH - drawH) / 2
              } else {
                drawH = cssH
                drawW = cssH * videoAspect
                drawX = (cssW - drawW) / 2
              }

              context.save()
              context.translate(drawX + drawW, drawY)
              context.scale(-1, 1)
              context.drawImage(currentVideo, 0, 0, drawW, drawH)
              context.restore()

              // 2. Run MediaPipe detection directly on currentCanvas!
              // Because currentCanvas already has the exact cropped & mirrored image,
              // MediaPipe's normalized [0, 1] coordinates map 1:1 to [0, cssW] and [0, cssH].
              const timestamp = performance.now()
              const result = currentLandmarker.detectForVideo(currentCanvas, timestamp)
              const hands = result.landmarks ?? []
              const temporalHands = temporalHandsRef.current
              const classifications: SmoothedGestureLabel[] = []
              const alpha = 1 - smoothingStrengthRef.current / 100
              const stableFramesRequired = 5
              const releaseFramesRequired = 3
              const trackingLossGraceFrames = 8

              const toPx = (nx: number) => nx * cssW
              const toPy = (ny: number) => ny * cssH

              for (let handIndex = 0; handIndex < 2; handIndex += 1) {
                const temporal = temporalHands[handIndex]
                const rawLandmarks = hands[handIndex]

                if (rawLandmarks) {
                  temporal.missingFrames = 0
                  const smoothed = smoothHandLandmarks(
                    temporal.landmarks,
                    rawLandmarks,
                    alpha,
                  )
                  temporal.landmarks = smoothed

                  const classification: { gesture: TrackedGesture; confidence: number } =
                    isVShapeZoomGesture(smoothed)
                      ? { gesture: 'V-Zoom', confidence: 0.95 }
                      : classifyGesture(smoothed)

                  if (classification.gesture !== 'Unknown') {
                    temporal.releaseFrames = 0

                    if (classification.gesture === temporal.activeGesture) {
                      temporal.candidateGesture = null
                      temporal.candidateFrames = 0
                      temporal.activeConfidence = classification.confidence
                    } else {
                      if (classification.gesture === temporal.candidateGesture) {
                        temporal.candidateFrames += 1
                      } else {
                        temporal.candidateGesture = classification.gesture
                        temporal.candidateFrames = 1
                      }

                      if (temporal.candidateFrames >= stableFramesRequired) {
                        temporal.activeGesture = classification.gesture
                        temporal.activeConfidence = classification.confidence
                        temporal.candidateGesture = null
                        temporal.candidateFrames = 0
                      }
                    }
                  } else {
                    temporal.candidateGesture = null
                    temporal.candidateFrames = 0
                    if (temporal.activeGesture) {
                      temporal.releaseFrames += 1
                      if (temporal.releaseFrames >= releaseFramesRequired) {
                        temporal.activeGesture = null
                        temporal.activeConfidence = 0
                        temporal.releaseFrames = 0
                      }
                    }
                  }

                  const displayLandmarks = rawLandmarks
                  context.strokeStyle = handIndex === 0 ? '#22d3ee' : '#a78bfa'
                  context.lineWidth = Math.max(2, cssW / 240)

                  connections.forEach(([from, to]) => {
                    const startPoint = displayLandmarks[from]
                    const endPoint = displayLandmarks[to]
                    if (!startPoint || !endPoint) return
                    context.beginPath()
                    context.moveTo(toPx(startPoint.x), toPy(startPoint.y))
                    context.lineTo(toPx(endPoint.x), toPy(endPoint.y))
                    context.stroke()
                  })

                  displayLandmarks.forEach((point, pointIndex) => {
                    const radius = pointIndex === 0
                      ? Math.max(4, cssW / 150)
                      : Math.max(2.5, cssW / 220)
                    context.beginPath()
                    context.arc(toPx(point.x), toPy(point.y), radius, 0, Math.PI * 2)
                    context.fillStyle = pointIndex === 0
                      ? '#ffffff'
                      : handIndex === 0 ? '#22d3ee' : '#c4b5fd'
                    context.fill()
                  })

                  // Diagnostic raw-bounds box.
                  const xs = displayLandmarks.map((p) => toPx(p.x))
                  const ys = displayLandmarks.map((p) => toPy(p.y))
                  const minX = Math.min(...xs)
                  const maxX = Math.max(...xs)
                  const minY = Math.min(...ys)
                  const maxY = Math.max(...ys)
                  const diagnosticColor = handIndex === 0 ? '#facc15' : '#fb923c'
                  context.save()
                  context.strokeStyle = diagnosticColor
                  context.lineWidth = Math.max(2, cssW / 320)
                  context.setLineDash([8, 5])
                  context.strokeRect(minX, minY, maxX - minX, maxY - minY)
                  context.setLineDash([])
                  context.font = `bold ${Math.max(12, cssW / 32)}px sans-serif`
                  context.textBaseline = 'bottom'
                  const diagnosticText = `H${handIndex + 1} RAW BOUNDS`
                  const labelY = Math.max(18, minY - 4)
                  const labelWidth = context.measureText(diagnosticText).width + 12
                  context.fillStyle = 'rgba(0,0,0,0.78)'
                  context.fillRect(minX, labelY - Math.max(18, cssW / 24), labelWidth, Math.max(20, cssW / 21))
                  context.fillStyle = diagnosticColor
                  context.fillText(diagnosticText, minX + 6, labelY - 3)
                  context.restore()

                  classifications.push({
                    handIndex,
                    gesture: temporal.activeGesture ?? 'STABILIZING',
                    confidence: temporal.activeGesture ? temporal.activeConfidence : 0,
                    isHeld: false,
                  })
                } else {
                  temporal.missingFrames += 1
                  temporal.candidateGesture = null
                  temporal.candidateFrames = 0

                  if (temporal.missingFrames > trackingLossGraceFrames) {
                    temporalHands[handIndex] = createHandTemporalState()
                  } else if (temporal.activeGesture) {
                    classifications.push({
                      handIndex,
                      gesture: temporal.activeGesture,
                      confidence: temporal.activeConfidence,
                      isHeld: true,
                    })
                  }
                }
              }

              // Cursor in [0,1] space — since detection runs on the mirrored canvas,
              // primaryIndexTip.x and .y are already in the exact [0, 1] viewport space!
              const primaryGesture = temporalHands[0].activeGesture
              const primaryRawGesture = hands[0] ? classifyGesture(hands[0]).gesture : 'Unknown'
              const primaryHandVisible = Boolean(hands[0])
              const primaryIndexTip = temporalHands[0].landmarks?.[8]
              if (primaryHandVisible && primaryIndexTip) {
                if (timestamp - lastCursorUpdateRef.current >= 30) {
                  onCursorUpdate({
                    x: THREE.MathUtils.clamp(primaryIndexTip.x, 0, 1),
                    y: THREE.MathUtils.clamp(primaryIndexTip.y, 0, 1),
                    visible: true,
                  })
                  lastCursorUpdateRef.current = timestamp
                }
              } else {
                onCursorUpdate({ x: 0.5, y: 0.5, visible: false })
              }

              // Gesture actions
              const primaryHandVisibleForActions = Boolean(hands[0])

              if (!primaryHandVisibleForActions) {
                // Drop a grabbed component immediately when the primary hand disappears.
                if (previousPrimaryGestureRef.current === 'Pinch') {
                  onPinchStateChange(false)
                }
                previousPalmPositionRef.current = null
                openPalmSinceRef.current = null
                previousVZoomRelativeDistanceRef.current = null
                vZoomStartDistanceRef.current = null
                filteredVZoomDistanceRef.current = null
                if (primaryGesture !== 'Closed Fist') previousPrimaryGestureRef.current = null
                if (actionStateRef.current !== 'locked' && actionStateRef.current !== 'idle') {
                  actionStateRef.current = 'idle'
                  setActionState('idle')
                }
              } else {
                if (primaryGesture !== previousPrimaryGestureRef.current) {
                  onPinchStateChange(
                    primaryGesture === 'Pinch' && actionStateRef.current !== 'locked',
                  )
                  if (primaryGesture === 'Closed Fist') {
                    const nextState: GestureActionState =
                      actionStateRef.current === 'locked' ? 'idle' : 'locked'
                    actionStateRef.current = nextState
                    setActionState(nextState)
                    previousPalmPositionRef.current = null
                    previousVZoomRelativeDistanceRef.current = null
                    vZoomStartDistanceRef.current = null
                    filteredVZoomDistanceRef.current = null
                  } else if (actionStateRef.current !== 'locked') {
                    let nextState: GestureActionState = 'idle'
                    if (primaryGesture === 'Open Palm') nextState = 'rotating'
                    else if (primaryGesture === 'Pinch') nextState = 'grabbing'
                    else if (primaryGesture === 'V-Zoom') nextState = 'zooming'
                    actionStateRef.current = nextState
                    setActionState(nextState)
                    previousPalmPositionRef.current = null
                    if (primaryGesture !== 'V-Zoom') {
                      previousVZoomRelativeDistanceRef.current = null
                      vZoomStartDistanceRef.current = null
                      filteredVZoomDistanceRef.current = null
                    }
                  }
                  previousPrimaryGestureRef.current = primaryGesture
                }

                if (actionStateRef.current === 'locked') {
                  previousPalmPositionRef.current = null
                  openPalmSinceRef.current = null
                  previousVZoomRelativeDistanceRef.current = null
                  vZoomStartDistanceRef.current = null
                  filteredVZoomDistanceRef.current = null
                } else {
                  const landmarks = temporalHands[0].landmarks
                  const wrist = landmarks?.[0]
                  const middleMcp = landmarks?.[9]
                  const indexTip = landmarks?.[8]
                  const middleTip = landmarks?.[12]

                  if (primaryGesture === 'V-Zoom' && wrist && middleMcp && indexTip && middleTip) {
                    const handScale = Math.hypot(
                      wrist.x - middleMcp.x,
                      wrist.y - middleMcp.y,
                      wrist.z - middleMcp.z,
                    )
                    const normalizedTipDistance = Math.hypot(
                      indexTip.x - middleTip.x,
                      indexTip.y - middleTip.y,
                      indexTip.z - middleTip.z,
                    ) / Math.max(handScale, 0.0001)

                    const zoomAlpha = 0.32
                    const filteredDistance = filteredVZoomDistanceRef.current === null
                      ? normalizedTipDistance
                      : filteredVZoomDistanceRef.current +
                        zoomAlpha * (normalizedTipDistance - filteredVZoomDistanceRef.current)
                    filteredVZoomDistanceRef.current = filteredDistance

                    if (vZoomStartDistanceRef.current === null) {
                      vZoomStartDistanceRef.current = filteredDistance
                      previousVZoomRelativeDistanceRef.current = 0
                    } else {
                      const relativeDistance = filteredDistance - vZoomStartDistanceRef.current
                      const previousRelative = previousVZoomRelativeDistanceRef.current ?? relativeDistance
                      const delta = relativeDistance - previousRelative

                      if (Math.abs(delta) > 0.00035) {
                        onZoomDelta(delta * zoomSensitivityRef.current * 2.2)
                      }
                      previousVZoomRelativeDistanceRef.current = relativeDistance
                    }

                    previousPalmPositionRef.current = null
                  } else {
                    previousVZoomRelativeDistanceRef.current = null
                    vZoomStartDistanceRef.current = null
                    filteredVZoomDistanceRef.current = null

                    // Use raw landmarks for the safety gate so smoothing and gesture
                    // stabilization cannot keep rotation running while a finger curls.
                    const canRotate = primaryGesture === 'Open Palm' && primaryRawGesture === 'Open Palm'
                    if (canRotate && landmarks) {
                      if (openPalmSinceRef.current === null) {
                        openPalmSinceRef.current = timestamp
                      }

                      const fullyOpenDuration = timestamp - openPalmSinceRef.current
                      if (fullyOpenDuration >= 250) {
                        const palmPoints = [0, 5, 9, 13, 17].map((index) => landmarks[index])
                        if (palmPoints.every((point) => point !== undefined)) {
                          const palmPosition = {
                            x: palmPoints.reduce((sum, point) => sum + point.x, 0) / palmPoints.length,
                            y: palmPoints.reduce((sum, point) => sum + point.y, 0) / palmPoints.length,
                          }
                          const previousPosition = previousPalmPositionRef.current
                          if (previousPosition) {
                            // MediaPipe coordinates are normalized to [0, 1]. The
                            // smaller scale prevents a short hand movement from
                            // spinning the model through a large angle.
                            const sensitivity = rotationSensitivityRef.current * 0.63
                            const rawDeltaX = (palmPosition.x - previousPosition.x) * sensitivity
                            const rawDeltaY = (palmPosition.y - previousPosition.y) * sensitivity
                            // Ignore tiny landmark jitter and cap single-frame jumps.
                            const deltaX = Math.abs(rawDeltaX) < 0.0015
                              ? 0
                              : THREE.MathUtils.clamp(rawDeltaX, -0.08, 0.08)
                            const deltaY = Math.abs(rawDeltaY) < 0.0015
                              ? 0
                              : THREE.MathUtils.clamp(rawDeltaY, -0.08, 0.08)
                            if (deltaX !== 0 || deltaY !== 0) {
                              onRotateDelta(deltaX, deltaY)
                            }
                          }
                          previousPalmPositionRef.current = palmPosition
                        }
                      } else {
                        // Do not accumulate movement while waiting for the open-palm gate.
                        previousPalmPositionRef.current = null
                      }
                    } else {
                      openPalmSinceRef.current = null
                      previousPalmPositionRef.current = null
                    }
                  }
                }
              }

              if (hands.length !== lastHandsCountRef.current) {
                lastHandsCountRef.current = hands.length
                setDetectedHands(hands.length)
              }

              if (timestamp - lastGestureUiUpdate >= 100) {
                setGestureLabels(classifications)
                lastGestureUiUpdate = timestamp
              }

              fpsFrameCount += 1
              const elapsed = timestamp - fpsWindowStart
              if (elapsed >= 1000) {
                setFps(Math.round((fpsFrameCount * 1000) / elapsed))
                fpsFrameCount = 0
                fpsWindowStart = timestamp
              }
            }
          }

          lastVideoTimeRef.current = currentVideo.currentTime
        }

        animationFrameRef.current = requestAnimationFrame(processFrame)
      }

      animationFrameRef.current = requestAnimationFrame(processFrame)
    } catch (error) {
      if (!mountedRef.current || (stream && streamRef.current !== stream)) {
        stream?.getTracks().forEach((track) => track.stop())
        return
      }

      const name = error instanceof DOMException ? error.name : ''

      if (!stream) {
        if (name === 'NotAllowedError' || name === 'PermissionDeniedError' || name === 'SecurityError') {
          setCameraError(
            'Camera permission was denied. Allow camera access in your browser site settings, then try again.',
          )
        } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
          setCameraError('No camera was found. Connect or enable a webcam, then try again.')
        } else if (name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError') {
          setCameraError(
            'The camera is unavailable or already in use by another app or browser tab. Close other apps using it, then try again.',
          )
        } else if (name === 'OverconstrainedError') {
          setCameraError('The camera does not support the requested settings. Try another camera.')
        } else {
          setCameraError('Could not start the camera. Check your camera connection and browser permissions.')
        }

        stream?.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        setCameraState('off')
        setTrackerState('idle')
      } else {
        setTrackerError(
          'Camera is running, but hand tracking could not load. Check that the local MediaPipe WASM files and hand_landmarker.task model are in the required folders, then restart the camera.',
        )
        setTrackerState('error')
      }
    }
  }

  const handleDragStart = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const handleDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.buttons !== 1) return
    setPosition((current) => ({
      x: Math.max(0, current.x + event.movementX),
      y: Math.max(0, current.y + event.movementY),
    }))
  }

  return (
    <div
      className="absolute z-30 flex flex-col overflow-hidden rounded-xl border border-cyan-400/20 bg-[#080d14]/95 shadow-2xl shadow-black/40 backdrop-blur-xl"
      style={{
        left: position.x,
        top: position.y,
        width: panelSize === 'compact' ? 'min(280px, calc(100% - 24px))' : panelSize === 'large' ? 'min(460px, calc(100% - 24px))' : 'min(360px, calc(100% - 24px))',
        height: panelSize === 'compact' ? 390 : panelSize === 'large' ? 700 : 560,
        maxHeight: 'calc(100% - 16px)',
      }}
    >
      <div
        onPointerDown={handleDragStart}
        onPointerMove={handleDrag}
        className="flex shrink-0 cursor-grab touch-none items-center justify-between border-b border-white/[0.07] px-3 py-2 active:cursor-grabbing"
      >
        <div className="flex items-center gap-2 text-[10px] font-semibold tracking-[0.15em] text-slate-300">
          <GripVertical size={14} className="text-slate-500" />
          <Camera size={14} className="text-cyan-400" />
          HAND TRACKING
          <span className={`ml-1 h-2 w-2 rounded-full ${cameraState === 'on' && trackerState === 'ready' ? 'bg-emerald-400' : cameraState === 'starting' || trackerState === 'loading' ? 'animate-pulse bg-amber-400' : 'bg-slate-600'}`} />
        </div>
        <div className="flex items-center gap-1" onPointerDown={(event) => event.stopPropagation()}>
          <button
            type="button"
            title="Make camera panel smaller"
            onClick={() => setPanelSize((current) => current === 'large' ? 'normal' : 'compact')}
            className="rounded p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
          ><Minus size={13} /></button>
          <button
            type="button"
            title="Make camera panel larger"
            onClick={() => setPanelSize((current) => current === 'compact' ? 'normal' : 'large')}
            className="rounded p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
          ><Plus size={13} /></button>
          <button
            type="button"
            title={cameraState === 'on' ? 'Stop camera' : 'Start camera'}
            disabled={cameraState === 'starting'}
            onClick={() => cameraState === 'on' ? stopCamera() : startCamera()}
            className={`rounded-md px-2 py-1.5 text-[10px] font-semibold disabled:opacity-50 ${cameraState === 'on' ? 'bg-red-400/10 text-red-300 hover:bg-red-400/20' : 'bg-cyan-400 text-[#031019] hover:bg-cyan-300'}`}
          >{cameraState === 'starting' ? 'STARTING' : cameraState === 'on' ? 'STOP' : 'START CAMERA'}</button>
          <button
            type="button"
            title={controlsCollapsed ? 'Show camera controls' : 'Hide camera controls'}
            onClick={() => setControlsCollapsed((value) => !value)}
            className="rounded p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
          >{controlsCollapsed ? <ChevronDown size={13} /> : <ChevronRight size={13} />}</button>
        </div>
      </div>

      <div className="relative aspect-video w-full shrink-0 overflow-hidden bg-[#03060a]">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="pointer-events-none absolute h-px w-px opacity-0"
        />
        <canvas
          ref={overlayRef}
          className={`pointer-events-none absolute inset-0 h-full w-full ${cameraState === 'on' ? 'block' : 'hidden'}`}
        />

        {cameraState === 'on' && trackerState === 'ready' && (
          <div className="pointer-events-none absolute left-2 right-2 top-2 z-10 flex flex-col items-start gap-1">
            {gestureLabels.length > 0 ? (
              gestureLabels.map((label) => (
                <div
                  key={label.handIndex}
                  className="rounded-md border border-white/10 bg-black/75 px-2 py-1 text-[10px] font-semibold text-white shadow-lg backdrop-blur"
                >
                  HAND {label.handIndex + 1}: {label.gesture.toUpperCase()}
                  {label.confidence > 0 ? ` · ${Math.round(label.confidence * 100)}%` : ''}
                  {label.isHeld ? ' · HOLD' : ''}
                </div>
              ))
            ) : (
              <div className="rounded-md border border-white/10 bg-black/65 px-2 py-1 text-[10px] text-slate-300 shadow-lg backdrop-blur">
                NO GESTURE DETECTED
              </div>
            )}
          </div>
        )}

        {cameraState !== 'on' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-500">
            {cameraState === 'starting' ? (
              <>
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-cyan-400/20 border-t-cyan-400" />
                <span className="text-[11px]">Starting camera...</span>
              </>
            ) : (
              <>
                <VideoOff size={24} />
                <span className="text-[11px]">Camera is off</span>
              </>
            )}
          </div>
        )}

        {cameraState === 'on' && trackerState === 'loading' && (
          <div className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-1 text-[9px] text-amber-300">
            Loading hand model...
          </div>
        )}
      </div>

      {!controlsCollapsed && <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
        {cameraError && (
          <p role="alert" className="rounded-lg border border-red-400/20 bg-red-400/[0.06] p-2 text-[10px] leading-4 text-red-300">
            {cameraError}
          </p>
        )}
        {trackerError && (
          <p role="alert" className="rounded-lg border border-amber-400/20 bg-amber-400/[0.06] p-2 text-[10px] leading-4 text-amber-200">
            {trackerError}
          </p>
        )}

        <div className="rounded-lg border border-white/[0.07] bg-white/[0.03] p-2">
          <div className="mb-1 flex items-center justify-between text-[10px]">
            <label htmlFor="landmark-smoothing" className="text-slate-500">LANDMARK SMOOTHING</label>
            <span className="font-semibold text-cyan-300">{smoothingStrength}%</span>
          </div>
          <input
            id="landmark-smoothing"
            type="range"
            min="0"
            max="90"
            step="5"
            value={smoothingStrength}
            onChange={(event) => setSmoothingStrength(Number(event.target.value))}
            className="w-full accent-cyan-400"
          />
          <p className="mt-1 text-[9px] leading-4 text-slate-600">Higher values reduce landmark jitter but add more movement lag.</p>
        </div>

        <div className="rounded-lg border border-white/[0.07] bg-white/[0.03] p-2">
          <div className="mb-1 flex items-center justify-between text-[10px]">
            <label htmlFor="rotation-sensitivity" className="text-slate-500">ROTATION SENSITIVITY</label>
            <span className="font-semibold text-cyan-300">{rotationSensitivity.toFixed(1)}×</span>
          </div>
          <input
            id="rotation-sensitivity"
            type="range"
            min="1"
            max="10"
            step="0.5"
            value={rotationSensitivity}
            onChange={(event) => setRotationSensitivity(Number(event.target.value))}
            className="w-full accent-cyan-400"
          />
          <p className="mt-1 text-[9px] leading-4 text-slate-600">Move your palm left/right and up/down to rotate the model.</p>
        </div>

        <div className="rounded-lg border border-white/[0.07] bg-white/[0.03] p-2">
          <div className="mb-1 flex items-center justify-between text-[10px]">
            <label htmlFor="zoom-sensitivity" className="text-slate-500">ZOOM SENSITIVITY</label>
            <span className="font-semibold text-cyan-300">{zoomSensitivity.toFixed(1)}×</span>
          </div>
          <input
            id="zoom-sensitivity"
            type="range"
            min="1"
            max="10"
            step="0.5"
            value={zoomSensitivity}
            onChange={(event) => setZoomSensitivity(Number(event.target.value))}
            className="w-full accent-cyan-400"
          />
          <p className="mt-1 text-[9px] leading-4 text-slate-600">Spread or close your index and middle fingers to zoom.</p>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div className="rounded-lg border border-white/[0.07] bg-white/[0.03] p-2">
            <p className="text-slate-500">DETECTION</p>
            <p className={`mt-1 font-semibold ${detectedHands > 0 ? 'text-emerald-300' : 'text-slate-300'}`}>
              {trackerState === 'ready' ? detectedHands > 0 ? `HAND DETECTED (${detectedHands})` : 'NOT DETECTED' : trackerState === 'loading' ? 'LOADING' : 'INACTIVE'}
            </p>
          </div>
          <div className="rounded-lg border border-white/[0.07] bg-white/[0.03] p-2">
            <p className="text-slate-500">INFERENCE FPS</p>
            <p className="mt-1 font-semibold text-cyan-300">{trackerState === 'ready' ? fps : '—'}</p>
          </div>
        </div>

        <div className={`rounded-lg border px-3 py-2 ${actionState === 'locked' ? 'border-amber-400/30 bg-amber-400/[0.08]' : 'border-cyan-400/20 bg-cyan-400/[0.04]'}`}>
          <div className="flex items-center justify-between">
            <p className="text-[9px] tracking-[0.16em] text-slate-500">GESTURE STATE</p>
            <span className={`h-2 w-2 rounded-full ${actionState === 'locked' ? 'bg-amber-400' : actionState === 'idle' ? 'bg-slate-500' : 'bg-cyan-400'}`} />
          </div>
          <p className={`mt-1 text-sm font-semibold uppercase tracking-[0.12em] ${actionState === 'locked' ? 'text-amber-300' : actionState === 'idle' ? 'text-slate-300' : 'text-cyan-300'}`}>
            {actionState}
          </p>
          <p className="mt-1 text-[9px] leading-4 text-slate-500">
            {actionState === 'locked' ? 'Actions paused · make a fist again to unlock' : 'Open palm: rotate · V-shape: zoom · Pinch and hold: drag a component'}
          </p>
        </div>

        <div className="rounded-lg border border-white/[0.07] bg-white/[0.03] px-2 py-2 text-[9px] leading-4 text-slate-500">
          Zoom: extend index + middle fingers; curl ring + pinky. Spread to zoom in, close to zoom out. Thumb is ignored.
        </div>

      </div>}
    </div>
  )
}

/* -----------------------------
   WORKSPACE
----------------------------- */

function Workspace({ onBack }: { onBack: () => void }) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [hierarchyOpen, setHierarchyOpen] = useState(true)

  const [viewMode, setViewMode] = useState<ViewMode>('Explore')

  const [modelScene, setModelScene] = useState<THREE.Object3D | null>(null)
  const [modelName, setModelName] = useState('')
  const [components, setComponents] = useState<ComponentInfo[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [cameraReset, setCameraReset] = useState(0)
  // High-frequency hand transforms live on refs/Three.js objects, not React state.
  // This prevents the entire workspace from re-rendering for every camera frame.
  const modelGroupRef = useRef<THREE.Group>(null)
  const handCursorRef = useRef<HandCursorPosition>({ x: 0.5, y: 0.5, visible: false })
  const handCursorElementRef = useRef<HTMLDivElement>(null)
  const [pinchActive, setPinchActive] = useState(false)

  const selectedComponent = components.find(
    (component) => component.id === selectedId,
  )

  /* -----------------------------
     LOAD GLB MODEL
  ----------------------------- */

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]

    if (!file) return

    setError('')
    setSelectedId(null)

    if (!file.name.toLowerCase().endsWith('.glb')) {
      setError('Please upload a valid .glb file.')
      event.target.value = ''
      return
    }

    if (file.size > 100 * 1024 * 1024) {
      setError('File is too large. Maximum supported size is 100 MB.')
      event.target.value = ''
      return
    }

    setLoading(true)

    try {
      const buffer = await file.arrayBuffer()
      const loader = new GLTFLoader()

      loader.parse(
        buffer,
        '',
        (gltf) => {
          try {
            const scene = gltf.scene.clone(true)

            const foundComponents: ComponentInfo[] = []
            let meshCount = 0

            scene.traverse((object) => {
              if (!(object instanceof THREE.Mesh)) return

              meshCount++

              const mesh = object

              // Clone materials so highlighting does not
              // modify shared materials in other meshes.
              if (Array.isArray(mesh.material)) {
                mesh.material = mesh.material.map((material) => material.clone())
              } else {
                mesh.material = mesh.material.clone()
              }

              const materials = Array.isArray(mesh.material)
                ? mesh.material
                : [mesh.material]

              materials.forEach((material) => {
                if (material instanceof THREE.MeshStandardMaterial) {
                  material.userData.nexusOriginalEmissive =
                    material.emissive.clone()

                  material.userData.nexusOriginalEmissiveIntensity =
                    material.emissiveIntensity
                }
              })

              const componentName =
                mesh.name.trim() || `Component ${String(meshCount).padStart(2, '0')}`

              foundComponents.push({
                id: mesh.uuid,
                name: componentName,
                geometry: mesh.geometry.type,
                vertices: mesh.geometry.attributes.position?.count ?? 0,
              })
            })

            if (meshCount === 0) {
              setError('This GLB file contains no renderable mesh components.')
              setLoading(false)
              return
            }

            setModelScene(scene)
            setModelName(file.name)
            setComponents(foundComponents)
            setSelectedId(null)
            setCameraReset((value) => value + 1)
            setError('')
          } catch {
            setError('Could not process this model. Please try another GLB file.')
          }

          setLoading(false)
        },
        () => {
          setError(
            'Could not load this GLB file. It may be corrupted or unsupported.',
          )
          setLoading(false)
        },
      )
    } catch {
      setError('Could not read the selected file.')
      setLoading(false)
    }

    event.target.value = ''
  }

  /* -----------------------------
     SELECTED COMPONENT HIGHLIGHT
  ----------------------------- */

  useEffect(() => {
    if (!modelScene) return

    modelScene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return

      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material]

      materials.forEach((material) => {
        if (!(material instanceof THREE.MeshStandardMaterial)) return

        const originalColor = material.userData.nexusOriginalEmissive

        if (originalColor) {
          material.emissive.copy(originalColor)
        }

        material.emissiveIntensity =
          material.userData.nexusOriginalEmissiveIntensity ?? 0

        if (object.uuid === selectedId) {
          material.emissive.set('#00cfff')
          material.emissiveIntensity = 0.8
        }
      })
    })
  }, [modelScene, selectedId])

  const resetModel = () => {
    setSelectedId(null)
    if (modelGroupRef.current) {
      modelGroupRef.current.rotation.set(0, 0, 0)
      modelGroupRef.current.scale.setScalar(1)
    }
    handCursorRef.current = { x: 0.5, y: 0.5, visible: false }
    if (handCursorElementRef.current) {
      handCursorElementRef.current.style.display = 'none'
    }
    setCameraReset((value) => value + 1)
  }

  // Apply high-frequency transforms directly to Three.js. No React render per frame.
  const handleGestureRotation = (deltaX: number, deltaY: number) => {
    const group = modelGroupRef.current
    if (!group || !Number.isFinite(deltaX) || !Number.isFinite(deltaY)) return

    // Trackball-style rotation: each hand movement produces one 3D rotation
    // around an axis perpendicular to the movement. Diagonal movement therefore
    // creates a combined 3D turn instead of two independent Euler updates.
    const movement = Math.hypot(deltaX, deltaY)
    if (movement < 0.0001) return

    const rotationAxis = new THREE.Vector3(deltaY, deltaX, 0).normalize()
    const rotationStep = new THREE.Quaternion().setFromAxisAngle(rotationAxis, movement)
    group.quaternion.premultiply(rotationStep).normalize()
  }

  const handleGestureZoom = (delta: number) => {
    const group = modelGroupRef.current
    if (!group || !Number.isFinite(delta)) return
    const nextScale = THREE.MathUtils.clamp(group.scale.x + delta, 0.55, 2.5)
    group.scale.setScalar(nextScale)
  }

  const handleHandCursorUpdate = (cursor: HandCursorPosition) => {
    handCursorRef.current = cursor
    const element = handCursorElementRef.current
    if (!element) return
    if (!cursor.visible) {
      element.style.display = 'none'
      return
    }
    element.style.display = 'block'
    element.style.left = `${cursor.x * 100}%`
    element.style.top = `${cursor.y * 100}%`
  }

  const handlePinchStateChange = (active: boolean) => {
    setPinchActive(active)
  }

  const clearModel = () => {
    setModelScene(null)
    setModelName('')
    setComponents([])
    setSelectedId(null)
    handCursorRef.current = { x: 0.5, y: 0.5, visible: false }
    if (handCursorElementRef.current) {
      handCursorElementRef.current.style.display = 'none'
    }
    setError('')
  }

  return (
    <main className="flex h-screen min-h-[550px] flex-col overflow-hidden bg-[#05080d] text-slate-200">

      {/* TOP TOOLBAR */}

      <header className="z-20 flex h-[68px] shrink-0 items-center justify-between border-b border-white/[0.07] bg-[#080d14] px-4 lg:px-6">

        <div className="flex min-w-0 items-center gap-3">
          <button
            onClick={onBack}
            title="Return to landing page"
            className="rounded-lg p-2 text-slate-400 transition hover:bg-white/[0.06] hover:text-white"
          >
            <ArrowLeft size={19} />
          </button>

          <div className="flex items-center gap-2">
            <Atom size={24} className="shrink-0 text-cyan-400" />
            <span className="text-lg font-bold tracking-[0.2em]">
              NEXUS<span className="text-cyan-400">.</span>
            </span>
          </div>

          <div className="mx-2 hidden h-6 w-px bg-white/10 sm:block" />

          <span className="hidden max-w-[180px] truncate text-xs text-slate-500 sm:block">
            {modelName || 'Workspace / Untitled'}
          </span>
        </div>

        <div className="flex items-center gap-2">

          <input
            ref={fileInputRef}
            type="file"
            accept=".glb,model/gltf-binary"
            className="hidden"
            onChange={handleFile}
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-cyan-400 px-3 py-2.5 text-xs font-semibold text-[#031019] transition hover:bg-cyan-300 disabled:cursor-wait disabled:opacity-50 sm:px-4"
          >
            <Upload size={15} />
            <span className="hidden sm:inline">
              {loading ? 'Loading...' : 'Upload GLB'}
            </span>
            <span className="sm:hidden">Upload</span>
          </button>

          {modelScene && (
            <button
              onClick={clearModel}
              title="Remove current model"
              className="rounded-lg border border-white/10 p-2.5 text-slate-400 transition hover:border-red-400/30 hover:text-red-400"
            >
              <X size={16} />
            </button>
          )}

          <button
            onClick={onBack}
            title="Close workspace"
            className="rounded-lg p-2 text-slate-400 transition hover:bg-white/[0.06] hover:text-white"
          >
            <X size={18} />
          </button>
        </div>
      </header>

      {/* MAIN WORKSPACE */}

      <div className="flex min-h-0 flex-1">

        {/* LEFT SIDEBAR */}

        <AnimatePresence initial={false}>
          {leftOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 260, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="hidden shrink-0 overflow-hidden border-r border-white/[0.07] bg-[#080d14] md:block"
            >
              <div className="flex h-full w-[260px] flex-col">

                <div className="flex h-14 items-center justify-between border-b border-white/[0.06] px-5">
                  <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.15em] text-slate-300">
                    <Layers3 size={16} className="text-cyan-400" />
                    HIERARCHY
                  </div>

                  <button
                    onClick={() => setLeftOpen(false)}
                    title="Collapse hierarchy"
                    className="text-slate-500 hover:text-white"
                  >
                    <PanelLeftClose size={17} />
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto p-4">

                  <button
                    onClick={() => setHierarchyOpen(!hierarchyOpen)}
                    className="flex w-full items-center gap-2 rounded-lg p-2 text-sm text-slate-300 hover:bg-white/[0.04]"
                  >
                    {hierarchyOpen ? (
                      <ChevronDown size={15} />
                    ) : (
                      <ChevronRight size={15} />
                    )}

                    <FileBox size={16} className="text-cyan-400" />

                    <span className="truncate">
                      {modelName || 'Untitled Project'}
                    </span>
                  </button>

                  {hierarchyOpen && (
                    <div className="ml-3 mt-3 space-y-1 border-l border-white/[0.08] pl-3">

                      {components.map((component, index) => (
                        <button
                          key={component.id}
                          onClick={() => setSelectedId(component.id)}
                          className={`flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-xs transition ${
                            selectedId === component.id
                              ? 'bg-cyan-400/10 text-cyan-300'
                              : 'text-slate-500 hover:bg-white/[0.04] hover:text-slate-200'
                          }`}
                        >
                          <Box size={14} className="shrink-0" />
                          <span className="truncate">
                            {component.name}
                          </span>
                          <span className="ml-auto text-[9px] text-slate-700">
                            {index + 1}
                          </span>
                        </button>
                      ))}

                      {components.length === 0 && (
                        <p className="px-2 py-3 text-xs text-slate-600">
                          No components
                        </p>
                      )}
                    </div>
                  )}

                  {!modelScene && (
                    <div className="mt-8 rounded-xl border border-dashed border-white/10 p-4 text-center">
                      <Upload size={20} className="mx-auto text-slate-600" />
                      <p className="mt-3 text-xs font-medium text-slate-400">
                        No model imported
                      </p>
                      <p className="mt-2 text-[11px] leading-5 text-slate-600">
                        Upload a GLB model to see its hierarchy.
                      </p>
                    </div>
                  )}

                </div>

                <div className="border-t border-white/[0.06] p-4">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Layers3 size={15} />
                    {components.length} mesh components
                  </div>
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* CENTER VIEWPORT */}

        <section className="relative flex min-w-0 flex-1 flex-col bg-[#05090f]">

          {/* VIEWPORT TOOLBAR */}

          <div className="flex h-14 shrink-0 items-center justify-between border-b border-white/[0.06] bg-[#080d14]/70 px-3 sm:px-5">

            <div className="flex items-center gap-2">

              {!leftOpen && (
                <button
                  onClick={() => setLeftOpen(true)}
                  title="Open hierarchy"
                  className="mr-1 rounded-lg p-2 text-slate-400 hover:bg-white/[0.06] hover:text-cyan-300"
                >
                  <PanelLeftOpen size={17} />
                </button>
              )}

              <div className="flex items-center gap-1 rounded-lg border border-white/[0.07] bg-white/[0.02] p-1">
                {(['Explore', 'Inspect', 'Disassemble'] as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`rounded-md px-3 py-1.5 text-[11px] transition ${
                      viewMode === mode
                        ? 'bg-cyan-400/10 text-cyan-300'
                        : 'text-slate-500 hover:text-slate-200'
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2">

              {modelScene && (
                <button
                  onClick={resetModel}
                  title="Reset camera and selection"
                  className="rounded-lg p-2 text-slate-400 transition hover:bg-white/[0.06] hover:text-cyan-300"
                >
                  <RotateCcw size={16} />
                </button>
              )}

              {!rightOpen && (
                <button
                  onClick={() => setRightOpen(true)}
                  title="Open inspector"
                  className="rounded-lg p-2 text-slate-400 hover:bg-white/[0.06] hover:text-cyan-300"
                >
                  <PanelRightOpen size={17} />
                </button>
              )}
            </div>
          </div>

          {/* VIEWPORT */}

          <div className="grid-background relative min-h-0 flex-1 overflow-hidden">

            <div className="absolute left-5 top-5 z-10 flex items-center gap-2 text-[10px] tracking-[0.2em] text-slate-600">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
              3D VIEWPORT
            </div>

            <CameraPreview
              onRotateDelta={handleGestureRotation}
              onZoomDelta={handleGestureZoom}
              onCursorUpdate={handleHandCursorUpdate}
              onPinchStateChange={handlePinchStateChange}
            />

            <div
                ref={handCursorElementRef}
                className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-1/2"
                style={{ left: '50%', top: '50%', display: 'none' }}
              >
                <div className="relative h-5 w-5">
                  <span className="absolute inset-0 rounded-full border-2 border-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.9)]" />
                  <span className="absolute left-1/2 top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white" />
                </div>
            </div>

            {modelScene ? (
              <Canvas
                key={cameraReset}
                camera={{ position: [0, 0, 6], fov: 45 }}
                dpr={[1, 1.5]}
                gl={{ antialias: true, alpha: true }}
                onPointerMissed={() => setSelectedId(null)}
              >
                <group ref={modelGroupRef}>
                  <ModelViewer
                    scene={modelScene}
                    onSelect={setSelectedId}
                  />
                </group>
                <GestureRaycastController
                  scene={modelScene}
                  cursorRef={handCursorRef}
                  pinchActive={pinchActive}
                  onSelect={setSelectedId}
                />
              </Canvas>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="flex max-w-sm flex-col items-center px-5 text-center">

                  <div className="relative flex h-32 w-32 items-center justify-center rounded-full border border-cyan-400/10 bg-cyan-400/[0.025]">
                    <div className="absolute inset-3 rounded-full border border-dashed border-cyan-400/20" />
                    <div className="absolute inset-7 rounded-full border border-white/[0.06]" />
                    <Box size={43} strokeWidth={1} className="text-cyan-400/60" />
                  </div>

                  <h2 className="mt-9 text-xl font-medium text-slate-200">
                    {loading ? 'Loading your model...' : 'Nothing to explore. Yet.'}
                  </h2>

                  <p className="mt-3 text-sm leading-7 text-slate-500">
                    {loading
                      ? 'Please wait while NEXUS processes your GLB model.'
                      : 'Upload a 3D model to begin exploring its structure and components.'}
                  </p>

                  {!loading && (
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-7 flex items-center gap-2 rounded-xl bg-cyan-400 px-6 py-3 text-sm font-semibold text-[#031019] transition hover:bg-cyan-300"
                    >
                      <Upload size={16} />
                      Upload GLB Model
                    </button>
                  )}

                  {loading && (
                    <div className="mt-7 h-1.5 w-40 overflow-hidden rounded-full bg-white/10">
                      <div className="h-full w-1/2 animate-pulse rounded-full bg-cyan-400" />
                    </div>
                  )}

                </div>
              </div>
            )}

            {/* ERROR MESSAGE */}

            {error && (
              <div className="absolute bottom-5 left-1/2 z-20 w-[90%] max-w-lg -translate-x-1/2 rounded-xl border border-red-400/20 bg-[#190d12]/95 p-4 text-sm text-red-300 shadow-xl">
                <div className="flex items-start justify-between gap-3">
                  <span>{error}</span>
                  <button
                    onClick={() => setError('')}
                    className="shrink-0 text-red-300/70 hover:text-white"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            )}

            {modelScene && (
              <div className="pointer-events-none absolute bottom-5 left-5 z-10 rounded-lg border border-white/[0.07] bg-[#080d14]/80 px-3 py-2 text-[10px] text-slate-400 backdrop-blur">
                Drag to rotate · Scroll to zoom · Click a mesh to select
              </div>
            )}

          </div>

          {/* BOTTOM TOOLBAR */}

          <div className="flex h-14 shrink-0 items-center justify-between border-t border-white/[0.06] bg-[#080d14] px-3 sm:px-5">

            <div className="flex items-center gap-4 text-xs text-slate-500">
              <div className="flex items-center gap-2">
                <MousePointer2 size={14} />
                <span className="hidden sm:block">Select</span>
              </div>

              <div className="flex items-center gap-2">
                <Rotate3D size={14} />
                <span className="hidden sm:block">Rotate</span>
              </div>

              <div className="flex items-center gap-2">
                <Eye size={14} />
                <span className="hidden sm:block">Zoom</span>
              </div>
            </div>

            <div className="flex items-center gap-3 text-[10px] text-slate-600">
              <span className="hidden sm:block">
                {modelScene ? `${components.length} COMPONENTS` : 'NO MODEL'}
              </span>
              <span className="rounded border border-white/[0.08] px-2 py-1">
                PHASE 4C
              </span>
            </div>
          </div>

        </section>

        {/* RIGHT INSPECTOR */}

        <AnimatePresence initial={false}>
          {rightOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 280, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="hidden shrink-0 overflow-hidden border-l border-white/[0.07] bg-[#080d14] lg:block"
            >
              <div className="flex h-full w-[280px] flex-col">

                <div className="flex h-14 items-center justify-between border-b border-white/[0.06] px-5">
                  <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.15em] text-slate-300">
                    <Eye size={16} className="text-cyan-400" />
                    INSPECTOR
                  </div>

                  <button
                    onClick={() => setRightOpen(false)}
                    title="Collapse inspector"
                    className="text-slate-500 hover:text-white"
                  >
                    <PanelRightClose size={17} />
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto p-5">

                  <div className="mb-7">
                    <p className="mb-3 text-[10px] font-semibold tracking-[0.18em] text-slate-500">
                      SELECTED COMPONENT
                    </p>

                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4">
                      <div className="flex items-center gap-3">
                        <div className="rounded-lg bg-cyan-400/[0.08] p-2.5 text-cyan-400">
                          <Box size={20} />
                        </div>

                        <div className="min-w-0">
                          <p className="truncate text-sm text-slate-300">
                            {selectedComponent?.name || 'Nothing selected'}
                          </p>
                          <p className="mt-1 text-[10px] text-slate-600">
                            {selectedComponent ? 'Mesh component' : 'Click a mesh to inspect'}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mb-7 border-t border-white/[0.06] pt-5">
                    <p className="mb-4 text-[10px] font-semibold tracking-[0.18em] text-slate-500">
                      GEOMETRY INFORMATION
                    </p>

                    {[
                      ['Geometry', selectedComponent?.geometry],
                      ['Vertices', selectedComponent?.vertices.toLocaleString()],
                      ['Components', components.length.toString()],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        className="flex items-center justify-between border-b border-white/[0.04] py-3"
                      >
                        <span className="text-xs text-slate-500">{label}</span>
                        <span className="max-w-[135px] truncate text-xs text-slate-300">
                          {value ?? '—'}
                        </span>
                      </div>
                    ))}
                  </div>

                  <div className="border-t border-white/[0.06] pt-5">
                    <p className="mb-3 text-[10px] font-semibold tracking-[0.18em] text-slate-500">
                      MODEL INFORMATION
                    </p>

                    <p className="break-all text-xs leading-6 text-slate-500">
                      {modelName || 'No model loaded'}
                    </p>

                    <p className="mt-3 text-xs text-slate-500">
                      {components.length} mesh components detected.
                    </p>
                  </div>

                </div>

                <div className="border-t border-white/[0.06] p-4">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Sparkles size={15} />
                    Component inspector
                  </div>
                </div>

              </div>
            </motion.aside>
          )}
        </AnimatePresence>

      </div>

      {/* STATUS BAR */}

      <footer className="flex h-8 shrink-0 items-center justify-between border-t border-white/[0.06] bg-[#060a10] px-4 text-[10px] text-slate-600">

        <div className="flex items-center gap-3">
          <span>NEXUS</span>
          <span className="text-slate-800">|</span>
          <span>{modelScene ? 'Model loaded' : 'Workspace ready'}</span>
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden sm:block">LOCAL SESSION</span>
          <span className="text-cyan-500/70">●</span>
        </div>

      </footer>
    </main>
  )
}

/* -----------------------------
   MAIN APPLICATION
----------------------------- */

function App() {
  const [workspace, setWorkspace] = useState(false)

  return (
    <AnimatePresence mode="wait">
      {workspace ? (
        <motion.div
          key="workspace"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          <Workspace onBack={() => setWorkspace(false)} />
        </motion.div>
      ) : (
        <motion.div
          key="landing"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          <LandingPage onStart={() => setWorkspace(true)} />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default App