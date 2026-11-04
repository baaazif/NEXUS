export interface HandLandmark {
  x: number
  y: number
  z: number
}
export interface GestureClassification {
  gesture: string
  confidence: number
}

interface Landmark {
  x: number
  y: number
  z?: number
}

const distance = (a: Landmark, b: Landmark): number => {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0))
}

const angle = (a: Landmark, b: Landmark, c: Landmark): number => {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) }
  const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) }

  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z
  const magnitude = Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z)

  if (magnitude === 0) return 0

  return Math.acos(Math.max(-1, Math.min(1, dot / magnitude))) * (180 / Math.PI)
}

export function classifyGesture(
  landmarks: Landmark[],
): GestureClassification {
  if (landmarks.length !== 21) {
    return { gesture: 'Unknown', confidence: 0 }
  }

  const wrist = landmarks[0]
  const palmSize = distance(wrist, landmarks[9]) || 0.001

  const fingers = [
    { tip: 8, pip: 6, mcp: 5 },
    { tip: 12, pip: 10, mcp: 9 },
    { tip: 16, pip: 14, mcp: 13 },
    { tip: 20, pip: 18, mcp: 17 },
  ]

  const extended = fingers.map(({ tip, pip, mcp }) => {
    const straightAngle = angle(landmarks[mcp], landmarks[pip], landmarks[tip])
    const tipDistance = distance(wrist, landmarks[tip])
    const pipDistance = distance(wrist, landmarks[pip])

    return straightAngle > 150 && tipDistance > pipDistance * 1.08
  })

  const [index, middle, ring, pinky] = extended

  const thumbTip = landmarks[4]
  const thumbIp = landmarks[3]
  const thumbMcp = landmarks[2]

  const thumbAngle = angle(thumbMcp, thumbIp, thumbTip)
  const thumbExtended =
    thumbAngle > 145 &&
    distance(wrist, thumbTip) > distance(wrist, thumbIp) * 1.05

  const pinchDistance = distance(thumbTip, landmarks[8]) / palmSize

  // Pinch has priority because it can coexist with other finger positions.
  if (pinchDistance < 0.28) {
    const confidence = Math.min(0.99, 0.65 + (0.28 - pinchDistance) * 1.1)
    return { gesture: 'Pinch', confidence }
  }

  const extendedCount =
    Number(thumbExtended) + extended.filter(Boolean).length

  if (extended.every(Boolean) && thumbExtended) {
    return { gesture: 'Open Palm', confidence: 0.95 }
  }

  if (!extended.some(Boolean) && !thumbExtended) {
    return { gesture: 'Closed Fist', confidence: 0.9 }
  }

  if (index && !middle && !ring && !pinky) {
    return {
      gesture: 'Pointing',
      confidence: 0.85 + (thumbExtended ? 0 : 0.05),
    }
  }

  if (index && middle && !ring && !pinky) {
    return { gesture: 'Two-Finger', confidence: 0.9 }
  }

  if (extendedCount >= 3) {
    return { gesture: 'Open Palm', confidence: 0.65 }
  }

  return { gesture: 'Unknown', confidence: 0.4 }
}
