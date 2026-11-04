
import { useMemo } from 'react'
import * as THREE from 'three'
import { OrbitControls } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'

interface ModelViewerProps {
  scene: THREE.Object3D
  onSelect: (id: string) => void
}

export default function ModelViewer({
  scene,
  onSelect,
}: ModelViewerProps) {
  const bounds = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())

    const maxDimension = Math.max(size.x, size.y, size.z) || 1
    const scale = 3 / maxDimension

    return { center, scale }
  }, [scene])

  const handleSelect = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation()
    onSelect(event.object.uuid)
  }

  return (
    <>
      <ambientLight intensity={1.1} />
      <directionalLight position={[5, 8, 5]} intensity={2} />
      <pointLight position={[-5, -2, -5]} intensity={1} color="#38bdf8" />

      <gridHelper args={[12, 24, '#16445a', '#102536']} position={[0, -1.7, 0]} />

      <group
        position={[
          -bounds.center.x * bounds.scale,
          -bounds.center.y * bounds.scale,
          -bounds.center.z * bounds.scale,
        ]}
        scale={bounds.scale}
      >
        <primitive object={scene} onPointerDown={handleSelect} />
      </group>

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.8}
        zoomSpeed={0.9}
        minDistance={1}
        maxDistance={30}
      />
    </>
  )
}
