"use client";

import { useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Float, PerspectiveCamera, Environment } from "@react-three/drei";
import { gsap } from "gsap";
import * as THREE from "three";

interface CardProps {
  position: [number, number, number];
  rotation: [number, number, number];
  title: string;
  description: string;
  color: string;
  index: number;
  activeIndex: number;
}

function FeatureCard({
  position,
  rotation,
  title,
  description,
  color,
  index,
  activeIndex,
}: CardProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const isActive = index === activeIndex;
  const isPrev = index === (activeIndex - 1 + 5) % 5;
  const isNext = index === (activeIndex + 1) % 5;

  useFrame((state) => {
    if (!meshRef.current) return;
    const time = state.clock.getElapsedTime();
    meshRef.current.position.y = position[1] + Math.sin(time + index) * 0.1;
  });

  return (
    <Float
      speed={2}
      rotationIntensity={isActive ? 0.2 : 0.05}
      floatIntensity={isActive ? 1 : 0.5}
    >
      <mesh
        ref={meshRef}
        position={position}
        rotation={rotation}
        scale={isActive ? 1.1 : isPrev || isNext ? 0.9 : 0.7}
      >
        <boxGeometry args={[3, 4, 0.2]} />
        <meshStandardMaterial
          color={color}
          metalness={0.5}
          roughness={0.2}
          emissive={isActive ? color : "#000000"}
          emissiveIntensity={isActive ? 0.2 : 0}
        />
      </mesh>

      {/* Glow plane behind */}
      <mesh position={[position[0], position[1], position[2] - 0.3]}>
        <planeGeometry args={[3.5, 4.5]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={isActive ? 0.3 : 0.05}
          side={THREE.DoubleSide}
        />
      </mesh>
    </Float>
  );
}

function Scene({ activeIndex }: { activeIndex: number }) {
  const { camera } = useThree();

  useEffect(() => {
    const angle = (activeIndex / 5) * Math.PI * 2;
    const radius = 8;
    const x = Math.sin(angle) * radius;
    const z = Math.cos(angle) * radius - 5;

    gsap.to(camera.position, {
      x,
      z,
      duration: 1,
      ease: "power2.out",
    });
    gsap.to(camera, {
      zoom: activeIndex === 0 ? 1 : 0.9,
      duration: 1,
    });
  }, [activeIndex, camera]);

  const features = [
    {
      title: "Predictive Analytics",
      description: "AI models predict failures weeks in advance",
      color: "#f59e0b",
      position: [0, 0, 0] as [number, number, number],
      rotation: [0, 0, 0] as [number, number, number],
    },
    {
      title: "Risk Scoring",
      description: "Dynamic risk assessment per asset",
      color: "#ef4444",
      position: [5, 0, 0] as [number, number, number],
      rotation: [0, -0.5, 0] as [number, number, number],
    },
    {
      title: "Root Cause AI",
      description: "Automated failure analysis",
      color: "#3b82f6",
      position: [3, 0, 4] as [number, number, number],
      rotation: [0, -2, 0] as [number, number, number],
    },
    {
      title: "Simulation Engine",
      description: "Monte Carlo intervention modeling",
      color: "#10b981",
      position: [-3, 0, 4] as [number, number, number],
      rotation: [0, 2, 0] as [number, number, number],
    },
    {
      title: "Fleet Command",
      description: "Unified operational dashboard",
      color: "#8b5cf6",
      position: [-5, 0, 0] as [number, number, number],
      rotation: [0, 0.5, 0] as [number, number, number],
    },
  ];

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#f59e0b" />
      <Environment preset="city" />

      {features.map((feature, i) => (
        <FeatureCard
          key={i}
          index={i}
          activeIndex={activeIndex}
          position={feature.position}
          rotation={feature.rotation}
          title={feature.title}
          description={feature.description}
          color={feature.color}
        />
      ))}

      {/* Floating particles */}
      {Array.from({ length: 20 }).map((_, i) => (
        <mesh
          key={`particle-${i}`}
          position={[
            (Math.random() - 0.5) * 20,
            (Math.random() - 0.5) * 10,
            (Math.random() - 0.5) * 10 - 5,
          ]}
        >
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshBasicMaterial color="#f59e0b" transparent opacity={0.6} />
        </mesh>
      ))}
    </>
  );
}

export default function FeatureCarousel() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  useEffect(() => {
    if (!isAutoPlaying) return;
    const interval = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % 5);
    }, 4000);
    return () => clearInterval(interval);
  }, [isAutoPlaying]);

  const features = [
    {
      title: "Predictive Analytics",
      desc: "AI models predict failures weeks in advance",
    },
    { title: "Risk Scoring", desc: "Dynamic risk assessment per asset" },
    { title: "Root Cause AI", desc: "Automated failure analysis" },
    { title: "Simulation Engine", desc: "Monte Carlo intervention modeling" },
    { title: "Fleet Command", desc: "Unified operational dashboard" },
  ];

  return (
    <div className="relative w-full h-[600px]">
      <Canvas
        className="w-full h-full"
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
      >
        <PerspectiveCamera makeDefault position={[0, 0, 8]} fov={50} />
        <Scene activeIndex={activeIndex} />
      </Canvas>

      {/* UI Overlay */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-4">
        <div className="glass-panel px-6 py-4 rounded-2xl text-center max-w-sm">
          <h3 className="text-xl font-bold text-white mb-1">
            {features[activeIndex].title}
          </h3>
          <p className="text-neutral-400 text-sm">
            {features[activeIndex].desc}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {features.map((_, i) => (
            <button
              key={i}
              onClick={() => {
                setActiveIndex(i);
                setIsAutoPlaying(false);
              }}
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                i === activeIndex
                  ? "w-8 bg-amber-500"
                  : "bg-neutral-600 hover:bg-neutral-500"
              }`}
            />
          ))}
        </div>

        <button
          onClick={() => setIsAutoPlaying(!isAutoPlaying)}
          className="text-xs text-neutral-500 hover:text-amber-500 transition-colors"
        >
          {isAutoPlaying ? "Pause" : "Auto-play"}
        </button>
      </div>
    </div>
  );
}
