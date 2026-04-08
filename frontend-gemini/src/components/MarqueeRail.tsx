"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";

interface MarqueeProps {
  items: string[];
  direction?: "left" | "right";
  speed?: number;
  className?: string;
}

export default function Marquee({ 
  items, 
  direction = "left", 
  speed = 20,
  className = "" 
}: MarqueeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    const content = contentRef.current;
    if (!content) return;

    const width = content.scrollWidth / 2;
    const duration = width / speed;

    const tween = gsap.to(content, {
      x: direction === "left" ? -width : width,
      duration: duration,
      ease: "none",
      repeat: -1,
      modifiers: {
        x: gsap.utils.unitize((x) => {
          const val = parseFloat(x);
          return direction === "left" 
            ? (val % width) - (val < 0 ? width : 0)
            : (val % width) + (val > 0 ? -width : 0);
        }),
      },
    });

    return () => {
      tween.kill();
    };
  }, [direction, speed]);

  const duplicatedItems = [...items, ...items, ...items, ...items];

  return (
    <div 
      ref={containerRef}
      className={`overflow-hidden whitespace-nowrap ${className}`}
    >
      <div 
        ref={contentRef}
        className="inline-flex items-center"
        style={{ willChange: "transform" }}
      >
        {duplicatedItems.map((item, i) => (
          <span 
            key={i}
            className="inline-flex items-center mx-8 text-2xl md:text-4xl font-bold text-neutral-700 hover:text-amber-500 transition-colors duration-300"
          >
            <span className="w-2 h-2 rounded-full bg-amber-500 mr-4" />
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}
