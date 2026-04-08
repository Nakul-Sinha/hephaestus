import { useEffect, useRef } from "react";
import Spline from "@splinetool/react-spline";

const SCENE_URL = "https://prod.spline.design/6bb58e3d-6f46-415a-85a9-9788abf1f309/scene.splinecode";

export const HeroSection = ({ enableSpline }: { enableSpline: boolean }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const splineWrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let gsapCtx: any;
    import("gsap").then(({ gsap }) => {
      import("gsap/ScrollTrigger").then(({ ScrollTrigger }) => {
        gsap.registerPlugin(ScrollTrigger);

        gsapCtx = gsap.context(() => {
          gsap.to(textRef.current, {
            y: -200,
            opacity: 0,
            ease: "none",
            scrollTrigger: {
              trigger: containerRef.current,
              start: "top top",
              end: "bottom top",
              scrub: 0.5,
            }
          });

          gsap.to(splineWrapRef.current, {
            scale: 0.8,
            y: 100,
            ease: "none",
            scrollTrigger: {
              trigger: containerRef.current,
              start: "top top",
              end: "bottom top",
              scrub: 0.5,
            }
          });
        }, containerRef);
      });
    });

    return () => {
      if (gsapCtx) gsapCtx.revert();
    };
  }, []);

  return (
    <section ref={containerRef} id="hero" className="relative h-[150vh] w-full bg-[#0b0b11]">
      <div className="sticky top-0 h-screen w-full overflow-hidden hero-mask">
        
        <div ref={splineWrapRef} className="absolute inset-0 z-0 flex items-center justify-center opacity-80 pointer-events-none md:pointer-events-auto">
          <div className="h-full w-full bg-gradient-to-b from-[#111] to-[#0a0f16]" />
        </div>

        <div className="absolute inset-0 z-10 bg-[radial-gradient(ellipse_at_center,transparent_0%,#0b0b11_80%)] pointer-events-none" />

        <div ref={textRef} className="relative z-20 flex h-full flex-col items-center justify-center text-center pointer-events-none px-4">
          <div className="mb-6 flex gap-2 text-xs uppercase tracking-[0.5em] text-white/50">
            <span className="text-ember">01</span> — The Core
          </div>
          <h1 className="font-display text-5xl md:text-8xl lg:text-[10rem] font-bold tracking-tighter text-white drop-shadow-2xl">
            HEPHAESTUS
          </h1>
          <p className="mt-6 max-w-xl text-sm md:text-lg text-white/60 drop-shadow-lg">
            Immersive 3D Web Experience. Scroll to pull back the camera and explore the Command Center.
          </p>
        </div>

      </div>
    </section>
  );
};
