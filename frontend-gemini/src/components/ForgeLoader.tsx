import { useEffect, useMemo, useRef, useState } from "react";

type LoaderMode = "webgpu" | "webgl" | "css";

const detectMode = (): LoaderMode => {
  if ("gpu" in navigator) return "webgpu";
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  if (gl) return "webgl";
  return "css";
};

export const ForgeLoader = ({ onFinish }: { onFinish: () => void }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mode, setMode] = useState<LoaderMode>("css");
  const [progress, setProgress] = useState(0);
  const showCanvas = mode !== "css";

  const loaderLabel = useMemo(() => {
    if (mode === "webgpu") return "WebGPU";
    if (mode === "webgl") return "WebGL";
    return "CSS";
  }, [mode]);

  useEffect(() => {
    setMode(detectMode());
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || mode === "css") return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame = 0;
    let animationId = 0;
    let running = true;

    const resize = () => {
      const { innerWidth, innerHeight } = window;
      canvas.width = innerWidth * window.devicePixelRatio;
      canvas.height = innerHeight * window.devicePixelRatio;
      canvas.style.width = `${innerWidth}px`;
      canvas.style.height = `${innerHeight}px`;
    };

    resize();
    window.addEventListener("resize", resize);

    const text = "HEPHAESTUS";
    const offscreen = document.createElement("canvas");
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    const fontSize = Math.min(200, window.innerWidth * 0.18);
    offscreen.width = window.innerWidth * window.devicePixelRatio;
    offscreen.height = window.innerHeight * window.devicePixelRatio;
    offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
    offCtx.fillStyle = "white";
    offCtx.textAlign = "center";
    offCtx.textBaseline = "middle";
    offCtx.font = `700 ${fontSize}px Unbounded, sans-serif`;
    offCtx.fillText(text, offscreen.width / 2, offscreen.height / 2);

    const imageData = offCtx.getImageData(0, 0, offscreen.width, offscreen.height);
    const points: { x: number; y: number; tx: number; ty: number; r: number; }[] = [];

    for (let y = 0; y < imageData.height; y += 8) {
      for (let x = 0; x < imageData.width; x += 8) {
        const idx = (y * imageData.width + x) * 4;
        if (imageData.data[idx + 3] > 140) {
          points.push({
            x: Math.random() * imageData.width,
            y: Math.random() * imageData.height,
            tx: x,
            ty: y,
            r: 1 + Math.random() * 1.8,
          });
        }
      }
    }

    const animate = () => {
      if (!running) return;
      frame += 1;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      ctx.fillStyle = "rgba(11, 11, 17, 0.22)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const t = Math.min(frame / 90, 1);
      setProgress(t);
      ctx.globalCompositeOperation = "lighter";

      points.forEach((p) => {
        p.x += (p.tx - p.x) * (0.08 + t * 0.06);
        p.y += (p.ty - p.y) * (0.08 + t * 0.06);
        const jitter = (1 - t) * 8 * Math.sin((p.x + p.y + frame) * 0.002);
        ctx.beginPath();
        ctx.fillStyle = `rgba(255, 106, 42, ${0.45 + t * 0.45})`;
        ctx.arc(p.x / window.devicePixelRatio + jitter, p.y / window.devicePixelRatio + jitter, p.r, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.restore();
      animationId = window.requestAnimationFrame(animate);
    };

    animationId = window.requestAnimationFrame(animate);

    return () => {
      running = false;
      window.removeEventListener("resize", resize);
      window.cancelAnimationFrame(animationId);
    };
  }, [mode]);

  useEffect(() => {
    const timer = window.setTimeout(() => onFinish(), 1700);
    return () => window.clearTimeout(timer);
  }, [onFinish]);

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-obsidian text-white">
      {showCanvas && <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />}
      <div className="relative z-10 flex flex-col items-center gap-6 text-center">
        <div className="text-sm uppercase tracking-[0.6em] text-white/50">{loaderLabel} Forge</div>
        <div className="text-4xl md:text-6xl font-display text-gradient glow-text">Hephaestus</div>
        <div className="h-[2px] w-64 overflow-hidden rounded-full bg-white/10">
          <div className="h-full bg-gradient-to-r from-ember via-auric to-neon" style={{ width: `${Math.min(progress * 100, 100)}%` }} />
        </div>
      </div>
    </div>
  );
};
