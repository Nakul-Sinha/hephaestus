import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Activity, PackageSearch, ShieldCheck } from "lucide-react";

type Zone = "fleet" | "incident" | "simulation" | "audit" | null;

const zones = [
  {
    id: "fleet",
    icon: <Activity size={24} className="text-ember" />,
    label: "Fleet Overview",
    color: "bg-ember/10 border-ember/30 text-ember",
    position: "top-[10%] left-[20%]",
    content: (
      <div className="space-y-4">
        <h4 className="font-display text-xl text-white">Asset Risk Heatmap</h4>
        <div className="flex gap-2">
          <div className="h-8 w-24 rounded bg-red-500/20 border border-red-500/40 flex items-center justify-center text-xs text-red-400">Critical</div>
          <div className="h-8 w-24 rounded bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-xs text-amber-400">Warning</div>
        </div>
        <div className="mt-4 rounded bg-white/5 p-3 border border-white/10">
          <p className="text-sm text-white/70">Top At-Risk: TR-94 Turbine</p>
          <p className="text-xs text-white/50">84% failure probability detected in 48h</p>
        </div>
      </div>
    )
  },
  {
    id: "incident",
    icon: <AlertTriangle size={24} className="text-auric" />,
    label: "Incident Workbench",
    color: "bg-auric/10 border-auric/30 text-auric",
    position: "top-[30%] right-[15%]",
    content: (
      <div className="space-y-4">
        <h4 className="font-display text-xl text-white">Root-cause Hypothesis</h4>
        <div className="space-y-2">
          <div className="rounded border border-white/20 bg-white/5 p-3">
            <div className="text-xs uppercase text-auric/70">Plan A: Fast Reroute</div>
            <div className="flex justify-between text-sm mt-1">
              <span>Risk: -80%</span>
              <span>Cost: $12k</span>
            </div>
          </div>
          <div className="rounded border border-white/10 bg-black/40 p-3">
            <div className="text-xs uppercase text-white/50">Plan B: Hot Swap</div>
            <div className="flex justify-between text-sm mt-1 text-white/60">
              <span>Risk: -99%</span>
              <span>Cost: $45k</span>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    id: "simulation",
    icon: <PackageSearch size={24} className="text-neon" />,
    label: "Simulation Panel",
    color: "bg-neon/10 border-neon/30 text-neon",
    position: "bottom-[30%] left-[10%]",
    content: (
      <div className="space-y-4">
        <h4 className="font-display text-xl text-white">30-Day Trajectory</h4>
        <div className="h-24 w-full relative border-b border-l border-white/20 overflow-hidden">
          <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none" viewBox="0 0 100 100">
            <path d="M0 80 Q 25 70, 50 40 T 100 10" fill="none" stroke="rgba(62,240,214,0.6)" strokeWidth="2" />
            <path d="M0 60 Q 25 50, 50 80 T 100 60" fill="none" stroke="rgba(255,106,42,0.6)" strokeWidth="2" strokeDasharray="4 4" />
          </svg>
        </div>
      </div>
    )
  },
  {
    id: "audit",
    icon: <ShieldCheck size={24} className="text-white" />,
    label: "Audit View",
    color: "bg-white/10 border-white/30 text-white",
    position: "bottom-[15%] right-[25%]",
    content: (
      <div className="space-y-4">
        <h4 className="font-display text-xl text-white">Agent Trace</h4>
        <div className="border-l border-white/20 pl-4 space-y-3 pb-2 text-sm text-white/60">
          <div className="relative">
            <div className="absolute -left-5 top-1.5 h-2 w-2 rounded-full bg-white/50" />
            <p className="text-white/80">Anomaly Detected [Agent Alpha]</p>
          </div>
          <div className="relative">
            <div className="absolute -left-5 top-1.5 h-2 w-2 rounded-full bg-ember/80" />
            <p className="text-ember">Human review requested</p>
          </div>
        </div>
      </div>
    )
  }
];

export const RPGZone = () => {
  const [activeZone, setActiveZone] = useState<Zone>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const { left, top, width, height } = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - left) / width;
    const y = (e.clientY - top) / height;

    let closestZone: Zone = null;
    let minDist = 0.15; 

    zones.forEach(z => {
      let zx = 0.5, zy = 0.5;
      if (z.id === 'fleet') { zx = 0.3; zy = 0.2; }
      if (z.id === 'incident') { zx = 0.8; zy = 0.4; }
      if (z.id === 'simulation') { zx = 0.2; zy = 0.7; }
      if (z.id === 'audit') { zx = 0.7; zy = 0.8; }

      const dist = Math.sqrt(Math.pow(x - zx, 2) + Math.pow(y - zy, 2));
      if (dist < minDist) {
        minDist = dist;
        closestZone = z.id as Zone;
      }
    });

    setActiveZone(closestZone);
  };

  return (
    <section 
      className="relative h-screen w-full bg-[#0a0f16] overflow-hidden snap-section"
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setActiveZone(null)}
      id="command-center"
    >
      <div className="absolute top-10 left-10 z-20 text-xs uppercase tracking-[0.4em] text-white/50">
        Command Center / Sector 7G
      </div>

      <div className="absolute inset-0 z-0 flex items-center justify-center opacity-80 pointer-events-none">
        <div 
          className="relative w-[120vw] h-[120vh] border border-[#232936] bg-[#0c121b]"
          style={{
            transform: 'rotateX(60deg) rotateZ(-45deg) scale(1.2)',
            transformOrigin: 'center center',
            backgroundImage: 'linear-gradient(#232936 2px, transparent 2px), linear-gradient(90deg, #232936 2px, transparent 2px)',
            backgroundSize: '100px 100px',
            boxShadow: 'inset 0 0 200px rgba(0,0,0,0.8)'
          }}
        >
          <div className="absolute top-0 left-[200px] w-[2px] h-full bg-gradient-to-b from-transparent via-ember to-transparent opacity-50 animate-[ping_4s_ease-in-out_infinite]" />
          <div className="absolute top-[300px] left-0 h-[2px] w-full bg-gradient-to-r from-transparent via-auric to-transparent opacity-50 animate-[ping_6s_ease-in-out_infinite]" />
          
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] rounded-full border-4 border-neon/30 bg-neon/5 shadow-[0_0_100px_rgba(62,240,214,0.2)]" />
        </div>
      </div>

      <div ref={containerRef} className="relative z-10 w-full h-full max-w-7xl mx-auto">
        {zones.map((zone) => {
          const isActive = activeZone === zone.id;
          
          return (
            <div key={zone.id} className={`absolute ${zone.position} transition-all duration-500`}>
              <div 
                className={`relative flex items-center justify-center w-16 h-16 rounded-xl backdrop-blur border shadow-2xl transition-all duration-500 cursor-pointer hover:scale-110 
                  ${isActive ? `${zone.color} shadow-${zone.color.split('/')[0].split('-')[1]}/50` : 'bg-black/40 border-white/10 text-white/40'}`}
                onMouseEnter={() => setActiveZone(zone.id as Zone)}
              >
                {zone.icon}
                {isActive && (
                  <div className="absolute -inset-2 border border-current rounded-xl opacity-20 animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite]" />
                )}
              </div>

              <AnimatePresence>
                {isActive && (
                  <motion.div
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    className="absolute top-20 left-0 w-80 rounded-xl border border-white/10 bg-black/60 p-6 backdrop-blur-xl shadow-2xl"
                  >
                    <div className="mb-4 flex items-center gap-3 border-b border-white/10 pb-4">
                      {zone.icon}
                      <h3 className="font-display font-medium tracking-wide text-white">
                        {zone.label}
                      </h3>
                    </div>
                    {zone.content}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </section>
  );
};
