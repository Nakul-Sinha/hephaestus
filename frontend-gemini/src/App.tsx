import { useState } from "react";
import { ForgeLoader } from "./components/ForgeLoader";
import { HeroSection } from "./components/HeroSection";
import { RPGZone } from "./components/RPGZone";
import FeatureCarousel from "./components/FeatureCarousel";
import Marquee from "./components/MarqueeRail";
import ParticleOverlay from "./components/ParticleOverlay";

function App() {
  const [loading, setLoading] = useState(true);

  return (
    <div className="grain-overlay relative min-h-screen bg-obsidian text-white no-scrollbar">
      {loading && <ForgeLoader onFinish={() => setLoading(false)} />}
      <ParticleOverlay />

      <header className="fixed left-0 right-0 top-0 z-50 border-b border-white/10 bg-black/40 backdrop-blur pointer-events-none">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4 md:px-16 pointer-events-auto">
          <div className="flex items-center gap-3 text-sm uppercase tracking-[0.4em] text-white/70">
            <span className="h-2 w-2 rounded-full bg-ember" />
            Hephaestus
          </div>
          <nav className="hidden items-center gap-6 text-xs uppercase tracking-[0.4em] text-white/60 md:flex">
            <a href="#hero" className="hover:text-white transition-colors">Hero</a>
            <a href="#command-center" className="hover:text-white transition-colors">Command Center</a>
            <a href="#simulation" className="hover:text-white transition-colors">Simulation</a>
          </nav>
          <button className="button-glow rounded-full border border-white/30 px-4 py-2 text-xs uppercase tracking-[0.3em] text-white/70 hover:border-white transition-colors duration-300">
            Request access
          </button>
        </div>
      </header>

      <main className="relative z-10 w-full overflow-hidden">
        <HeroSection enableSpline={!loading} />
        
        <Marquee items={["THE INDUSTRIAL A.I. OVERLORD"]} />
        
        {/* Command Center: Map spatial UI zones per person-1.md */}
        <RPGZone />

        <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        
        <section id="simulation" className="relative py-24 bg-graphite">
          <div className="max-w-6xl mx-auto px-6 mb-16 text-center">
            <div className="text-xs uppercase tracking-[0.4em] text-auric mb-4">Core Tech</div>
            <h2 className="font-display text-4xl md:text-6xl font-bold tracking-tight text-white mb-6">Interactive Simulation Engine</h2>
            <p className="max-w-2xl mx-auto text-white/50 text-lg">
              Explore our core capabilities powered by 3D WebGL computation and real-time inference mechanics. 
              Drag to rotate the carousel of interventions.
            </p>
          </div>
          
          <FeatureCarousel />
        </section>

        <Marquee items={["HEPHAESTUS FORGE EXPERIMENT"]} direction="right" speed={10} />
        
      </main>

      <footer className="relative z-20 border-t border-white/10 bg-[radial-gradient(ellipse_at_bottom,rgba(255,106,42,0.1)_0%,#0a0f16_60%)] px-6 py-12 text-center text-xs uppercase tracking-[0.4em] text-white/40 md:px-16">
        Hephaestus Forge © 2026 — Built for real-time decision craft
      </footer>
    </div>
  );
}

export default App;
