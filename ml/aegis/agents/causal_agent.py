"""
Causal Agent (Agent 5)

Responsibilities:
    - Produce ranked root-cause hypotheses
    - Build hypothesis graph with confidence and evidence weighting
    - Correlate telemetry spikes, events, and maintenance history
    - Handle contradictions between evidence sources
    - Uses LLM for causal reasoning synthesis

LLM Strategy:
    - Primary: Ollama local inference (Mistral/Llama) via /api/generate with JSON mode
    - Fallback: Google Gemini API (when Ollama is unavailable or errors after retries)
"""
