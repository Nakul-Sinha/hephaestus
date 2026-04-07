"""
Reporter Agent (Agent 9)

Responsibilities:
    - Convert technical pipeline output to stakeholder-specific summaries
    - Generate three report variants:
      1. Operator Playbook — step-by-step maintenance instructions
      2. Manager Summary — one-page cost/impact overview
      3. Audit Trace — full agent-by-agent decision log
    - Output: structured JSON + natural language markdown

LLM Strategy:
    - Primary: Ollama local inference (Mistral/Llama) for report composition
    - Fallback: Google Gemini API when local inference is unavailable
"""
