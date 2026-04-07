"""
Planner Agent (Agent 6)

Responsibilities:
    - Generate 3+ candidate intervention plans per incident
    - Each plan includes: recommended action, required parts, crew skills,
      estimated duration, maintenance window requirements
    - Uses LLM for structured plan synthesis with JSON schemas

LLM Strategy:
    - Primary: Ollama local inference (Mistral/Llama) with JSON output enforcement
    - Fallback: Google Gemini API with response_schema validation
"""
