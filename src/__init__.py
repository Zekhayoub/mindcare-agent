"""MindCare Agent — carbon-aware emotional support system.

Modules:
    config: Centralized YAML configuration loader with schema validation.
    analysis: Context detection, intensity estimation, emotion scoring.
    carbon: CO2 footprint calculation per query (ECO/HYBRID/AGENT).
    prompts: Centralized prompt templates for ReAct agent and HYBRID verification.
    signals: Independent routing signal evaluators (safety, confidence, complexity, shift).
    scorer: Weighted multi-signal aggregation with three routing zones.
    strategist: LangGraph StateGraph orchestrating the full routing pipeline.

Sub-packages:
    tools: ML classifier, CSV advice database, RAG retrieval, geolocation.
    agent: LangChain ReAct agent construction, invocation, and NER utilities.
"""
