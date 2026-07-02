from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, TypedDict

from advanced_nlp import retrieve_relevant_context
from sensitive_detector import (
    Detection,
    RiskAssessment,
    answer_question,
    build_summary,
    mask_value,
    redact_text,
)


@dataclass(frozen=True)
class AIConfig:
    provider: str = "gemini"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.1
    secondary_provider: str = ""
    secondary_api_key: str = ""
    secondary_model: str = ""
    fast_responses: bool = False

    @property
    def enabled(self) -> bool:
        return bool(_configured_models(self))


class ComplianceGraphState(TypedDict, total=False):
    text: str
    detections: list[Detection]
    risk: RiskAssessment
    question: str
    config: AIConfig
    redacted_text: str
    findings: list[dict[str, Any]]
    category_counts: dict[str, int]
    summary: dict[str, list[str]]
    answer: str
    ai_status: str
    error: str
    retrieved_context: list[str]


def generate_ai_summary(
    text: str,
    detections: list[Detection],
    risk: RiskAssessment,
    config: AIConfig,
) -> tuple[dict[str, list[str]], str]:
    fallback = build_summary(text, detections, risk)
    if not config.enabled:
        return fallback, _inactive_status(config)

    state = _run_graph({"text": text, "detections": detections, "risk": risk, "config": config})
    summary = state.get("summary") or fallback
    return summary, state.get("ai_status", "Rules fallback")


def answer_question_with_ai(
    question: str,
    text: str,
    detections: list[Detection],
    risk: RiskAssessment,
    config: AIConfig,
) -> tuple[str, str]:
    fallback = answer_question(question, text, detections, risk)
    if config.fast_responses and _should_answer_locally(question):
        return fallback, "Instant document analysis"
    if not config.enabled:
        return fallback, _inactive_status(config)

    state = _run_graph(
        {
            "text": text,
            "detections": detections,
            "risk": risk,
            "question": question,
            "config": config,
        }
    )
    return state.get("answer") or fallback, state.get("ai_status", "Rules fallback")


def _should_answer_locally(question: str) -> bool:
    normalized = question.lower().strip()
    fast_signals = (
        "what sensitive",
        "sensitive data",
        "how many",
        "count",
        "summarize",
        "summary",
        "compliance",
        "risk",
        "remediate",
        "remediation",
        "audit note",
        "redact",
        "mask",
    )
    return any(signal in normalized for signal in fast_signals)


def _run_graph(initial_state: ComplianceGraphState) -> ComplianceGraphState:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _fallback_state(initial_state, "LangGraph is not installed.")

    try:
        graph = StateGraph(ComplianceGraphState)
        graph.add_node("prepare_context", _prepare_context)
        graph.add_node("generate_summary", _generate_summary)
        graph.add_node("answer_question", _answer_question)
        graph.add_edge(START, "prepare_context")
        graph.add_edge("prepare_context", "generate_summary")
        graph.add_edge("generate_summary", "answer_question")
        graph.add_edge("answer_question", END)
        compiled = graph.compile()
        return compiled.invoke(initial_state)
    except Exception as exc:
        return _fallback_state(initial_state, str(exc))


def _prepare_context(state: ComplianceGraphState) -> ComplianceGraphState:
    text = state["text"]
    detections = state["detections"]
    category_counts = Counter(item.category for item in detections)
    findings = [
        {
            "category": item.category,
            "masked_value": mask_value(item.value),
            "severity": item.severity,
            "confidence": round(item.confidence, 2),
            "reason": item.reason,
        }
        for item in detections
    ]
    retrieved_context = retrieve_relevant_context(
        redact_text(text, detections),
        state.get("question", "sensitive data compliance risk remediation"),
    )
    return {
        **state,
        "redacted_text": redact_text(text, detections)[:12_000],
        "findings": findings,
        "category_counts": dict(category_counts),
        "retrieved_context": retrieved_context,
    }


def _generate_summary(state: ComplianceGraphState) -> ComplianceGraphState:
    config = state["config"]
    fallback = build_summary(state["text"], state["detections"], state["risk"])
    prompt = _summary_prompt(state)
    summaries: list[dict[str, list[str]]] = []
    errors: list[str] = []

    for model_config in _configured_models(config):
        try:
            response = _build_llm(model_config).invoke(
                [
                    (
                        "system",
                        "You are a security and privacy compliance analyst. Use only the supplied redacted document context. "
                        "Do not invent facts and do not reveal or reconstruct masked sensitive values.",
                    ),
                    ("human", prompt),
                ]
            )
            summaries.append(_parse_summary_json(_message_content(response)))
        except Exception as exc:
            errors.append(str(exc))

    if summaries:
        return {**state, "summary": _merge_summaries(summaries), "ai_status": "Enhanced analysis"}

    error = "; ".join(errors) or "No configured model returned a response."
    return {
        **state,
        "summary": fallback,
        "ai_status": f"Rules fallback ({error})",
        "error": error,
    }


def _answer_question(state: ComplianceGraphState) -> ComplianceGraphState:
    question = state.get("question", "").strip()
    if not question:
        return state

    config = state["config"]
    fallback = answer_question(question, state["text"], state["detections"], state["risk"])
    prompt = _qa_prompt(state, question)
    answers: list[str] = []
    errors: list[str] = []

    for model_config in _configured_models(config):
        try:
            response = _build_llm(model_config).invoke(
                [
                    (
                        "system",
                        "You answer document compliance questions using only supplied redacted context and findings. "
                        "Be concise. Never reveal or guess raw sensitive values.",
                    ),
                    ("human", prompt),
                ]
            )
            answer = _message_content(response).strip()
            if answer:
                answers.append(answer)
        except Exception as exc:
            errors.append(str(exc))

    if answers:
        return {
            **state,
            "answer": _merge_answers(answers),
            "ai_status": "Enhanced analysis",
        }

    error = "; ".join(errors) or "No configured model returned a response."
    return {
        **state,
        "answer": fallback,
        "ai_status": f"Rules fallback ({error})",
        "error": error,
    }


def _build_llm(config: AIConfig):
    if config.provider == "groq":
        os.environ["GROQ_API_KEY"] = config.api_key.strip()
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.model or "qwen/qwen3-32b",
            temperature=config.temperature,
            max_retries=2,
        )

    if config.provider == "gemini":
        previous_google_key = os.environ.get("GOOGLE_API_KEY")
        previous_gemini_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GOOGLE_API_KEY"] = config.api_key.strip()
        os.environ.pop("GEMINI_API_KEY", None)
        from langchain_google_genai import ChatGoogleGenerativeAI

        try:
            return ChatGoogleGenerativeAI(
                model=config.model or "gemini-2.5-flash",
                temperature=config.temperature,
                max_retries=2,
                google_api_key=config.api_key.strip(),
            )
        finally:
            if previous_gemini_key is not None:
                os.environ["GEMINI_API_KEY"] = previous_gemini_key
            if previous_google_key is not None:
                os.environ["GOOGLE_API_KEY"] = previous_google_key
            else:
                os.environ.pop("GOOGLE_API_KEY", None)

    raise ValueError("Unsupported AI provider.")


def _configured_models(config: AIConfig) -> list[AIConfig]:
    configs = []
    if config.provider in {"groq", "gemini"} and config.api_key.strip():
        configs.append(
            AIConfig(
                provider=config.provider,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                fast_responses=config.fast_responses,
            )
        )
    if config.secondary_provider in {"groq", "gemini"} and config.secondary_api_key.strip():
        configs.append(
            AIConfig(
                provider=config.secondary_provider,
                api_key=config.secondary_api_key,
                model=config.secondary_model,
                temperature=config.temperature,
                fast_responses=config.fast_responses,
            )
        )
    return configs


def _summary_prompt(state: ComplianceGraphState) -> str:
    risk = state["risk"]
    return f"""
Return strict JSON with exactly these keys:
- "Compliance observations": array of 3 to 5 short strings
- "Security risks": array of 3 to 5 short strings
- "Suggested remediation steps": array of 3 to 5 short strings

Make the response document-specific. Mention the highest-count categories, the highest-severity categories,
and one visible redacted context cue. Do not return generic boilerplate unless it is tied to the supplied findings.

Risk level: {risk.level}
Risk score: {risk.score}/100
Risk reasons: {risk.reasons}
Detected category counts: {json.dumps(state["category_counts"], indent=2)}
Masked findings: {json.dumps(state["findings"], indent=2)}

Most relevant document sections:
{_format_context(state.get("retrieved_context", []))}

Redacted document excerpt:
{state["redacted_text"]}
""".strip()


def _qa_prompt(state: ComplianceGraphState, question: str) -> str:
    risk = state["risk"]
    return f"""
Question: {question}

Risk level: {risk.level}
Risk score: {risk.score}/100
Detected category counts: {json.dumps(state["category_counts"], indent=2)}
Masked findings: {json.dumps(state["findings"], indent=2)}

Most relevant document sections:
{_format_context(state.get("retrieved_context", []))}

Redacted document excerpt:
{state["redacted_text"]}
""".strip()


def _parse_summary_json(content: str) -> dict[str, list[str]]:
    cleaned = content.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    required = ["Compliance observations", "Security risks", "Suggested remediation steps"]
    parsed: dict[str, list[str]] = {}
    for key in required:
        value = data.get(key, [])
        if isinstance(value, str):
            parsed[key] = [value]
        elif isinstance(value, list):
            parsed[key] = [str(item) for item in value if str(item).strip()]
        else:
            parsed[key] = []
    return parsed


def _merge_summaries(summaries: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    required = ["Compliance observations", "Security risks", "Suggested remediation steps"]
    merged: dict[str, list[str]] = {}
    for key in required:
        items: list[str] = []
        seen = set()
        for summary in summaries:
            for item in summary.get(key, []):
                normalized = re.sub(r"\W+", " ", item.lower()).strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    items.append(item)
        merged[key] = items[:5]
    return merged


def _merge_answers(answers: list[str]) -> str:
    unique_answers = []
    seen = set()
    for answer in answers:
        normalized = re.sub(r"\W+", " ", answer.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_answers.append(answer)

    if len(unique_answers) <= 1:
        return unique_answers[0] if unique_answers else ""

    best = max(unique_answers, key=lambda answer: (_answer_score(answer), len(answer)))
    supporting = [answer for answer in unique_answers if answer != best]
    additions = _extract_unique_sentences(best, supporting)
    if additions:
        return best.rstrip() + "\n\nAdditional useful detail: " + " ".join(additions[:2])
    return best


def _answer_score(answer: str) -> int:
    terms = ["risk", "sensitive", "compliance", "remediation", "detected", "redact", "privacy", "security"]
    lowered = answer.lower()
    return sum(1 for term in terms if term in lowered)


def _extract_unique_sentences(base_answer: str, other_answers: list[str]) -> list[str]:
    base_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", base_answer.lower()))
    additions = []
    for answer in other_answers:
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
            terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", sentence.lower()))
            if terms and len(terms - base_terms) >= 3:
                additions.append(sentence)
                base_terms.update(terms)
    return additions


def _message_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _format_context(chunks: list[str]) -> str:
    if not chunks:
        return "No retrieved sections available."
    return "\n\n".join(f"[Section {index + 1}]\n{chunk}" for index, chunk in enumerate(chunks))


def _fallback_state(state: ComplianceGraphState, error: str) -> ComplianceGraphState:
    config = state.get("config", AIConfig())
    question = state.get("question", "")
    summary = build_summary(state["text"], state["detections"], state["risk"])
    answer = answer_question(question, state["text"], state["detections"], state["risk"]) if question else ""
    return {
        **state,
        "summary": summary,
        "answer": answer,
        "ai_status": f"Rules fallback ({error})",
        "error": error,
    }


def _inactive_status(config: AIConfig) -> str:
    return "Standard analysis"
