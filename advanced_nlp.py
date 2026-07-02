from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import requests


@dataclass(frozen=True)
class NLPInsight:
    label: str
    text: str
    count: int


@dataclass(frozen=True)
class SemanticRiskSignal:
    label: str
    score: float
    source: str


@dataclass(frozen=True)
class NLPReport:
    entities: list[NLPInsight] = field(default_factory=list)
    semantic_signals: list[SemanticRiskSignal] = field(default_factory=list)
    status: str = "Standard NLP"


CONFIDENTIAL_LABELS = [
    "confidential business information",
    "personal data",
    "financial data",
    "credentials or secrets",
    "public information",
]


def analyze_document_nlp(redacted_text: str) -> NLPReport:
    entities = _extract_spacy_entities(redacted_text)
    semantic_signals = _classify_with_huggingface(redacted_text)
    status = "Enhanced NLP" if semantic_signals else "Standard NLP"
    return NLPReport(entities=entities, semantic_signals=semantic_signals, status=status)


def _extract_spacy_entities(text: str) -> list[NLPInsight]:
    try:
        import spacy
        from spacy.pipeline import EntityRuler
    except ImportError:
        return _keyword_entities(text)

    try:
        nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
    except Exception:
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        assert isinstance(ruler, EntityRuler)
        ruler.add_patterns(
            [
                {"label": "CONFIDENTIAL", "pattern": [{"LOWER": "confidential"}]},
                {"label": "CONFIDENTIAL", "pattern": [{"LOWER": "internal"}, {"LOWER": "use"}, {"LOWER": "only"}]},
                {"label": "BUSINESS", "pattern": [{"LOWER": "pricing"}, {"LOWER": "strategy"}]},
                {"label": "BUSINESS", "pattern": [{"LOWER": "acquisition"}]},
                {"label": "BUSINESS", "pattern": [{"LOWER": "merger"}]},
                {"label": "SECRET", "pattern": [{"LOWER": "password"}]},
                {"label": "SECRET", "pattern": [{"LOWER": "token"}]},
            ]
        )

    doc = nlp(text[:100_000])
    counts = Counter((ent.label_, ent.text.strip()) for ent in doc.ents if ent.text.strip())
    return [
        NLPInsight(label=label, text=value, count=count)
        for (label, value), count in counts.most_common(25)
    ]


def _keyword_entities(text: str) -> list[NLPInsight]:
    labels = {
        "CONFIDENTIAL": ["confidential", "internal use only", "trade secret", "proprietary"],
        "BUSINESS": ["pricing strategy", "acquisition", "merger", "board meeting"],
        "SECRET": ["password", "token", "api key", "secret"],
    }
    insights: list[NLPInsight] = []
    lowered = text.lower()
    for label, terms in labels.items():
        for term in terms:
            count = lowered.count(term)
            if count:
                insights.append(NLPInsight(label=label, text=term, count=count))
    return insights


def _classify_with_huggingface(text: str) -> list[SemanticRiskSignal]:
    api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
    if not api_key:
        return []

    model = os.getenv("HUGGINGFACE_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
    endpoint = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": _compact_text(text, 3_000),
        "parameters": {"candidate_labels": CONFIDENTIAL_LABELS, "multi_label": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    labels = data.get("labels", [])
    scores = data.get("scores", [])
    signals = []
    for label, score in zip(labels, scores):
        try:
            signals.append(SemanticRiskSignal(label=str(label), score=float(score), source="Semantic classifier"))
        except (TypeError, ValueError):
            continue
    return signals[:5]


def _compact_text(text: str, max_chars: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def split_document(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    try:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
    except Exception:
        chunks = _fallback_split(text, chunk_size, chunk_overlap)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = _compact_text(text, max(len(text), 1))
    if not cleaned:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += step
    return chunks


def retrieve_relevant_context(text: str, query: str, top_k: int = 4) -> list[str]:
    chunks = split_document(text)
    if not chunks:
        return []

    query_text = query or "sensitive data compliance security risk remediation"
    local_results = _retrieve_with_hash_vectors(chunks, query_text, top_k)
    if local_results:
        return local_results

    if os.getenv("ENABLE_CHROMA_RAG", "").strip().lower() in {"1", "true", "yes"}:
        try:
            return _retrieve_with_chroma(chunks, query_text, top_k)
        except Exception:
            pass

    return _retrieve_with_lexical_score(chunks, query_text, top_k)


def _retrieve_with_chroma(chunks: list[str], query: str, top_k: int) -> list[str]:
    import chromadb

    client = chromadb.Client()
    collection_name = "document_context_" + hashlib.sha1("".join(chunks).encode("utf-8")).hexdigest()[:12]
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=[_hash_embedding(chunk) for chunk in chunks],
    )
    result = collection.query(
        query_embeddings=[_hash_embedding(query)],
        n_results=min(top_k, len(chunks)),
        include=["documents"],
    )
    documents = result.get("documents") or [[]]
    return [str(item) for item in documents[0]]


def _retrieve_with_hash_vectors(chunks: list[str], query: str, top_k: int) -> list[str]:
    query_vector = _hash_embedding(query)
    scored = []
    for chunk in chunks:
        chunk_vector = _hash_embedding(chunk)
        similarity = sum(query_value * chunk_value for query_value, chunk_value in zip(query_vector, chunk_vector))
        lexical_boost = _lexical_overlap_score(chunk, query)
        scored.append((similarity + lexical_boost, chunk))
    return [chunk for score, chunk in sorted(scored, reverse=True)[:top_k] if score > 0] or chunks[:top_k]


def _retrieve_with_lexical_score(chunks: list[str], query: str, top_k: int) -> list[str]:
    query_terms = set(_terms(query))
    scored = []
    for chunk in chunks:
        chunk_terms = _terms(chunk)
        overlap = sum(1 for term in chunk_terms if term in query_terms)
        density = overlap / max(1, len(chunk_terms))
        scored.append((overlap + density, chunk))
    return [chunk for score, chunk in sorted(scored, reverse=True)[:top_k] if score > 0] or chunks[:top_k]


def _lexical_overlap_score(chunk: str, query: str) -> float:
    query_terms = set(_terms(query))
    if not query_terms:
        return 0.0
    chunk_terms = set(_terms(chunk))
    return len(query_terms & chunk_terms) / len(query_terms)


def _hash_embedding(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    terms = _terms(text)
    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
