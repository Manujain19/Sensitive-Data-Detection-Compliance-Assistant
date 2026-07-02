from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    category: str
    value: str
    start: int
    end: int
    severity: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    score: int
    reasons: list[str] = field(default_factory=list)


PATTERNS: dict[str, dict[str, object]] = {
    "Aadhaar Number": {
        "regex": r"(?<!\d)(?:\d{4}[\s-]?\d{4}[\s-]?\d{4})(?!\d)",
        "severity": "High",
        "confidence": 0.94,
        "reason": "12 digit Indian national identity number pattern.",
    },
    "PAN Number": {
        "regex": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "severity": "High",
        "confidence": 0.96,
        "reason": "Indian PAN format: five letters, four digits, one letter.",
    },
    "Email Address": {
        "regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "severity": "Medium",
        "confidence": 0.98,
        "reason": "Standard email address pattern.",
    },
    "Phone Number": {
        "regex": r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\w)",
        "severity": "Medium",
        "confidence": 0.85,
        "reason": "Indian mobile number pattern, optionally prefixed with country code.",
    },
    "Credit Card Number": {
        "regex": r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)",
        "severity": "High",
        "confidence": 0.75,
        "reason": "13 to 19 digit payment-card-like sequence validated with Luhn when possible.",
    },
    "IFSC Code": {
        "regex": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
        "severity": "High",
        "confidence": 0.95,
        "reason": "Indian bank IFSC code pattern.",
    },
    "Bank Account Number": {
        "regex": r"(?i)\b(?:account|acct|a/c|bank account)\s*(?:number|no|#)?\s*[:\-]?\s*([0-9]{9,18})\b",
        "severity": "High",
        "confidence": 0.88,
        "reason": "Bank account label followed by a 9 to 18 digit number.",
    },
    "API Key": {
        "regex": r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?",
        "severity": "High",
        "confidence": 0.9,
        "reason": "Credential label followed by a long token.",
    },
    "Password": {
        "regex": r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^'\"\s,;]{6,})['\"]?",
        "severity": "High",
        "confidence": 0.86,
        "reason": "Password-like label with a non-trivial value.",
    },
    "Employee ID": {
        "regex": r"\b(?:EMP|EID|Employee\s*ID)[-:\s]?[A-Z0-9]{3,12}\b",
        "severity": "Medium",
        "confidence": 0.82,
        "reason": "Common employee identifier prefix or label.",
    },
    "Prompt Injection Attempt": {
        "regex": r"(?i)\b(?:ignore\s+(?:all\s+)?previous\s+instructions|reveal\s+(?:the\s+)?system\s+prompt|developer\s+message|jailbreak|bypass\s+(?:the\s+)?(?:policy|safety)|disable\s+(?:the\s+)?(?:guardrails|safety)|exfiltrate|send\s+(?:the\s+)?(?:secrets|data)|output\s+(?:the\s+)?(?:api\s+key|password|token))\b",
        "severity": "High",
        "confidence": 0.91,
        "reason": "AI guardrail detector found prompt-injection or jailbreak language.",
    },
}


CONFIDENTIAL_TERMS = {
    "confidential": "Confidential Business Information",
    "internal use only": "Confidential Business Information",
    "trade secret": "Confidential Business Information",
    "nda": "Confidential Business Information",
    "non disclosure": "Confidential Business Information",
    "salary": "Confidential Business Information",
    "pricing strategy": "Confidential Business Information",
    "merger": "Confidential Business Information",
    "acquisition": "Confidential Business Information",
    "board meeting": "Confidential Business Information",
    "proprietary": "Confidential Business Information",
}

ML_CONTEXT_PATTERNS: dict[str, dict[str, object]] = {
    "Person Name": {
        "regex": r"(?i)\b(?:full\s+name|employee\s+name|client\s+name|customer\s+name|candidate\s+name|contact\s+person)\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        "severity": "Medium",
        "confidence": 0.78,
        "reason": "AI/ML contextual entity detector identified a person-name field.",
    },
    "Date of Birth": {
        "regex": r"(?i)\b(?:dob|date\s+of\s+birth|birth\s+date)\s*[:\-]?\s*([0-3]?\d[\/\-.][01]?\d[\/\-.](?:19|20)\d{2})",
        "severity": "Medium",
        "confidence": 0.8,
        "reason": "AI/ML contextual entity detector identified a birth-date field.",
    },
    "Address / Location": {
        "regex": r"(?i)\b(?:address|office\s+location|home\s+address|billing\s+address)\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9,.\-\/\s]{12,140})",
        "severity": "Medium",
        "confidence": 0.76,
        "reason": "AI/ML contextual entity detector identified an address or location field.",
    },
    "Confidential Project Name": {
        "regex": r"(?i)\b(?:project\s+codename|code\s+name|deal\s+name|secret\s+project|internal\s+project)\s*[:\-]\s*([A-Za-z][A-Za-z0-9 _\-]{2,80})",
        "severity": "Medium",
        "confidence": 0.77,
        "reason": "AI/ML contextual entity detector identified confidential project language.",
    },
}


HIGH_RISK_CATEGORIES = {
    "Aadhaar Number",
    "PAN Number",
    "Credit Card Number",
    "Bank Account Number",
    "IFSC Code",
    "API Key",
    "Password",
    "Prompt Injection Attempt",
}

ML_ENTITY_LABELS = {
    "PERSON": "Person Name",
    "ORG": "Organization Name",
    "GPE": "Address / Location",
    "LOC": "Address / Location",
}

PATTERN_PRIORITY = [
    "Credit Card Number",
    "API Key",
    "Password",
    "Bank Account Number",
    "IFSC Code",
    "Aadhaar Number",
    "PAN Number",
    "Email Address",
    "Phone Number",
    "Employee ID",
    "Prompt Injection Attempt",
]


def detect_sensitive_data(text: str) -> list[Detection]:
    detections: list[Detection] = []
    occupied: list[tuple[int, int]] = []

    for category in PATTERN_PRIORITY:
        config = PATTERNS[category]
        pattern = re.compile(str(config["regex"]))
        for match in pattern.finditer(text):
            value, start, end = _extract_value(match)
            if not value or _overlaps(start, end, occupied):
                continue

            if category == "Aadhaar Number" and not _looks_like_aadhaar(value):
                continue
            if category == "Credit Card Number" and not _is_probable_card(value):
                continue

            occupied.append((start, end))
            detections.append(
                Detection(
                    category=category,
                    value=value.strip(),
                    start=start,
                    end=end,
                    severity=str(config["severity"]),
                    confidence=float(config["confidence"]),
                    reason=str(config["reason"]),
                )
            )

    for term, category in CONFIDENTIAL_TERMS.items():
        for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
            start, end = match.span()
            if _overlaps(start, end, occupied):
                continue
            occupied.append((start, end))
            detections.append(
                Detection(
                    category=category,
                    value=match.group(0),
                    start=start,
                    end=end,
                    severity="Medium",
                    confidence=0.74,
                    reason="Keyword suggests confidential or business-sensitive content.",
                )
            )

    for detection in _detect_ml_contextual_entities(text, occupied):
        occupied.append((detection.start, detection.end))
        detections.append(detection)

    return sorted(detections, key=lambda item: (item.start, item.category))


def classify_risk(detections: Iterable[Detection], text: str) -> RiskAssessment:
    detections = list(detections)
    if not detections:
        return RiskAssessment("Low Risk", 5, ["No configured sensitive-data patterns were detected."])

    category_counts = Counter(item.category for item in detections)
    high_count = sum(1 for item in detections if item.category in HIGH_RISK_CATEGORIES)
    medium_count = len(detections) - high_count
    unique_categories = len(category_counts)

    score = min(100, high_count * 18 + medium_count * 8 + unique_categories * 5)
    if len(text) > 20_000 and len(detections) > 20:
        score = min(100, score + 10)
    if any(item.category == "Prompt Injection Attempt" for item in detections):
        score = min(100, score + 25)

    reasons = [
        f"AI/ML ensemble found {len(detections)} sensitive item(s) across {unique_categories} category/categories.",
        f"{high_count} high-severity item(s) and {medium_count} medium-severity item(s) detected.",
    ]
    if any(item.category in {"API Key", "Password"} for item in detections):
        reasons.append("Secrets or credentials were found and should be rotated immediately.")
    if any(item.category in {"Aadhaar Number", "PAN Number"} for item in detections):
        reasons.append("Indian personal identifiers were found, increasing privacy and compliance risk.")
    if any(item.category in {"Credit Card Number", "Bank Account Number", "IFSC Code"} for item in detections):
        reasons.append("Financial data was found, increasing fraud and PCI/security exposure.")
    if any(item.category in {"Person Name", "Date of Birth", "Address / Location", "Organization Name"} for item in detections):
        reasons.append("ML/NLP entity detection found contextual personal or organizational data.")
    if any(item.category == "Prompt Injection Attempt" for item in detections):
        reasons.append("Prompt-injection language was detected and must be ignored by AI workflows.")

    if any(item.category == "Prompt Injection Attempt" for item in detections):
        level = "High Risk"
    elif score >= 65 or high_count >= 3:
        level = "High Risk"
    elif score >= 25 or high_count >= 1 or len(detections) >= 4:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    return RiskAssessment(level, score, reasons)


def build_summary(text: str, detections: list[Detection], risk: RiskAssessment) -> dict[str, list[str]]:
    category_counts = Counter(item.category for item in detections)
    profile = _document_profile(text)
    high_categories = sorted({item.category for item in detections if item.severity == "High"})
    medium_categories = sorted({item.category for item in detections if item.severity == "Medium"})
    excerpt = _document_excerpt(text)
    observations = [
        f"Document risk is classified as {risk.level} with a score of {risk.score}/100.",
        f"AI/ML detection found {len(detections)} sensitive item(s) in {len(category_counts)} category/categories.",
        f"Document context appears to be {profile}.",
    ]
    if category_counts:
        top_categories = ", ".join(f"{category} ({count})" for category, count in category_counts.most_common(6))
        observations.append(f"Most prominent sensitive data types: {top_categories}.")
        if high_categories:
            observations.append(f"High-priority categories requiring review: {', '.join(high_categories)}.")
        if excerpt:
            observations.append(f"Context cue from document: {excerpt}.")
    else:
        observations.append("No configured sensitive-data indicators were identified.")

    risks = []
    if any(item.category in {"Aadhaar Number", "PAN Number", "Email Address", "Phone Number"} for item in detections):
        risks.append("Personal data exposure may trigger privacy, consent, retention, and access-control obligations.")
    if any(item.category in {"Person Name", "Date of Birth", "Address / Location"} for item in detections):
        risks.append("ML-detected contextual personal data may require privacy review even when structured IDs are absent.")
    if any(item.category in {"Credit Card Number", "Bank Account Number", "IFSC Code"} for item in detections):
        risks.append("Financial information could be misused for fraud if shared or stored insecurely.")
    if any(item.category in {"API Key", "Password"} for item in detections):
        risks.append("Credential exposure can lead to unauthorized system or data access.")
    if any(item.category == "Prompt Injection Attempt" for item in detections):
        risks.append("Document contains adversarial instructions that could manipulate AI responses if not isolated.")
    if any(item.category == "Confidential Business Information" for item in detections):
        risks.append("Business-sensitive language suggests possible leakage of internal or proprietary information.")
    if "legal" in profile or "contract" in profile:
        risks.append("Legal or contractual material may require tighter sharing controls and retention review.")
    if "credential" in profile or any(item.category in {"API Key", "Password"} for item in detections):
        risks.append("Technical access details can create immediate account-takeover or service-abuse risk.")
    if "hr" in profile:
        risks.append("Employee or onboarding data may create workforce privacy and internal-access obligations.")
    if not risks:
        risks.append("Residual risk remains because automated scans can miss contextual or image-only sensitive data.")

    remediation = []
    if any(item.category in {"API Key", "Password"} for item in detections):
        remediation.append("Rotate exposed credentials and invalidate leaked tokens.")
    if any(item.category == "Prompt Injection Attempt" for item in detections):
        remediation.append("Treat embedded AI instructions as untrusted content and exclude them from control prompts.")
    if any(item.category in {"Credit Card Number", "Bank Account Number", "IFSC Code"} for item in detections):
        remediation.append("Limit finance-data access to approved users and create a redacted payment copy.")
    if any(item.category in {"Aadhaar Number", "PAN Number"} for item in detections):
        remediation.append("Mask national identifiers and verify purpose, consent, and retention requirements.")
    if any(item.category == "Confidential Business Information" for item in detections):
        remediation.append("Classify the document as confidential and restrict external distribution.")
    remediation.extend(
        [
            "Mask or redact sensitive fields before sharing the document.",
            "Restrict access using least-privilege permissions and maintain an access audit trail.",
            "Store uploaded documents only as long as needed and delete temporary copies after review.",
        ]
    )
    if "pdf" in text[:200].lower():
        remediation.append("For scanned PDFs, add OCR validation before relying on this result.")

    return {
        "Compliance observations": observations,
        "Security risks": risks,
        "Suggested remediation steps": remediation,
    }


def _document_profile(text: str) -> str:
    lowered = text.lower()
    signals = {
        "finance/payment document": ["invoice", "payment", "bank", "ifsc", "account", "vendor", "card"],
        "legal/contract document": ["nda", "agreement", "contract", "legal", "attorney", "clause"],
        "credential/security document": ["api key", "password", "token", "secret", "access", "credential"],
        "hr/employee document": ["employee", "onboarding", "payroll", "salary", "hr", "candidate"],
        "business-confidential document": ["confidential", "proprietary", "trade secret", "board", "strategy"],
    }
    scores = {
        label: sum(lowered.count(term) for term in terms)
        for label, terms in signals.items()
    }
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return best_label if best_score else "a general business document"


def _document_excerpt(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    sentence_match = re.search(r"(.{40,220}?[.!?])(?:\s|$)", cleaned)
    excerpt = sentence_match.group(1) if sentence_match else cleaned[:180]
    return excerpt[:220]


def answer_question(question: str, text: str, detections: list[Detection], risk: RiskAssessment) -> str:
    normalized = question.lower().strip()
    category_counts = Counter(item.category for item in detections)

    if not normalized:
        return "Ask a question about the uploaded document or the detected sensitive data."

    if "how many" in normalized or "count" in normalized:
        matched_category = _match_category(normalized, category_counts)
        if matched_category:
            return f"{category_counts[matched_category]} {matched_category.lower()} item(s) were found."
        return f"{len(detections)} sensitive item(s) were found in total."

    if "what sensitive" in normalized or "sensitive data" in normalized or "exists" in normalized:
        if not category_counts:
            return "No configured sensitive-data types were detected."
        return "Detected categories: " + ", ".join(
            f"{category} ({count})" for category, count in category_counts.most_common()
        )

    if "compliance" in normalized or "risk" in normalized:
        summary = build_summary(text, detections, risk)
        return " ".join(summary["Compliance observations"] + summary["Security risks"])

    if "summarize" in normalized or "summary" in normalized:
        return _summarize_text(text, detections, risk)

    if "remediate" in normalized or "remediation" in normalized or "first" in normalized:
        summary = build_summary(text, detections, risk)
        steps = summary["Suggested remediation steps"][:4]
        return "Recommended priority order: " + " ".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))

    if "audit note" in normalized or "audit" in normalized:
        top_categories = ", ".join(f"{category} ({count})" for category, count in category_counts.most_common(5)) or "none"
        return (
            f"Audit note: Document classified as {risk.level} with score {risk.score}/100. "
            f"{len(detections)} sensitive item(s) detected across {len(category_counts)} category/categories: {top_categories}. "
            "Recommended action: restrict access, redact sensitive values, and record owner approval before sharing."
        )

    if "redact" in normalized or "mask" in normalized:
        return "Use the redacted preview/download in the app. It masks detected values while preserving document context."

    if "prompt injection" in normalized or "jailbreak" in normalized or "guardrail" in normalized:
        count = category_counts.get("Prompt Injection Attempt", 0)
        if count:
            return f"{count} prompt-injection or jailbreak signal(s) were detected. Treat them as untrusted document content and do not follow embedded instructions."
        return "No prompt-injection or jailbreak signals were detected in the configured scanner."

    matched_category = _match_category(normalized, category_counts)
    if matched_category:
        count = category_counts[matched_category]
        examples = [mask_value(item.value) for item in detections if item.category == matched_category][:5]
        return f"{count} {matched_category.lower()} item(s) were found. Examples: {', '.join(examples)}."

    return (
        "I can answer questions about detected data types, counts, risk level, compliance concerns, "
        "remediation, and a short document summary."
    )


def redact_text(text: str, detections: list[Detection]) -> str:
    redacted = []
    cursor = 0
    for item in sorted(detections, key=lambda detection: detection.start):
        if item.start < cursor:
            continue
        redacted.append(text[cursor:item.start])
        redacted.append(f"[REDACTED {item.category.upper()}]")
        cursor = item.end
    redacted.append(text[cursor:])
    return "".join(redacted)


def mask_value(value: str) -> str:
    value = value.strip()
    if len(value) <= 4:
        return "*" * len(value)
    visible = min(4, math.ceil(len(value) * 0.25))
    return f"{value[:2]}{'*' * max(4, len(value) - visible)}{value[-2:]}"


def _extract_value(match: re.Match[str]) -> tuple[str, int, int]:
    if match.lastindex:
        for index in range(1, match.lastindex + 1):
            value = match.group(index)
            if value:
                return value, match.start(index), match.end(index)
    return match.group(0), match.start(), match.end()


def _detect_ml_contextual_entities(text: str, occupied: list[tuple[int, int]]) -> list[Detection]:
    detections: list[Detection] = []

    for category, config in ML_CONTEXT_PATTERNS.items():
        pattern = re.compile(str(config["regex"]))
        for match in pattern.finditer(text):
            value, start, end = _extract_value(match)
            value = _clean_ml_value(value)
            if not value or _overlaps(start, end, occupied) or _overlaps(start, end, [(item.start, item.end) for item in detections]):
                continue
            detections.append(
                Detection(
                    category=category,
                    value=value,
                    start=start,
                    end=end,
                    severity=str(config["severity"]),
                    confidence=float(config["confidence"]),
                    reason=str(config["reason"]),
                )
            )

    nlp = _load_spacy_ner()
    if nlp is None:
        return detections

    try:
        doc = nlp(text[:100_000])
    except Exception:
        return detections

    existing_ranges = occupied + [(item.start, item.end) for item in detections]
    for ent in doc.ents:
        category = ML_ENTITY_LABELS.get(ent.label_)
        value = ent.text.strip()
        if not category or not value:
            continue
        start, end = ent.start_char, ent.end_char
        if len(value) < 3 or _overlaps(start, end, existing_ranges):
            continue
        if category == "Organization Name" and value.lower() in {"pdf", "csv", "txt"}:
            continue
        detections.append(
            Detection(
                category=category,
                value=value,
                start=start,
                end=end,
                severity="Medium",
                confidence=0.72,
                reason=f"spaCy NER model identified {ent.label_} entity in document context.",
            )
        )
        existing_ranges.append((start, end))

    return detections


@lru_cache(maxsize=1)
def _load_spacy_ner():
    try:
        import os
        import spacy

        return spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
    except Exception:
        return None


def _clean_ml_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n.,;")
    stop_markers = [" email", " phone", " pan", " aadhaar", " account", " password", " api"]
    lowered = cleaned.lower()
    for marker in stop_markers:
        index = lowered.find(marker)
        if index > 0:
            cleaned = cleaned[:index].strip(" \t\r\n.,;")
            break
    return cleaned[:160]


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in ranges)


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def _looks_like_aadhaar(value: str) -> bool:
    digits = _digits_only(value)
    return len(digits) == 12 and not digits.startswith(("0", "1")) and len(set(digits)) > 1


def _is_probable_card(value: str) -> bool:
    digits = _digits_only(value)
    if not 13 <= len(digits) <= 19:
        return False
    if len(set(digits)) == 1:
        return False
    return _luhn_valid(digits)


def _luhn_valid(digits: str) -> bool:
    total = 0
    reverse_digits = digits[::-1]
    for index, char in enumerate(reverse_digits):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def _match_category(question: str, category_counts: Counter[str]) -> str | None:
    aliases = {
        "aadhaar": "Aadhaar Number",
        "adhar": "Aadhaar Number",
        "pan": "PAN Number",
        "email": "Email Address",
        "phone": "Phone Number",
        "mobile": "Phone Number",
        "credit": "Credit Card Number",
        "card": "Credit Card Number",
        "bank": "Bank Account Number",
        "ifsc": "IFSC Code",
        "api": "API Key",
        "token": "API Key",
        "password": "Password",
        "employee": "Employee ID",
        "confidential": "Confidential Business Information",
    }
    for alias, category in aliases.items():
        if alias in question and category in category_counts:
            return category
    return None


def _summarize_text(text: str, detections: list[Detection], risk: RiskAssessment) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    first_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    excerpt = " ".join(first_sentences[:3])[:600]
    if not excerpt:
        excerpt = "The uploaded document contains little or no extractable text."
    return (
        f"{excerpt} Risk classification: {risk.level}. "
        f"Sensitive items detected: {len(detections)}."
    )
