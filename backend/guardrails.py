"""
Guardrails run at three points in the pipeline:

  1. INPUT guard   - blocks unsafe/injection-y transcribed text before
                     it ever reaches retrieval.
  2. RETRIEVAL guard - if nothing relevant was retrieved (score below
                     threshold), refuse to answer rather than let the
                     generator hallucinate from nothing.
  3. OUTPUT guard  - checks the generated answer is actually grounded
                     in the retrieved chunks (lexical-overlap check) and
                     scrubs answers that look like the model ignored
                     context.

Each guard returns a GuardResult so the harness can log *which* guard
fired, not just that "something" was blocked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import settings

UNSAFE_PATTERNS = [
    r"\bignore (all|previous) instructions\b",
    r"\bsystem prompt\b",
    r"\bhow (do|to) (make|build|synthesize) (a )?(bomb|explosive|weapon)\b",
    r"\bhack (into|the)\b",
    r"\b(self.?harm|suicide) method\b",
]

_GROUNDING_STOPWORDS = {
    # English functional & interrogative stopwords
    "a", "an", "the", "and", "or", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "with", "from", "based", "retrieved", "passage", "context", "answer", "this",
    "that", "these", "those", "what", "which", "who", "whom", "where", "when", "why",
    "how", "cause", "causes", "reason", "reasons", "do", "does", "did", "done", "can",
    "could", "should", "would", "be", "been", "being", "have", "has", "had", "as",
    "at", "by", "if", "into", "it", "its", "no", "not", "such", "than", "too", "very",
    "tell", "me", "about", "describe", "explain",
    # Hindi functional & interrogative stopwords
    "का", "के", "की", "को", "में", "से", "पर", "है", "हैं", "था", "थी", "थे", "और", "या",
    "क्या", "क्यों", "कैसे", "कहाँ", "कब", "किस", "किसके", "किसलिए", "कारण", "होता", "होती",
    "होते", "हुए", "हुआ", "हुई", "कर", "करता", "करती", "करते", "एक", "यह", "वह", "ये",
    "वे", "भी", "तो", "ने", "तक", "लिए", "बारे", "बताओ", "समझाओ", "दीजिए",
    # Marathi functional & interrogative stopwords
    "चा", "ची", "चे", "च्या", "ला", "ना", "त", "मध्ये", "वरून", "आहे", "आहेत", "होता",
    "होती", "होते", "आणि", "किंवा", "काय", "कसे", "कुठे", "केव्हा", "कोणी", "कोणाचा",
    "कारणे", "झाला", "झाली", "झाले", "करा", "करणे", "एक", "हा", "ही", "हे", "ते",
    "सुद्धा", "पण", "पर्यंत", "बद्दल", "सांगा", "स्पष्ट",
}


@dataclass
class GuardResult:
    passed: bool
    guard_name: str
    reason: str = ""
    detail: dict | None = None


def check_input_safety(text: str) -> GuardResult:
    lowered = text.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            return GuardResult(
                passed=False,
                guard_name="input_safety",
                reason="Query matched an unsafe/prompt-injection pattern.",
                detail={"pattern": pattern},
            )
    if len(text.strip()) < 2:
        return GuardResult(
            passed=False,
            guard_name="input_safety",
            reason="Transcript is empty or too short to be a real query.",
        )
    return GuardResult(passed=True, guard_name="input_safety")


def check_retrieval_relevance(top_score: float, query: str = "", retrieved_text: str = "") -> GuardResult:
    """Stage 2 Multi-Factor Query <-> Retrieved Context Relevance Validator:
    
    1. Score Gate: Checks vector similarity score against minimum threshold.
    2. Subject & Content Entity Match: Extracts non-stopword content terms from the query.
       If the retrieved context does not contain the query's core subject entities/terms,
       it rejects the context rather than generating an off-topic answer.
    """
    if top_score < settings.MIN_RETRIEVAL_SCORE:
        return GuardResult(
            passed=False,
            guard_name="retrieval_relevance",
            reason=(
                f"I couldn't find sufficiently relevant information about '{query}' in the provided dataset, "
                "so I won't generate an unsupported answer."
            ),
            detail={"top_score": top_score, "threshold": settings.MIN_RETRIEVAL_SCORE, "check": "score_gate"},
        )

    # Extract substantive content terms (excluding generic question words and stopwords)
    query_raw_terms = _tokens(query)
    query_content_terms = query_raw_terms - _GROUNDING_STOPWORDS
    retrieved_terms = _tokens(retrieved_text)

    # If query contains substantive content terms, verify they actually appear in retrieved text
    if query_content_terms:
        matched_terms = query_content_terms & retrieved_terms
        overlap_ratio = len(matched_terms) / len(query_content_terms)
        
        # Must match at least 1 core content term and reach minimum content overlap
        if not matched_terms or overlap_ratio < 0.20:
            return GuardResult(
                passed=False,
                guard_name="retrieval_relevance",
                reason=(
                    f"I couldn't find sufficiently relevant information about '{query}' in the provided dataset, "
                    "so I won't generate an unsupported answer."
                ),
                detail={
                    "top_score": top_score,
                    "content_overlap_ratio": round(overlap_ratio, 3),
                    "missing_query_subjects": sorted(query_content_terms - retrieved_terms),
                    "check": "subject_relevance_validator",
                },
            )

    return GuardResult(
        passed=True,
        guard_name="retrieval_relevance",
        detail={"top_score": top_score, "matched_content_terms": sorted(query_content_terms & retrieved_terms) if query_content_terms else []},
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def check_answer_grounded(answer: str, context_chunks: list[str]) -> GuardResult:
    """Lexical-overlap grounding check: checks that the answer is grounded
    in the retrieved context. Supports bilingual entities, numerals, and direct tokens."""
    if not answer.strip():
        return GuardResult(False, "answer_grounded", "Empty answer.")

    answer_tokens = _tokens(answer)
    context_tokens = set()
    for c in context_chunks:
        context_tokens |= _tokens(c)

    if not answer_tokens:
        return GuardResult(False, "answer_grounded", "Answer had no scoreable tokens.")

    # Score only meaningful answer terms; otherwise boilerplate can make an
    # ungrounded answer appear supported.
    factual_answer_tokens = answer_tokens - _GROUNDING_STOPWORDS
    if not factual_answer_tokens:
        return GuardResult(False, "answer_grounded", "Answer contains no factual tokens to verify.")
    overlap = len(factual_answer_tokens & context_tokens) / len(factual_answer_tokens)
    
    # Check numbers/dates overlap (critical factual anchors)
    num_pat = re.compile(r"\b\d+\b")
    ans_nums = set(num_pat.findall(answer))
    ctx_nums = set(num_pat.findall(" ".join(context_chunks)))
    has_num_grounding = bool(ans_nums and (ans_nums & ctx_nums))

    # A number is supported only when it is also present in context; it never
    # bypasses the lexical grounding requirement by itself.
    if ans_nums and not has_num_grounding:
        return GuardResult(False, "answer_grounded", "Answer contains a number/date absent from retrieved context.", detail={"overlap": overlap})
    if overlap >= settings.GROUNDING_OVERLAP_THRESHOLD:
        return GuardResult(True, "answer_grounded", detail={"overlap": overlap, "numeric_grounding": has_num_grounding})

    return GuardResult(
        passed=False,
        guard_name="answer_grounded",
        reason=(
            f"Answer overlaps only {overlap:.2%} with retrieved context "
            f"(threshold {settings.GROUNDING_OVERLAP_THRESHOLD:.0%}) — "
            "flagged as potentially ungrounded / hallucinated."
        ),
        detail={"overlap": overlap},
    )


REFUSAL_MESSAGE = (
    "I couldn't find enough relevant information in the indexed corpus to "
    "answer that confidently, so I'm not going to guess. Try rephrasing, "
    "or ask something closer to the dataset's topic area."
)

UNSAFE_MESSAGE = "I can't help with that request."
