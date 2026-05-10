"""Negation-aware symptom extraction from free-text (PDF/OCR/clinical notes).

Bug this module exists to prevent:
    Naive substring matching emits a positive "blood cough" symptom when the
    source text actually says "no history of blood cough". The classic
    clinical-NLP fix is NegEx-style scoped negation detection (Chapman 2001).

Design:
    1. Build a phrase -> canonical-symptom map from configs/symptoms.yaml
       (display names + snake_case names + every synonym).
    2. Split the text into sentences.
    3. For each sentence, find every phrase match.
    4. For each match, scan the LEFT context within the sentence for a
       negation cue. Stop scanning at clause-breakers ("but", "however",
       "except", ";", etc.) so the negation does NOT leak across clauses.
    5. Return canonical names that were NOT negated; expose negated matches
       separately for transparency.

This module touches no existing code paths. It exposes a single function:
    extract_symptoms(text: str) -> dict
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pulsepoint_ai.core.config import get_symptoms

# Negation pre-cues — must appear within the negation window to the LEFT of
# the symptom phrase. Multi-word cues are matched as exact token sequences.
_NEGATION_CUES: tuple[str, ...] = (
    "no history of",
    "no sign of",
    "no signs of",
    "no evidence of",
    "no complaint of",
    "no complaints of",
    "no",
    "not",
    "non",
    "never",
    "denies",
    "denied",
    "denying",
    "without",
    "absent",
    "absence of",
    "ruled out",
    "rules out",
    "rule out",
    "negative for",
    "free of",
    "free from",
    "lacks",
    "lack of",
    "neither",
    "nor",
    "doesn't have",
    "does not have",
    "has no",
    "have no",
    "had no",
    "w/o",
)

# Clause-breakers that close the negation scope (so "no fever, but headache"
# does NOT mark headache as negated).
_TERMINATORS: tuple[str, ...] = (
    "but",
    "however",
    "although",
    "though",
    "except",
    "yet",
    "while",
    "whereas",
)

# Window (in tokens) within which a negation cue is allowed to scope.
_NEGATION_WINDOW_TOKENS = 6

# Sentence splitter. Keeps the splitter character out of the returned chunks.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;\n])\s+|(?<=[.!?;])\n+")


@dataclass(frozen=True)
class SymptomMatch:
    canonical: str       # e.g. "hemoptysis"
    phrase: str          # the exact text matched, e.g. "blood cough"
    negated: bool
    sentence: str        # the surrounding sentence for transparency


def _build_phrase_index() -> list[tuple[str, str]]:
    """Returns a list of (lowercased_phrase, canonical_name), sorted by phrase
    length descending so longest match wins (prevents 'cough' eating 'blood cough').
    """
    cfg = get_symptoms()
    pairs: list[tuple[str, str]] = []
    for entry in cfg.get("symptoms", []):
        canonical = entry["name"]
        # Always match the canonical snake_case (de-snaked + raw)
        pairs.append((canonical.replace("_", " ").lower(), canonical))
        pairs.append((canonical.lower(), canonical))
        if entry.get("display"):
            pairs.append((entry["display"].lower(), canonical))
        for syn in entry.get("synonyms", []) or []:
            pairs.append((syn.lower(), canonical))
    # Dedupe while preserving first-seen mapping
    seen: dict[str, str] = {}
    for phrase, canonical in pairs:
        if phrase and phrase not in seen:
            seen[phrase] = canonical
    # Sort descending by length so multi-word phrases match before substrings.
    return sorted(seen.items(), key=lambda kv: -len(kv[0]))


# Cache the index — symptoms.yaml is static at runtime.
_PHRASE_INDEX: list[tuple[str, str]] | None = None


def _phrase_index() -> list[tuple[str, str]]:
    global _PHRASE_INDEX
    if _PHRASE_INDEX is None:
        _PHRASE_INDEX = _build_phrase_index()
    return _PHRASE_INDEX


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _is_negated(sentence_lower: str, match_start: int) -> bool:
    """Scan the LEFT context within the sentence for a negation cue.

    Stop scanning when we hit a clause-terminator so negations don't leak
    across clauses. The scan is bounded by `_NEGATION_WINDOW_TOKENS` tokens.
    """
    left = sentence_lower[:match_start].strip()
    if not left:
        return False

    # Tokenize the left context (simple whitespace+punct split).
    left_tokens = re.findall(r"[a-z0-9'/]+", left)
    if not left_tokens:
        return False

    # Walk backwards up to the window size, but bail out on terminators first.
    window = left_tokens[-_NEGATION_WINDOW_TOKENS:]
    # If a terminator appears inside the window, drop everything to its left:
    for i in range(len(window) - 1, -1, -1):
        if window[i] in _TERMINATORS:
            window = window[i + 1:]
            break

    # Reconstruct windowed-left as a single phrase to match multi-word cues.
    windowed_phrase = " ".join(window)
    for cue in _NEGATION_CUES:
        # Cue must appear as a whole-word run at any position in the window.
        if re.search(rf"(?:^|\s){re.escape(cue)}(?:\s|$)", windowed_phrase):
            return True
    return False


def extract_symptoms(text: str) -> dict:
    """Extract canonical symptom names from free-text with negation scoping.

    Returns:
        {
          "symptoms":   list[str]            # canonical names, deduped, ordered by first appearance
          "negated":    list[str]            # canonical names found but inside a negation scope
          "matches":    list[SymptomMatch]   # full per-occurrence audit trail (as dicts)
        }
    """
    if not text or not text.strip():
        return {"symptoms": [], "negated": [], "matches": []}

    index = _phrase_index()
    text_lc = text.lower()

    # Track character spans we've already consumed (longest-match-wins).
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        for s, e in consumed:
            if start < e and end > s:
                return True
        return False

    sentences = _split_sentences(text)
    # Compute sentence offsets so we can locate each match's sentence.
    sentence_spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sent in sentences:
        idx = text.find(sent, cursor)
        if idx < 0:
            idx = cursor
        sentence_spans.append((idx, idx + len(sent), sent))
        cursor = idx + len(sent)

    matches: list[SymptomMatch] = []

    for phrase, canonical in index:
        # Word-boundary search across the full text.
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", re.IGNORECASE)
        for m in pat.finditer(text_lc):
            start, end = m.start(), m.end()
            if overlaps(start, end):
                continue
            consumed.append((start, end))

            # Find the sentence this match falls in.
            sentence = text[start:end]
            sentence_local_start = 0
            for s_start, s_end, s_text in sentence_spans:
                if s_start <= start < s_end:
                    sentence = s_text
                    sentence_local_start = start - s_start
                    break

            negated = _is_negated(sentence.lower(), sentence_local_start)
            matches.append(
                SymptomMatch(
                    canonical=canonical,
                    phrase=text[start:end],
                    negated=negated,
                    sentence=sentence,
                )
            )

    # Order matches by their appearance in the original text.
    matches.sort(key=lambda mm: text_lc.find(mm.phrase.lower()))

    seen_pos: set[str] = set()
    seen_neg: set[str] = set()
    symptoms: list[str] = []
    negated_list: list[str] = []
    for mm in matches:
        if mm.negated:
            if mm.canonical not in seen_neg and mm.canonical not in seen_pos:
                seen_neg.add(mm.canonical)
                negated_list.append(mm.canonical)
        else:
            if mm.canonical not in seen_pos:
                seen_pos.add(mm.canonical)
                symptoms.append(mm.canonical)
                # If this canonical was previously marked negated (a later
                # positive mention overrides), drop from negated.
                if mm.canonical in seen_neg:
                    negated_list = [n for n in negated_list if n != mm.canonical]
                    seen_neg.discard(mm.canonical)

    return {
        "symptoms": symptoms,
        "negated": negated_list,
        "matches": [
            {
                "canonical": mm.canonical,
                "phrase": mm.phrase,
                "negated": mm.negated,
                "sentence": mm.sentence,
            }
            for mm in matches
        ],
    }
