"""
Curator — Observe Layer
The first primitive: holistic, passive, continuous.

Everything that passes through an interaction is automatically a candidate
for future relevance. Nothing is explicitly saved. The system watches
continuously and passively.

Flow:
1. Evaluate Signal Quality (instantaneous, fresh, discarded after use)
2. Embed the input
3. Find semantically related conceptions
4. For each related: classify as confirming or contradicting
5. Confirming → update weight upward
6. Contradicting → create competing conception alongside existing
7. No match → create new conception with low confidence, high recency
8. Log the observation
"""

import json
import os
import math
import hashlib
import time
import anthropic
from schema import (
    connect,
    SignalQuality,
    create_conception,
    update_weight,
    find_related_conceptions,
    get_conception,
    log_observation,
    INITIAL_CONFIDENCE,
    EXPLICIT_INSTRUCTION_MAGNITUDE,
)

MOCK_MODE = os.environ.get("CURATOR_MOCK", "false").lower() == "true"
_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=_api_key) if _api_key else None


# --- Embedding ---

def embed(text: str) -> list[float]:
    """Get embedding. Uses mock when no API key available."""
    return _mock_embed(text)  # real embedding API TBD


def _mock_embed(text: str) -> list[float]:
    """
    Mock embedding using character-level features.
    Produces 1024-dim normalized vector.
    Words that share characters will have some similarity — enough to test the pipeline.
    Replace with real embedding API in production.
    """
    seed = int(hashlib.md5(text.lower().encode()).hexdigest(), 16)
    vec = []
    for i in range(1024):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        vec.append((seed / 0xFFFFFFFF) * 2 - 1)
    magnitude = math.sqrt(sum(x * x for x in vec))
    return [x / magnitude for x in vec]


# --- Signal Quality ---

def evaluate_signal_quality(text: str) -> SignalQuality:
    """
    Instantaneous evaluation of incoming input clarity.
    Not accumulated — evaluated fresh each time, then discarded.
    Gates Surface decisions: ambiguous input → surface ambiguity, not resolution.
    """
    if MOCK_MODE or not client:
        return _mock_signal_quality(text)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""You evaluate the clarity and specificity of user input for a memory system.

Return ONLY a JSON object with:
- score: float 0.0 to 1.0 (0 = completely ambiguous, 1 = crystal clear)
- reason: one short sentence explaining the score

Examples:
- "I like coffee" → {"score": 0.85, "reason": "Clear preference statement with specific subject"}
- "yeah that" → {"score": 0.1, "reason": "No referent, entirely context-dependent"}
- "I changed my mind about the project deadline" → {"score": 0.7, "reason": "Clear change signal but prior state unknown"}

Return only the JSON, no other text.""",
        messages=[{"role": "user", "content": f"Evaluate: {text}"}]
    )
    try:
        result = json.loads(response.content[0].text)
        return SignalQuality(score=result["score"], reason=result["reason"])
    except Exception:
        return SignalQuality(score=0.5, reason="Could not evaluate signal quality")


def _mock_signal_quality(text: str) -> SignalQuality:
    """Mock signal quality based on simple heuristics."""
    words = text.strip().split()
    vague = {"yeah", "that", "it", "thing", "stuff", "this", "those", "them"}
    if len(words) <= 2 and any(w.lower() in vague for w in words):
        return SignalQuality(score=0.1, reason="Very short, vague referents only")
    if len(words) < 4:
        return SignalQuality(score=0.35, reason="Short input, limited specificity")
    if any(w.lower() in vague for w in words[:3]):
        return SignalQuality(score=0.45, reason="Starts with vague referent")
    return SignalQuality(score=0.8, reason="Clear, specific statement")


# --- Signal Classification ---

def classify_signal(existing_content: str, new_observation: str) -> dict:
    """
    Given an existing conception and a new observation, classify the relationship.
    Returns type, confidence_delta, is_explicit_instruction, reasoning.
    """
    if MOCK_MODE or not client:
        return _mock_classify(existing_content, new_observation)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="""You classify the relationship between an existing belief and a new observation.

Return ONLY a JSON object with:
- type: "confirming", "contradicting", or "unrelated"
- confidence_delta: float between -1.0 and 1.0
- is_explicit_instruction: true if the new observation is a direct user correction
- reasoning: one sentence

Rules:
- Small confirming signals: +0.05 to +0.1
- Strong confirming: +0.15 to +0.3
- Mild contradiction: -0.05 to -0.15
- Strong contradiction: -0.2 to -0.4
- Explicit correction: -0.8 to -1.0
- Contradicting observations create a competing conception automatically

Return only the JSON, no other text.""",
        messages=[{
            "role": "user",
            "content": f"Existing: {existing_content}\n\nNew: {new_observation}"
        }]
    )
    try:
        return json.loads(response.content[0].text)
    except Exception:
        return {"type": "unrelated", "confidence_delta": 0.0,
                "is_explicit_instruction": False, "reasoning": "Classification failed"}


def _mock_classify(existing: str, new_obs: str) -> dict:
    """Mock classification using keyword heuristics."""
    contradiction_words = {"actually", "no", "wrong", "instead", "changed", "not", "never", "but"}
    confirming_words = {"yes", "definitely", "exactly", "right", "indeed", "always", "still"}

    new_lower = new_obs.lower()
    existing_lower = existing.lower()

    # Check for shared key nouns (crude semantic overlap)
    existing_words = set(existing_lower.split())
    new_words = set(new_lower.split())
    overlap = existing_words & new_words - {"i", "am", "is", "the", "a", "an", "to", "my", "and"}

    if not overlap:
        return {"type": "unrelated", "confidence_delta": 0.0,
                "is_explicit_instruction": False, "reasoning": "No semantic overlap detected"}

    explicit = any(w in new_lower for w in {"actually", "no,", "that's wrong", "i changed"})

    if any(w in new_lower for w in contradiction_words):
        delta = -0.8 if explicit else -0.2
        return {"type": "contradicting", "confidence_delta": delta,
                "is_explicit_instruction": explicit,
                "reasoning": "Contradiction keywords detected with shared subject"}

    if any(w in new_lower for w in confirming_words) or overlap:
        return {"type": "confirming", "confidence_delta": 0.1,
                "is_explicit_instruction": False,
                "reasoning": "Confirming signal with shared subject"}

    return {"type": "unrelated", "confidence_delta": 0.0,
            "is_explicit_instruction": False, "reasoning": "No clear relationship"}


# --- Extract Conceptions ---

def extract_conceptions(text: str, signal_quality: SignalQuality) -> list[str]:
    """Extract atomic conceptions from raw input."""
    if signal_quality.score < 0.3:
        return []

    if MOCK_MODE or not client:
        return _mock_extract(text, signal_quality)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=f"""Extract atomic conceptions from user input for a personal context system.

A conception is a single, standalone belief, preference, fact, or state about the person or context.
Signal quality: {signal_quality.score:.2f} ({signal_quality.reason})

Rules:
- Each conception is self-contained and specific
- Phrase as third-person: "User prefers...", "User is working on...", "User believes..."
- Lower signal quality → extract less, be more conservative
- If signal < 0.5, max 1-2 conceptions
- Return empty list if nothing clear

Return ONLY a JSON array of strings. No other text.""",
        messages=[{"role": "user", "content": text}]
    )
    try:
        result = response.content[0].text.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        return json.loads(result)
    except Exception:
        return []


def _mock_extract(text: str, signal_quality: SignalQuality) -> list[str]:
    """Mock extraction — wraps the input as a simple conception."""
    if signal_quality.score < 0.5:
        return [f"User mentioned: {text.strip()}"]
    return [f"User stated: {text.strip()}"]


# --- Batch Prefill (fast path) ---

def batch_evaluate_signal_quality(texts: list[str]) -> list[SignalQuality]:
    """Evaluate signal quality for N messages in one API call."""
    if MOCK_MODE or not client:
        return [_mock_signal_quality(t) for t in texts]

    numbered = "\n".join(f"{i+1}. {t[:300]}" for i, t in enumerate(texts))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system="""You evaluate the clarity and specificity of multiple user inputs for a memory system.

Return ONLY a JSON array with one object per input:
[{"score": 0.0-1.0, "reason": "one sentence"}, ...]

Score guide:
- 0.0-0.2: vague, no-context ("yeah that", "ok", "hmm")
- 0.3-0.5: short/ambiguous but has some content
- 0.6-0.8: clear preference or fact
- 0.9-1.0: explicit, specific, self-contained

Return only the JSON array, no other text.""",
        messages=[{"role": "user", "content": f"Evaluate these inputs:\n{numbered}"}]
    )
    try:
        result = json.loads(response.content[0].text.strip())
        return [SignalQuality(score=r["score"], reason=r["reason"]) for r in result]
    except Exception:
        return [SignalQuality(score=0.5, reason="batch eval failed") for _ in texts]


def batch_extract_conceptions(texts: list[str], signal_qualities: list[SignalQuality]) -> list[list[str]]:
    """Extract conceptions from N messages in one API call."""
    if MOCK_MODE or not client:
        return [_mock_extract(t, sq) for t, sq in zip(texts, signal_qualities)]

    # Only include messages above threshold
    items = []
    for i, (text, sq) in enumerate(zip(texts, signal_qualities)):
        if sq.score >= 0.3:
            items.append(f"{i+1}. [score={sq.score:.2f}] {text[:300]}")

    if not items:
        return [[] for _ in texts]

    numbered = "\n".join(items)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system="""Extract atomic conceptions from multiple user inputs for a personal context system.

A conception is a single, standalone belief, preference, fact, or state about the person.
- Phrase as third-person: "User prefers...", "User is working on...", "User uses..."
- Each conception is self-contained and specific
- Lower score → extract less (score < 0.5: max 1 conception; score >= 0.7: up to 3)
- Skip inputs that contain no extractable facts

Return ONLY a JSON object mapping input number to array of conception strings:
{"1": ["conception a", "conception b"], "2": [], "3": ["conception c"]}

Return only the JSON, no other text.""",
        messages=[{"role": "user", "content": f"Extract from these inputs:\n{numbered}"}]
    )
    try:
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        return [result.get(str(i+1), []) for i in range(len(texts))]
    except Exception:
        return [[] for _ in texts]


def batch_observe(conn, texts: list[str], source: str = "prefill", batch_size: int = 20) -> dict:
    """
    Fast-path observe for prefill. Skips per-conception classification.
    All new conceptions created with low initial confidence — weight updates happen
    through normal interactions after prefill.

    Returns summary stats.
    """
    total_created = 0
    total_skipped = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # 1 call: signal quality for whole batch
        signal_qualities = batch_evaluate_signal_quality(batch)

        # 1 call: extract conceptions for whole batch
        all_conceptions = batch_extract_conceptions(batch, signal_qualities)

        # No classification — just store everything with low confidence
        for text, sq, conceptions in zip(batch, signal_qualities, all_conceptions):
            if not conceptions:
                total_skipped += 1
                continue
            for conception_text in conceptions:
                if not conception_text.strip():
                    continue
                embedding = embed(conception_text)
                create_conception(conn, conception_text, embedding, source,
                                  initial_confidence=INITIAL_CONFIDENCE)
                total_created += 1

        log_observation(conn, f"[batch:{len(batch)}]", 0.7, [])

    return {"created": total_created, "skipped": total_skipped}


# --- Core Observe Function ---

def observe(conn, text: str, source: str = "conversation") -> dict:
    """
    The Observe primitive.
    Takes raw input, processes it through the full observation pipeline,
    and updates the conception space accordingly.
    """
    result = {
        "input": text,
        "signal_quality": None,
        "conceptions_extracted": [],
        "actions": []
    }

    sq = evaluate_signal_quality(text)
    result["signal_quality"] = {"score": sq.score, "reason": sq.reason}

    conceptions_text = extract_conceptions(text, sq)
    result["conceptions_extracted"] = conceptions_text

    if not conceptions_text:
        log_observation(conn, text, sq.score, [])
        return result

    affected_ids = []

    for conception_text in conceptions_text:
        embedding = embed(conception_text)
        related = find_related_conceptions(conn, embedding)

        if not related:
            cid = create_conception(conn, conception_text, embedding, source)
            affected_ids.append(cid)
            result["actions"].append({
                "conception": conception_text,
                "action": "created",
                "id": cid,
                "reason": "No related conception found"
            })
            continue

        for conception_id, similarity in related:
            existing = get_conception(conn, conception_id)
            if not existing:
                continue

            classification = classify_signal(existing.content, conception_text)

            if classification["type"] == "confirming":
                update_weight(conn, conception_id, classification["confidence_delta"])
                affected_ids.append(conception_id)
                result["actions"].append({
                    "conception": conception_text,
                    "action": "confirmed",
                    "id": conception_id,
                    "existing": existing.content,
                    "delta": classification["confidence_delta"],
                    "reason": classification.get("reasoning", "")
                })

            elif classification["type"] == "contradicting":
                # Weaken existing
                update_weight(conn, conception_id, classification["confidence_delta"])

                # Create competing conception alongside — coexistence, not overwrite
                new_confidence = 0.4 if classification.get("is_explicit_instruction") else INITIAL_CONFIDENCE
                new_id = create_conception(conn, conception_text, embedding, source,
                                           initial_confidence=new_confidence)
                affected_ids.extend([conception_id, new_id])
                result["actions"].append({
                    "conception": conception_text,
                    "action": "competing_conception_created",
                    "existing_id": conception_id,
                    "existing": existing.content,
                    "new_id": new_id,
                    "delta_on_existing": classification["confidence_delta"],
                    "explicit_instruction": classification.get("is_explicit_instruction", False),
                    "reason": classification.get("reasoning", "")
                })

            else:
                new_id = create_conception(conn, conception_text, embedding, source)
                affected_ids.append(new_id)
                result["actions"].append({
                    "conception": conception_text,
                    "action": "created",
                    "id": new_id,
                    "reason": "Semantically related but conceptually unrelated"
                })

    log_observation(conn, text, sq.score, affected_ids)
    return result


if __name__ == "__main__":
    import sys
    # Run in mock mode for testing without API key
    os.environ["CURATOR_MOCK"] = "true"

    conn = connect("test_curator.db")
    print("=== Observe Layer Test (mock mode) ===\n")

    tests = [
        "I really prefer working late at night, usually after 10pm",
        "Yeah I'm definitely a night owl, mornings are rough",
        "Actually I've been trying to switch to mornings lately",
        "yeah that thing",
    ]

    for i, text in enumerate(tests, 1):
        print(f"--- Input {i}: '{text}'")
        r = observe(conn, text)
        sq = r["signal_quality"]
        print(f"  Signal quality: {sq['score']:.2f} — {sq['reason']}")
        print(f"  Extracted: {r['conceptions_extracted']}")
        for action in r["actions"]:
            a = action["action"]
            if a == "created":
                print(f"  → Created conception #{action['id']}")
            elif a == "confirmed":
                print(f"  → Confirmed #{action['id']} (delta +{action['delta']})")
            elif a == "competing_conception_created":
                print(f"  → Weakened #{action['existing_id']} (delta {action['delta_on_existing']})")
                print(f"  → Created competing conception #{action['new_id']}")
        print()

    # Show final state
    print("=== Final Conception Space ===")
    rows = conn.execute(
        "SELECT id, content, recency, confidence FROM conceptions ORDER BY recency DESC"
    ).fetchall()
    for row in rows:
        print(f"  #{row[0]} conf={row[2]:.3f} rec={row[1]:.3f} | {row[3][:80]}")
